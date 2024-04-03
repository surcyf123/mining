import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchmetrics.functional import pairwise_cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import time
import itertools
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SimplifiedDiversityRewardModel:
    diversity_model_path = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.diversity_model_path)
        self.model = AutoModel.from_pretrained(self.diversity_model_path).to(self.device)
        self.reward_bottom_k = 2
        self.history_reward_bottom_k = 2
        self.historic_embeddings = torch.tensor([]).to(self.device)
        self.history_range = (500, 15500)
        
    def get_embeddings(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(embeddings, encoded_input["attention_mask"])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def update_historic_embeddings(self, embeddings):
        def unique(embeddings):
            unique_embeddings = [embeddings[0]]
            last_emb = embeddings[0]
            for emb in embeddings:
                if not torch.all(torch.eq(emb, last_emb)):
                    unique_embeddings.append(emb)
                last_emb = emb
            return torch.stack(unique_embeddings)
 
        embeddings_unique = unique(embeddings)
        historic_embeddings = torch.cat([self.historic_embeddings, embeddings_unique])
        self.historic_embeddings = historic_embeddings[-self.history_range[1]:, :]
    
    def get_historic_rewards(self, embeddings):
        def regularise(rewards):
            return 1 / (1 + torch.exp(-1000 * rewards + 50))

        if self.historic_embeddings.shape[0] < (self.history_range[0] + self.history_reward_bottom_k):
            return None
        
        similarity = pairwise_cosine_similarity(embeddings, self.historic_embeddings[self.history_range[0]:])
        rewards = torch.topk((1 - torch.abs(similarity)), self.history_reward_bottom_k, largest=False)[0][:, -1]
        return regularise(rewards) 

    def get_batch_rewards(self, embeddings):
        def regularise(rewards):
            return 1 / (1 + torch.exp(-40 * rewards + 4))

        similarity = pairwise_cosine_similarity(embeddings, embeddings)
        rewards = torch.topk((1 - torch.abs(similarity)), self.reward_bottom_k, largest=False)[0][:, -1]
        return regularise(rewards) 
    
    def get_rewards(self, completions):
        # logging.info("Calculating rewards for completions...")
        start_time = time.time()
        
        if not completions:
            return torch.tensor([]).to(self.device)
        
        embeddings = self.get_embeddings(completions)
        batch_rewards = self.get_batch_rewards(embeddings)
        historic_rewards = self.get_historic_rewards(embeddings)
        
        self.update_historic_embeddings(embeddings)
        
        end_time = time.time()
        # logging.info(f"Calculated rewards in {end_time - start_time} seconds.")
        
        return batch_rewards * historic_rewards if historic_rewards is not None else batch_rewards

def get_diversity_penalty(completions, device="cpu"):
    model = SimplifiedDiversityRewardModel(device=device)
    return model.get_rewards(completions)

def compute_combined_score_for_multiprocessing(completion, selected_completions, device):
    potential_set = selected_completions + [completion]
    completions_texts = [comp['completion'] for comp in potential_set]
    diversity_penalties = get_diversity_penalty(completions_texts, device=device)
    
    return sum(comp['score'] * penalty for comp, penalty in zip(potential_set, diversity_penalties))

def greedy_select_completions(completions_list, responses_needed, device="cpu"):
    """Select the best n completions using a greedy approach with parallel processing."""
    
    if not completions_list:
        logging.warning("No completions available for greedy selection.")
        return []

    completions_list.sort(key=lambda x: x['score'], reverse=True)
    selected_completions = [completions_list.pop(0)]

    while len(selected_completions) < responses_needed:
        with ProcessPoolExecutor() as executor:
            combined_scores = list(executor.map(compute_combined_score_for_multiprocessing, completions_list, 
                                                itertools.repeat(selected_completions), itertools.repeat(device)))
        
        best_idx = max(range(len(combined_scores)), key=combined_scores.__getitem__)
        selected_completions.append(completions_list.pop(best_idx))
    
    return selected_completions

# Test the function
test_completions = {
    "reward_score": ["who are you", "who are you"]
}
penalty = get_diversity_penalty(test_completions["reward_score"])
print(penalty)
