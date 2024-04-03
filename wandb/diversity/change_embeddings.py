import time
import random
import nltk
from nltk.corpus import wordnet
from transformers import T5Tokenizer, T5ForConditionalGeneration

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(paragraph):
    words = nltk.word_tokenize(paragraph)
    pos_tags = nltk.pos_tag(words)
    new_words = []
    for word, pos_tag in pos_tags:
        if pos_tag[0].lower() in ['n', 'v']:  # replace nouns and verbs
            synonyms = get_synonyms(word)
            if synonyms:
                new_words.append(random.choice(synonyms))
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def add_random_noise(paragraph):
    words = nltk.word_tokenize(paragraph)
    random.shuffle(words)
    return ' '.join(words)

def paraphrase(paragraph):
    text = "paraphrase: " + paragraph
    max_len = len(text) + 20
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks, 
        max_length=max_len, temperature=1.5, num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def sentence_shuffle(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    random.shuffle(sentences)
    return ' '.join(sentences)

def process_paragraphs(paragraphs, operation):
    start = time.time()
    new_paragraphs = [operation(paragraph) for paragraph in paragraphs]
    end = time.time()
    print(f"Time taken for {operation.__name__}: {end - start} seconds")
    return new_paragraphs

def main(paragraphs):
    operations = [synonym_replacement, add_random_noise, paraphrase, sentence_shuffle]
    for operation in operations:
        new_paragraphs = process_paragraphs(paragraphs, operation)
        print(new_paragraphs[0])  # print the first paragraph as an example

paragraphs = ["Your first paragraph here.", "Your second paragraph here."]  # replace with your paragraphs
main(paragraphs)


def process():
    data = request.get_json()
    revised_data = {}
    for key, paragraph in data.items():
        revised_paragraphs = {}
        for operation in [synonym_replacement, add_random_noise, paraphrase, sentence_shuffle]:
            revised_paragraphs[operation.__name__] = process_paragraph(paragraph, operation)
        revised_data[key] = revised_paragraphs
    return jsonify(revised_data)

if __name__ == '__main__':
    app.run(debug=True)