import json
import time
import random
import nltk
from nltk.corpus import wordnet
from transformers import T5Tokenizer, T5ForConditionalGeneration
import schedule
import threading

nltk.download('punkt')
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
            if random.uniform(0, 1) < 0.33:  # 10% chance to replace a word
                synonyms = get_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def back_translate(paragraph):
    return paragraph  # Implement your back translation here

def add_neutral_sentence(paragraph):
    if random.uniform(0, 1) < 0.2:  # 20% chance to add a neutral sentence
        neutral_sentences = ["This is a fact that cannot be denied.", 
                             "It is clear from the evidence presented.", 
                             "This is an important point to consider.", 
                             "The data supports this conclusion.", 
                             "This is a widely accepted view."]
        return paragraph + " " + random.choice(neutral_sentences)
    else:
        return paragraph

def process_paragraph(paragraph):
    paragraph = synonym_replacement(paragraph)
    # paragraph = back_translate(paragraph)
    paragraph = add_neutral_sentence(paragraph)
    return paragraph

def process_data(data):
    revised_data = {}
    for key, paragraphs in data.items():
        revised_paragraphs = [process_paragraph(paragraph) for paragraph in paragraphs]
        revised_data[key] = revised_paragraphs
    return revised_data

def task_to_run_every_ten_seconds():
    # Load your data from the json file
    with open('grouped_answers0805.json') as f:
        data = json.load(f)

    # Process the data
    revised_data = process_data(data)

    # Save the response to a file
    with open('grouped_answers0806.json', 'w') as f:
        json.dump(revised_data, f, indent = 4)

    print("Task completed.")

def run_schedule():
    while 1:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    schedule.every(5).seconds.do(task_to_run_every_ten_seconds)
    t = threading.Thread(target=run_schedule)
    t.start()
