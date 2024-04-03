from flask import Flask, request, jsonify
import nltk
from nltk.corpus import wordnet
import random
from functools import lru_cache
import os

# SYNONYM_REPLACEMENT_PROB = float(os.getenv("SYNONYM_REPLACEMENT_PROB", 0.4))
# NEUTRAL_SENTENCE_PROB = float(os.getenv("NEUTRAL_SENTENCE_PROB", 0.2))

def download_nltk_resources():
    resources = ["punkt", "wordnet", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            nltk.data.find('tokenizers/' + resource)
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

app = Flask(__name__)

@lru_cache(maxsize=None)  # Unlimited cache size. You can limit if required.
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
            if random.uniform(0, 1) < 0.20:  # 10% chance to replace a word
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

# def back_translate(paragraph):
#     return paragraph  # Implement your back translation here

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

@app.route('/process_paragraph', methods=['POST'])
def process_paragraph_endpoint():
    paragraph = request.json.get('paragraph')
    return jsonify({'result': process_paragraph(paragraph)})

if __name__ == '__main__':
    app.run(debug=True, port=7500)

