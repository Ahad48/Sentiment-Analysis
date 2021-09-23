import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
import contractions
import pickle
from keras.models import load_model

def preprocess_text(text:str, lemma:bool = True)->str:

    def get_pos(word:str)->str:

        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    # Give error if input is not string
    assert isinstance(text, str), "Input variable should be a string"
    
    # Remove the html content and return the clean text
    text = BeautifulSoup(text, "lxml").text
    
    # Remove \n and \t
    text = re.sub(r'\n\t','',text)
    
    # Remove email ids
    text = re.sub(r'\S*@\S*\s?','',text)
    
    # Remove urls
    pattern = r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b'
    text = re.sub(pattern,'', text)
    
    # Fix contractions
    text = contractions.fix(text)
    
    # Remove Punctuations
    text = re.sub(r'[^\w\s]','',text)
    
    # Remove number from the text
    text = re.sub(r'[0-9]','',text)
    
    # Remove extra spaces fromt he text
    text = re.sub(' +',' ',text)
    
    # Lowercase the text
    text = text.lower()
    
    if lemma:
        # Tokenize
        tokens = word_tokenize(text)

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word, get_pos(word)) for word in tokens]

        text = ' '.join(words)
    
    return text


def load_models(model_path:str):
    # Loading the saved model
    model = load_model(model_path)

    # Loading the label binarizer
    f = open("models/labelbinarizer.pkl","rb")
    lb = pickle.load(f)
    f.close()

    # Loading the tokenizer
    f = open("models/tokenizer.pkl","rb")
    tokenizer = pickle.load(f)
    f.close()

    return model, lb, tokenizer



