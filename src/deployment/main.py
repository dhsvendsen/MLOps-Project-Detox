from src.models.model import LightningBert
import torch
import re
import string
import pickle
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
import hydra
from transformers import BertTokenizer


with open('../../models/latest_training_dict.pickle', 'rb') as handle:
    b = pickle.load(handle)

def load_model(cfg):
    model = LightningBert(cfg)
    state_dict = torch.load(cfg.predict['modelpath'])
    model.load_state_dict(state_dict)
    return model
my_model=load_model()

example ="extremely angry comment!!! wtf"
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Stemming or Lemmatization
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

proc= preprocess(example)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
out = tokenizer.encode_plus(proc, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
"""
preds=my_model(out['input_ids'],out['attention_mask'])
preds = torch.sigmoid(preds) 
"""
print(type(my_model))

