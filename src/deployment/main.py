from model import LightningBert
import torch
import string
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from transformers import BertTokenizer
from fastapi import FastAPI
from http import HTTPStatus
import regex as re
import os

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "cwd":os.getcwd()
    }
    return response

with open('latest_training_dict.pickle', 'rb') as handle:
    b = pickle.load(handle)

def load_model(b):
    model = LightningBert(b)
    state_dict = torch.load('detox_checkpoint1.pth')
    model.load_state_dict(state_dict)
    return model
my_model=load_model(b)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model.to(device)

@app.post("/text_model/")
def is_toxic(comment: str):
    # Remove punctuation
    text = comment.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Stemming or Lemmatization
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text.split()])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    out = tokenizer.encode_plus(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
    preds=my_model(out['input_ids'],out['attention_mask'])
    preds = torch.sigmoid(preds)
    #maybe this will error on GPU 
    preds=preds.tolist()

    response={
        "message": HTTPStatus.OK.phrase,
        "pred":preds
    }
    return response
