# -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download("stopwords")


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Load the test and train data
    test_text = pd.read_csv(os.path.join(input_filepath, "test.csv"))
    test_labels = pd.read_csv(os.path.join(input_filepath, "test_labels.csv"))
    train_data = pd.read_csv(os.path.join(input_filepath, "train.csv"))

    # replace -1 with 0 in test_labels
    test_labels = test_labels.replace(-1, 0)

    # Combine test data and labels
    test_data = pd.merge(test_text, test_labels, on="id")

    # Preprocessing
    def preprocess(text):
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
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
        return text

    test_data["comment_text"] = test_data["comment_text"].apply(preprocess)
    train_data["comment_text"] = train_data["comment_text"].apply(preprocess)

    # make a copy of the data
    test_df = test_data.copy(deep=True)
    train_df = train_data.copy(deep=True)

    # Tokenize the test and train data
    test_tokens = []
    for text in tqdm(test_data["comment_text"], desc="Tokenizing Test Data"):
        test_tokens.append(
            tokenizer.encode_plus(
                text,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt",
            )
        )
    train_tokens = []
    for text in tqdm(train_data["comment_text"], desc="Tokenizing Train Data"):
        train_tokens.append(
            tokenizer.encode_plus(
                text,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt",
            )
        )

    # Convert the tokens to their corresponding IDs
    test_ids = [t["input_ids"] for t in test_tokens]
    train_ids = [t["input_ids"] for t in train_tokens]

    test_mask = [t["attention_mask"] for t in test_tokens]
    train_mask = [t["attention_mask"] for t in train_tokens]

    test_ids = torch.cat(test_ids, dim=0)
    train_ids = torch.cat(train_ids, dim=0)
    test_mask = torch.cat(test_mask, dim=0)
    train_mask = torch.cat(train_mask, dim=0)

    test_data = torch.empty(2, 153164, 64)
    test_data[0] = test_ids
    test_data[1] = test_mask

    train_data = torch.empty(2, 159571, 64)
    train_data[0] = train_ids
    train_data[1] = train_mask

#    test_tensor = torch.cat((torch.tensor(test_ids), torch.tensor(test_mask)), dim=1)

#    train_tensor = torch.cat((torch.tensor(train_ids), torch.tensor(train_mask)), dim=1)

    test_split, val_split = torch.split(test_data, test_data.size(1) // 2, dim=1)

    # save torch tensors
    torch.save(test_split, os.path.join(output_filepath, "tokens_test.pt"))
    torch.save(val_split, os.path.join(output_filepath, "tokens_val.pt"))
    torch.save(train_data, os.path.join(output_filepath, "tokens_train.pt"))

    # convert  test_data.drop(columns=['comment_text']) to torch tensor
    labels_test = torch.tensor(test_df.drop(columns=["comment_text", "id"]).values)
    labels_test_split, labels_val_split = torch.split(
        labels_test, labels_test.size(0) // 2, dim=0
    )
    labels_train = torch.tensor(train_df.drop(columns=["comment_text", "id"]).values)

    torch.save(labels_test_split, os.path.join(output_filepath, "labels_test.pt"))
    torch.save(labels_val_split, os.path.join(output_filepath, "labels_val.pt"))
    torch.save(labels_train, os.path.join(output_filepath, "labels_train.pt"))

    # torch.save({'data': test_tensor, 'label': test_data.drop(columns=['comment_text'])},
    #           os.path.join(output_filepath, "test_preprocessed.pt"))
    # torch.save({'data': train_tensor, 'label': train_data.drop(columns=['comment_text'])},
    #           os.path.join(output_filepath, "train_preprocessed.pt"))


if __name__ == "__main__":
    input_filepath = "data/raw"
    output_filepath = "data/processed"
    main(input_filepath, output_filepath)
