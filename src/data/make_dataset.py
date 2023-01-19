import os
import re
import string

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from transformers import BertTokenizer

nltk.download("stopwords")


def main(input_filepath: str, output_filepath: str):
    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the test and train data
    test_text = pd.read_csv(os.path.join(input_filepath, "test.csv"))
    test_labels = pd.read_csv(os.path.join(input_filepath, "test_labels.csv"))
    train_data = pd.read_csv(os.path.join(input_filepath, "train.csv"))

    # Combine test data and labels
    test_data = pd.merge(test_text, test_labels, how="inner", on="id")
    categories = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    # Filter out datapoints that are masked in the kaggle dataset
    good_indices = test_data[categories].sum(1) != -6
    test_data = test_data[good_indices]

    # Balance training data
    train_data["any"] = (train_data[categories].sum(1) > 0) * 1
    group_train = train_data.groupby("any")
    print(f"Size of trainingdata before balancing: {len(train_data)}")
    train_data = group_train.apply(
        lambda x: x.sample(group_train.size().min()).reset_index(drop=True)
    )
    print(f"Size of trainingdata after balancing: {len(train_data)}")

    # Balance test data
    test_data["any"] = (test_data[categories].sum(1) > 0) * 1
    group_test = test_data.groupby("any")
    print(f"Size of testdata before balancing: {len(test_data)}")
    test_data = group_test.apply(
        lambda x: x.sample(group_test.size().min()).reset_index(drop=True)
    )
    print(f"Size of testdata after balancing: {len(test_data)}")
    test_data = test_data.sample(frac=1)

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
                max_length=32,
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
                max_length=32,
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

    test_data = torch.empty(2, test_ids.shape[0], test_ids.shape[1])
    test_data[0] = test_ids
    test_data[1] = test_mask

    train_data = torch.empty(2, train_ids.shape[0], train_ids.shape[1])
    train_data[0] = train_ids
    train_data[1] = train_mask

    test_split, val_split = torch.split(test_data, test_data.shape[1] // 2, dim=1)

    # save torch tensors
    torch.save(test_split, os.path.join(output_filepath, "tokens_test.pt"))
    torch.save(val_split, os.path.join(output_filepath, "tokens_val.pt"))
    torch.save(train_data, os.path.join(output_filepath, "tokens_train.pt"))

    # convert  test_data.drop(columns=['comment_text']) to torch tensor
    # labels_test = torch.tensor(test_df.drop(columns=["comment_text", "id"]).values)
    labels_test = torch.tensor(test_df["any"].values).reshape(-1, 1)
    labels_test_split, labels_val_split = torch.split(
        labels_test, test_data.shape[1] // 2, dim=0
    )
    # labels_train = torch.tensor(train_df.drop(columns=["comment_text", "id"]).values)
    labels_train = torch.tensor(train_df["any"].values).reshape(-1, 1)

    torch.save(labels_test_split, os.path.join(output_filepath, "labels_test.pt"))
    torch.save(labels_val_split, os.path.join(output_filepath, "labels_val.pt"))
    torch.save(labels_train, os.path.join(output_filepath, "labels_train.pt"))


if __name__ == "__main__":
    input_filepath = "data/raw"
    output_filepath = "data/processed"
    main(input_filepath, output_filepath)
