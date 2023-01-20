import numpy as np
import torch

from tests import _PATH_DATA

processed_path = f"{_PATH_DATA}/processed"


def test_len():
    for group, n in zip(["train", "val", "test"], [32450, 6243, 6243]):
        tokens = torch.load(f"{processed_path}/tokens_{group}.pt")
        labels = torch.load(f"{processed_path}/labels_{group}.pt")
        # Number
        assert (
            tokens.shape[1] == n
        ), "Dataset did not have the correct number of samples"
        assert (
            labels.shape[0] == n
        ), "Dataset did not have the correct number of samples"
    # Shape


def test_balance():
    # Label balance
    for group in ["train", "val", "test"]:
        labels = torch.load(f"{processed_path}/labels_{group}.pt").numpy()
        assert (np.mean(labels) < 0.7) and (np.mean(labels) > 0.3), "Data is unbalanced"
