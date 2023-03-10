import json

import torch
from model import LightningBertBinary
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if torch.has_cuda:
    device = "cuda"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} accelerator")


def predict():
    # load config
    with open("config/config.json") as file:
        cfg = json.load(file)

    model = LightningBertBinary(cfg)
    state_dict = torch.load(cfg["paths"]["path_checkpoint"])
    model.load_state_dict(state_dict)
    model.to(device)
    inputs = torch.load(cfg["paths"]["path_test_tokens"])
    labels = torch.load(cfg["paths"]["path_test_labels"])
    test = TensorDataset(
        inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float()
    )
    test_loader = DataLoader(test, batch_size=cfg["model"]["batch_size"], shuffle=False)

    with torch.no_grad():
        acc = torch.tensor([0], dtype=torch.float)
        baseline_acc = torch.tensor([0], dtype=torch.float)
        for tokens, mask, labels in tqdm(test_loader, desc="Looping over testdata"):
            # Only count if there is any toxicity
            outputs = model(tokens.to(device), mask.to(device))
            outputs = torch.sigmoid(outputs).to("cpu")
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            equals = ((outputs == labels) * 1).float()
            acc += equals.mean().item()
            predict0 = ((labels == 1) * 1).float()
            baseline_acc += predict0.mean().item()
        print(
            f"Accuracy: {acc/len(test_loader)}, Baseline: {baseline_acc/len(test_loader)}"
        )


if __name__ == "__main__":
    predict()
