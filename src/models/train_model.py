import torch
import json
from model import LightningBertBinary
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from google.cloud import storage

# test

if torch.has_cuda:
    device = "cuda"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} accelerator")


def train():

    # Load config
    with open("config/config.json") as file:
        cfg = json.load(file)

    model = LightningBertBinary(cfg)

    trainer = Trainer(max_epochs=cfg["model"]["n_epochs"], accelerator=device)
    trainer.fit(model)

    # Save
    model.to("cpu")
    #torch.save(model.state_dict(), cfg["paths"]["path_checkpoint"])
    storage_client = storage.Client("my-vertex-bucket")
    bucket = storage_client.bucket("my-vertex-bucket")
    blob = bucket.blob("model/model.pt")
    with blob.open("wb", ignore_flush=True) as f:
        torch.save(model.state_dict(), f)


if __name__ == "__main__":
    train()
