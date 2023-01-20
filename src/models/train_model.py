import json

import torch
from model import LightningBertBinary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="test")

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

    trainer = Trainer(
        max_epochs=cfg["model"]["n_epochs"], accelerator=device, logger=wandb_logger
    )
    trainer.fit(model)

    # Save
    model.to("cpu")
    torch.save(model.state_dict(), cfg["paths"]["path_checkpoint"])


if __name__ == "__main__":
    train()
