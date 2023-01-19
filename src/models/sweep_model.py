import torch
import json
from model import LightningBertBinary
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb
import yaml

from pytorch_lightning.loggers import WandbLogger

sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "layer_dim": {"max": 2000, "min": 50},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}


if torch.has_cuda:
    device = "cuda"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} accelerator")


print("Starting sweep")
sweep_id = wandb.sweep(sweep_configuration, project="gpt-4", entity="oldboys")


def train():
    wandb.init()
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    layer_dim = wandb.config.layer_dim
    print(f"lr: {lr}, batch_size: {batch_size}")

    # Load config
    with open("config/config.json") as file:
        cfg = json.load(file)

    cfg["model"]["learning_rate"] = lr
    cfg["model"]["batch_size"] = batch_size
    cfg['model']['layer_dim'] = layer_dim
    print(cfg)
    # Initialize a WandbLogger
    wandb_logger = WandbLogger(project="gpt-4")

    model = LightningBertBinary(cfg)
    wandb_logger.watch(model)
    trainer = Trainer(
        max_epochs=cfg["model"]["n_epochs"], accelerator=device, logger=wandb_logger
    )
    trainer.fit(model)

    # Save
    #model.to("cpu")
    #torch.save(model.state_dict(), cfg["paths"]["path_checkpoint"])


if __name__ == "__main__":
    # train()
    #
    wandb.agent(sweep_id, function=train, count=4)
