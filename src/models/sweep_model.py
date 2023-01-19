import torch
import json
from model import LightningBertBinary
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb
import yaml

from pytorch_lightning.loggers import WandbLogger

wandb.init(config={"learning_rate": 0.01, "batch_size": 32, "epochs": 5}, project="gpt-4", entity="oldboys")


# Initialize a WandbLogger
wandb_logger = WandbLogger(project="gpt-4")

if torch.has_cuda:
    device = "cuda"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} accelerator")


sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'val_loss'
		},
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}

print("Starting sweep")
sweep_id = wandb.sweep(sweep_configuration, project="gpt-4", entity="oldboys")

def train():
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    

    # Load config
    with open("config/config.json") as file:
        cfg = json.load(file)
    
    model = LightningBertBinary(cfg)
    wandb_logger.watch(model)
    trainer = Trainer(max_epochs=cfg["model"]["n_epochs"], accelerator=device, logger = wandb_logger)
    trainer.fit(model)

    # Save
    model.to("cpu")
    torch.save(model.state_dict(), cfg["paths"]["path_checkpoint"])



if __name__ == "__main__":
    #train()
    #
    wandb.agent(sweep_id, train)
    
