import os
import torch
import pytorch_lightning as pl
# TODO: get paths right, the train_subset.yaml paths have too many ../../ >.<
print(os.getcwd())
print(os.path.dirname(os.path.abspath(__file__)))
#from src.models.model import LightningBert
from model import LightningBert
from pytorch_lightning import Trainer
import hydra
import wandb
from omegaconf import OmegaConf

if torch.has_cuda:
    acc="cuda"
elif torch.has_mps:
    acc="mps"
else:
    acc=None


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def train(cfg):
    
    model = LightningBert(cfg)

    trainer = Trainer(max_epochs=cfg.train['n_epochs'],
        accelerator=acc)
    trainer.fit(model)

    # Save
    torch.save(model.state_dict(), cfg.train["modelpath"])

if __name__ == "__main__":
    train()
