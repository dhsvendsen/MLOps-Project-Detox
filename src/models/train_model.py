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

if torch.has_mps:
    acc="mps"
else:
    acc=None

acc=None #wandb + mps doesn't work :(

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def train(cfg):
    
    wandb.init(
      # Set the project where this run will be logged
      project=cfg.wandb["project"], 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=cfg.wandb["name"],
      entity=cfg.wandb["entity"]
      )

    # Pytorch wandb logger
    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb['project'])
    #wandb.config = OmegaConf.to_container(
    #    cfg.train, resolve=True, throw_on_missing=True
    #)
    #wandb.log(wandb.config)

    model = LightningBert(cfg)
    wandb_logger.watch(model, log="all", log_freq=100)
    trainer = Trainer(max_epochs=cfg.train['n_epochs'],
        accelerator=acc,
        logger=wandb_logger)
    trainer.fit(model)

    # Save
    torch.save(model.state_dict(), cfg.train["modelpath"])

if __name__ == "__main__":
    train()
