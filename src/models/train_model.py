import os
print(os.getcwd())
print(os.path.dirname(os.path.abspath(__file__)))
#from src.models.model import LightningBert
from model import LightningBert
from pytorch_lightning import Trainer
import hydra

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def train(cfg):
    model = LightningBert(cfg)
    trainer = Trainer(max_epochs=cfg.train['n_epochs'])
    trainer.fit(model)

if __name__ == "__main__":
    train()
