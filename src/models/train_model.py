import pickle
import subprocess

import hydra
import torch
from model import LightningBert
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

# specify the name of the bucket and the folder
bucket_name = "dtumlops-storage"
folder_name = "models"


if torch.has_cuda:
    acc = "cuda"
elif torch.has_mps:
    acc = "mps"
else:
    acc = "cpu"
print(f"Using {acc} accelerator")


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def train(cfg):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with open("../../../src/deployment/latest_training_dict.pickle", "wb") as handle:
        pickle.dump(cfg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = LightningBert(cfg_dict)

    trainer = Trainer(max_epochs=cfg.train["n_epochs"], accelerator=acc)
    trainer.fit(model)

    # Save
    torch.save(model.state_dict(), cfg.train["modelpath"])

    # use the gsutil cp command to upload the model file to the bucket
    subprocess.run(
        ["gsutil", "cp", cfg.train["modelpath"], f"gs://{bucket_name}/{folder_name}/"]
    )


if __name__ == "__main__":
    train()
