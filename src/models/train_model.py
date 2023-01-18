import torch
import json
from model import LightningBertBinary
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb

if torch.has_cuda:
    device = "cuda"
elif torch.has_mps:
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} accelerator")


def train():

    # Load config
    with open('config/config.json') as file:
        cfg = json.load(file)
    
    model = LightningBertBinary(cfg)

    trainer = Trainer(max_epochs=cfg["model"]["n_epochs"], accelerator=device)
    trainer.fit(model)

    # Save
    model.to("cpu")
    torch.save(model.state_dict(), cfg["paths"]["path_checkpoint"])

    # use the gsutil cp command to upload the model file to the bucket
    subprocess.run(
        ["gsutil", "cp", cfg.train["modelpath"], f"gs://{bucket_name}/{folder_name}/"]
    )


if __name__ == "__main__":
    train()
