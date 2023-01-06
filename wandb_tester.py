import wandb
import random

wandb.init(project="wandb-6-1")
for _ in range(100):
    wandb.log({"test_metric": random.random()})
