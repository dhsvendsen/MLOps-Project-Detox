import os
print(os.getcwd())
print(os.path.dirname(os.path.abspath(__file__)))
#from src.models.model import LightningBert
from model import LightningBert
from pytorch_lightning import Trainer

# this will come from hydra or something in the future
conf = {
    "datapath":["data/processed/input_train_n10000_l32.pt","data/processed/labels_train_n10000.pt"],
    "model_name":"bert-base-uncased",
    "lr":0.001,
    "n_epochs":1,
    "batch_size":16
}

model = LightningBert(conf)
trainer = Trainer(max_epochs=conf['n_epochs'])
trainer.fit(model)


