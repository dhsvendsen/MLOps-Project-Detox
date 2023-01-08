import torch
from torch import optim
from pytorch_lightning import LightningModule
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader



class LightningBert(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = BertForSequenceClassification.from_pretrained(
            self.cfg.model["pretrained_name"],
            torchscript=True,
            num_labels=6,
        )
        
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, _ = self.bert(inputs, labels=labels)
        # good place to log train loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.train["lr"])
    
    def train_dataloader(self):
        inputs = torch.load(self.cfg.train["datapaths"][0])
        labels = torch.load(self.cfg.train["datapaths"][1])
        train = TensorDataset(inputs, labels)
        train_loader = DataLoader(train, batch_size=self.cfg.train["batch_size"], shuffle=True)
        return train_loader