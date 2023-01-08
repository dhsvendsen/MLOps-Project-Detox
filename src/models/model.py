import torch
from torch import optim, nn
from pytorch_lightning import LightningModule
from transformers import BertModel
from torch.utils.data import TensorDataset, DataLoader



class LightningBert(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained(
            self.cfg.model["pretrained_name"]
        )
        # Only train last layer
        for param in self.bert.parameters():
            param.requires_grad = False
        self.class_layer=nn.Linear(self.bert.config.hidden_size, 6) 
        # BCE, since we are doing multi-label classification
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, batch):
        bertput = self.bert(batch)
        output = self.class_layer(bertput.pooler_output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # Forward pass
        outputs = self(inputs)
        # TODO: implement logging of the loss
        return {"loss": self.loss(outputs, labels)}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.train["lr"])
    
    def train_dataloader(self):
        inputs = torch.load(self.cfg.train["datapaths"][0])
        labels = torch.load(self.cfg.train["datapaths"][1])
        train = TensorDataset(inputs, labels)
        train_loader = DataLoader(train, batch_size=self.cfg.train["batch_size"], shuffle=True)
        return train_loader

    def val_dataloader(self):
        inputs = torch.load(self.cfg.train["datapaths"][2])
        labels = torch.load(self.cfg.train["datapaths"][3])
        val = TensorDataset(inputs, labels)
        val_loader = DataLoader(val, batch_size=self.cfg.train["batch_size"], shuffle=True)
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        return {"val_loss": self.loss(outputs, labels)}