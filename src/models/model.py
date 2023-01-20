import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel


class LightningBert(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = self.cfg["model"]["lr"]
        self.batch_size = self.cfg["model"]["batch_size"]

        self.bert = BertModel.from_pretrained(self.cfg["model"]["pretrained_name"])
        # Only train last layer
        for param in self.bert.parameters():
            param.requires_grad = False
        self.class_layer = nn.Linear(self.bert.config.hidden_size, 6)
        # BCE, since we are doing multi-label classification
        # The weights are the fraction of 0's to 1's in each class
        # So we multiply the 1-terms in the loss function with this for balance
        pos_weights = torch.tensor(
            [9.43, 99.04, 17.88, 332.83, 19.25, 112.57], dtype=torch.float32
        )
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, tokens, mask):
        if tokens.ndim != 2:
            raise ValueError("Expected tokens to a 2D tensor")
        bertput = self.bert(tokens, mask)
        output = self.class_layer(bertput.pooler_output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, masks, labels = batch
        # Forward pass
        outputs = self(inputs, masks)
        # TODO: implement logging of the loss
        loss = self.loss(outputs, labels)
        self.log("train-loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        inputs = torch.load(self.cfg["paths"]["path_train_tokens"])
        labels = torch.load(self.cfg["paths"]["path_train_labels"])
        train = TensorDataset(
            inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float()
        )
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        inputs = torch.load(self.cfg["paths"]["path_val_tokens"])
        labels = torch.load(self.cfg["paths"]["path_val_labels"])
        val = TensorDataset(
            inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float()
        )
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        return val_loader

    def validation_step(self, batch, batch_idx):
        inputs, mask, labels = batch
        outputs = self(inputs, mask)
        return {"val_loss": self.loss(outputs, labels)}


class LightningBertBinary(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = self.cfg["model"]["lr"]
        self.batch_size = self.cfg["model"]["batch_size"]

        self.bert = BertModel.from_pretrained(self.cfg["model"]["pretrained_name"])
        # Only train last layer
        for param in self.bert.parameters():
            param.requires_grad = False
        self.hidden_dim = self.cfg["model"]["layer_dim"]
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, self.hidden_dim)
        self.class_layer = nn.Linear(self.hidden_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tokens, mask):
        if tokens.ndim != 2:
            raise ValueError("Expected tokens to a 2D tensor")
        bertput = self.bert(tokens, mask)
        intermediate = self.hidden_layer(bertput.pooler_output)
        output = self.class_layer(intermediate)
        return output

    def training_step(self, batch, batch_idx):
        inputs, masks, labels = batch
        # Forward pass
        outputs = self(inputs, masks)
        # TODO: implement logging of the loss
        loss = self.loss(outputs, labels)
        self.log("train-loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        inputs = torch.load(self.cfg["paths"]["path_train_tokens"])
        labels = torch.load(self.cfg["paths"]["path_train_labels"])
        train = TensorDataset(
            inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float()
        )
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        inputs = torch.load(self.cfg["paths"]["path_val_tokens"])
        labels = torch.load(self.cfg["paths"]["path_val_labels"])
        val = TensorDataset(
            inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float()
        )
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        return val_loader

    def validation_step(self, batch, batch_idx):
        inputs, mask, labels = batch
        outputs = self(inputs, mask)
        self.log("val_loss", self.loss(outputs, labels))
        return {"val_loss": self.loss(outputs, labels)}
