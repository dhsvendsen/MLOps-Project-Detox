import torch
from torch import optim, nn
from pytorch_lightning import LightningModule
from transformers import BertModel
from torch.utils.data import TensorDataset, DataLoader



class LightningBert(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = self.cfg["train"]["lr"]
        self.batch_size = self.cfg["train"]["batch_size"]
        
        self.bert = BertModel.from_pretrained(
            self.cfg["model"]["pretrained_name"]
        )
        # Only train last layer
        for param in self.bert.parameters():
            param.requires_grad = False
        self.class_layer=nn.Linear(self.bert.config.hidden_size, 6) 
        # BCE, since we are doing multi-label classification
        # The weights are the fraction of 0's to 1's in each class
        # So we multiply the 1-terms in the loss function with this for balance
        pos_weights = torch.tensor([9.43, 99.04, 17.88, 332.83, 19.25, 112.57], dtype=torch.float32)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
    def forward(self, tokens, mask):
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

    #def forward(self, tokenbatch):
        #bertput = self.bert(tokenbatch[0], tokenbatch[1])
        #output = self.class_layer(bertput.pooler_output)
        #return output

    #def training_step(self, batch, batch_idx):
        #inputs, labels = batch
        ## Forward pass
        #outputs = self(inputs)
        ## TODO: implement logging of the loss
        #loss = self.loss(outputs, labels)
        #self.log("train-loss", loss)
        #return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        inputs = torch.load(self.cfg["train"]["datapaths"][0])
        labels = torch.load(self.cfg["train"]["datapaths"][1])
        #train = TensorDataset(inputs, labels)
        train = TensorDataset(inputs[0], inputs[1], labels)
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        inputs = torch.load(self.cfg["train"]["datapaths"][2])
        labels = torch.load(self.cfg["train"]["datapaths"][3])
        #val = TensorDataset(inputs, labels)
        val = TensorDataset(inputs[0], inputs[1], labels)
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        inputs, mask, labels = batch
        outputs = self(inputs, mask)
        return {"val_loss": self.loss(outputs, labels)}

    #def validation_step(self, batch, batch_idx):
        #inputs, labels = batch
        #outputs = self(inputs)
        #return {"val_loss": self.loss(outputs, labels)}