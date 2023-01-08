import torch
from model import LightningBert
import hydra
from torch.utils.data import DataLoader, TensorDataset

if torch.has_mps:
    acc="mps"
else:
    acc=None

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def predict(cfg):
    # Load model and state dict from predict config
    model = LightningBert(cfg)
    state_dict = torch.load(cfg.predict['modelpath'])
    model.load_state_dict(state_dict)
    inputs = torch.load(cfg.predict["datapaths"][0])
    labels = torch.load(cfg.predict["datapaths"][1])
    test = TensorDataset(inputs, labels)
    test_set = DataLoader(test, batch_size=16, shuffle=True)

    with torch.no_grad():
        metric = torch.tensor([0], dtype=torch.float)
        for inputs, labels in test_set:
            print(inputs)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs) 
            outputs[outputs >= 0.5] = 1
            print(labels, outputs)
            equals = outputs == labels
            metric = torch.mean(equals.type(torch.FloatTensor))
            break
        print(f"Percetage \"1s\" caught: {metric.item()*100/len(test_set)}%")


if __name__ == "__main__":
    predict()
