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
    test = TensorDataset(inputs[0], inputs[1], labels)
    test_set = DataLoader(test, batch_size=cfg.predict['batch_size'], shuffle=False)

    with torch.no_grad():
        label_specific = torch.tensor([0], dtype=torch.float)
        any_toxicity = torch.tensor([0], dtype=torch.float)
        for i, (tokens, mask, labels) in enumerate(test_set):
            # Only count if there is any toxicity
            if labels.sum() > 0:
                outputs = model(tokens, mask)
                outputs = torch.sigmoid(outputs) 
                outputs[outputs >= 0.5] = 1
                correct_1s = outputs == labels
                label_specific += (correct_1s.sum()/labels.sum()).type(torch.FloatTensor)
                correct_rows = correct_1s.sum(axis=1) > 0
                toxic_rows = labels.sum(axis=1) > 0
                any_toxicity += (correct_rows.sum()/toxic_rows.sum()).type(torch.FloatTensor)


        print(f"Percetage of all \"1s\" caught: {label_specific.item()*100/len(test_set)}%")
        print(f"Percetage of toxic (any kind) comments caught: {any_toxicity.item()*100/len(test_set)}%")

if __name__ == "__main__":
    predict()
