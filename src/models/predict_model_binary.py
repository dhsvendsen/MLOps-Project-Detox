import torch
from model import LightningBertBinary
import hydra
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

if torch.has_mps:
    device="mps"
else:
    device="cpu"


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def predict(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Load model and state dict from predict config
    model = LightningBertBinary(cfg_dict)
    state_dict = torch.load(cfg.predict['modelpath'])
    model.load_state_dict(state_dict)
    model.to(device)
    inputs = torch.load(cfg.predict["datapaths"][0])
    labels = torch.load(cfg.predict["datapaths"][1])
    test = TensorDataset(inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float())
    test_loader = DataLoader(test, batch_size=cfg.predict['batch_size'], shuffle=False)

    with torch.no_grad():
        acc = torch.tensor([0], dtype=torch.float)
        baseline_acc = torch.tensor([0], dtype=torch.float)
        for i, (tokens, mask, labels) in enumerate(test_loader):
            # Only count if there is any toxicity
            outputs = model(tokens.to(device), mask.to(device))
            outputs = torch.sigmoid(outputs).to("cpu")
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            equals = ((outputs == labels)*1).float()
            acc += equals.mean().item()
            predict0 = ((labels == 0)*1).float()
            baseline_acc += predict0.mean().item() 
        print(f"{acc/len(test_loader)} {baseline_acc/len(test_loader)}")
                

if __name__ == "__main__":
    predict()
