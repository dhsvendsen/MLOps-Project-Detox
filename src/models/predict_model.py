import torch
from model import LightningBert
import hydra
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

if torch.has_mps:
    acc="mps"
else:
    acc="cpu"


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def predict(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Load model and state dict from predict config
    model = LightningBert(cfg_dict)
    state_dict = torch.load(cfg.predict['modelpath'])
    model.load_state_dict(state_dict)
    model.to(acc)
    inputs = torch.load(cfg.predict["datapaths"][0])
    labels = torch.load(cfg.predict["datapaths"][1])
    test = TensorDataset(inputs[0].type(torch.int64), inputs[1].type(torch.int64), labels.float())
    test_set = DataLoader(test, batch_size=cfg.predict['batch_size'], shuffle=False)

    with torch.no_grad():
        label_specific = torch.tensor([0], dtype=torch.float)
        any_toxicity = torch.tensor([0], dtype=torch.float)
        acc_vector = torch.tensor([0,0,0,0,0,0], dtype=torch.float)
        acc_baseline = torch.tensor([0,0,0,0,0,0], dtype=torch.float)
        for i, (tokens, mask, labels) in enumerate(test_set):
            # Only count if there is any toxicity
            if labels.sum() > 0:
                outputs = model(tokens.to(acc), mask.to(acc))
                outputs = torch.sigmoid(outputs).to("cpu")
                outputs[outputs >= 0.5] = 1
                correct_1s = outputs == labels
                label_specific += (correct_1s.sum()/labels.sum()).type(torch.FloatTensor)
                correct_rows = correct_1s.sum(axis=1) > 0
                toxic_rows = labels.sum(axis=1) > 0
                any_toxicity += (correct_rows.sum()/toxic_rows.sum()).type(torch.FloatTensor)
                # accuracy
                outputs[outputs < 0.5] = 0
                correct_0s_1s = outputs == labels
                acc_vector += correct_0s_1s.to(torch.float16).mean(axis=0)
                acc_baseline += (labels==0).to(torch.float16).mean(axis=0)
                

            if i%25==0:
                print(f"{100*i/len(test_set):4f}% done")
            
            if i==200:
                break
                

        print(f"Percetage of all \"1s\" caught: {label_specific.item()*100/(i+1)}%")
        print(f"Percetage of toxic (any kind) comments caught: {any_toxicity.item()*100/(i+1)}%")
        print(f"Class based accuracy {acc_vector.numpy()/(i+1)}")
        print(f"Class based baseline (0) {acc_baseline.numpy()/(i+1)}")
        #print(f"Percetage of all \"1s\" caught: {label_specific.item()*100/len(test_set)}%")
        #print(f"Percetage of toxic (any kind) comments caught: {any_toxicity.item()*100/len(test_set)}%")

if __name__ == "__main__":
    predict()
