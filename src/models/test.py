from omegaconf import OmegaConf
import hydra

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def train(cfg):

    Hey = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    print(Hey.keys())


train()