import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf

from seq_models.trainer import get_disc_trainer
from seq_models.data import (
    get_loaders,
    make_discriminative_loader,
)
from scripts.utils import (
    flatten_config, 
    convert_to_dict,
)
from seq_models.model.mlm_diffusion import ValueShell

@hydra.main(config_path="../configs", config_name="train_discr_model")
def main(config):
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    wandb.init(
         project="train_value_function",
         config=log_config,
    )

    # there must be a way to get hydra to do this for me
    if config.target_cols and len(config.target_cols) > 0:
        config.model.network.target_channels = len(config.target_cols)

    pprint.PrettyPrinter(depth=4).pprint(convert_to_dict(config))

    # TODO: get the model 
    model = ValueShell()

    if torch.cuda.is_available():
        model.cuda()

    train_dl, valid_dl = get_loaders(config)
    disc_trainer = get_disc_trainer(config, len(train_dl))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # catch really annoying BioPython warnings
        
        disc_trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=valid_dl,
        )

if __name__ == "__main__":    
    main()
    sys.exit()