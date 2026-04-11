"""
Setup Utility functions for SuperYOLO
"""

import os
import yaml
import torch
from pathlib import Path
from utils.wandb_logging.wandb_utils import WandbLogger


def setup_directories(save_dir):
    """Create and return training directories"""
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    
    return last, best, results_file


def save_run_settings(save_dir, opt, hyp):
    """Save hyperparameters and options to YAML files"""
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)


def setup_seeds_and_config(opt, device, init_seeds, rank):
    """Initialize random seeds and load data configuration"""
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')
    
    return plots, cuda, data_dict, is_coco


def setup_logging_and_wandb(opt, hyp, weights, rank, data_dict, save_dir):
    """Setup logging and Weights & Biases integration
    
    Note: WandbLogger may modify opt.weights, opt.epochs, opt.hyp during initialization
    if resuming from a checkpoint. That's why we re-read them after initialization.
    """
    loggers = {'wandb': None}  # loggers dict
    wandb_logger = None
    
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, weights_only=False).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            # WandbLogger might have updated opt during initialization (when resuming)
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp
        else:
            # WandB not active, use original values
            epochs = opt.epochs
    else:
        # Not in main process, use original values
        epochs = opt.epochs
    
    return loggers, wandb_logger, data_dict, weights, epochs, hyp