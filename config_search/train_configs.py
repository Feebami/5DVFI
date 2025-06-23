import argparse
import gc
import os
import sys

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from base_model.train import VFIDDPM
from dataset import video_dataset
from utility import utils

# Set high precision for matrix multiplications
torch.set_float32_matmul_precision('high')
# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    # Time limit for each training session in minutes
    parser.add_argument('--minutes', type=int, required=True)
    # Optional checkpoint version for resuming training
    parser.add_argument('--ckpt_version', type=int)
    args = parser.parse_args()

    # Iterate through all config files in config_search/configs
    for config_file in os.listdir('config_search/configs'):
        print('= '*16 + f'Training {config_file}' + ' ='*16)
        
        # Load configuration from YAML file
        config = OmegaConf.load('config_search/configs/' + config_file)

        # Prepare dataset transformations
        transform = utils.get_v90k_transform(config.img_size)
        # Initialize video dataset
        data = video_dataset.V90KDataset(transform)

        # Configure data loader with performance optimizations
        dataloader = DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,               # Parallel data loading
            persistent_workers=True,      # Maintain workers between epochs
            prefetch_factor=4,            # Preload batches
            pin_memory=True               # Faster data transfer to GPU
        )

        # Handle checkpoint resuming if version is specified
        if args.ckpt_version is not None:
            # Construct checkpoint path based on version
            ckpt_folder = os.path.join(
                config_file.split('.')[0] + '_logs',
                'lightning_logs',
                f'version_{args.ckpt_version}',
                'checkpoints'
            )
            # Select first checkpoint in directory
            ckpt_path = os.path.join(ckpt_folder, os.listdir(ckpt_folder)[0])
            print('Loading from checkpoint')
        else:
            ckpt_path = None
        
        # Initialize diffusion model
        module = VFIDDPM(config)

        # Configure Lightning Trainer
        trainer = L.Trainer(
            max_steps=config.base_steps,      # Total training steps
            default_root_dir=f"{config_file.split('.')[0]}_logs",  # Log dir
            precision='bf16-mixed',           # Mixed-precision training
            gradient_clip_val=1.0,            # Prevent exploding gradients
            max_time={'minutes': args.minutes} # Time limit per config
        )

        # Start/resume training
        trainer.fit(
            module, 
            dataloader,
            ckpt_path=ckpt_path  # None for new training
        )

        # === CLEANUP ===
        # Explicitly delete large objects
        del module
        del trainer
        del dataloader
        del data

        # Force garbage collection
        gc.collect()

        # Clear GPU memory
        torch.cuda.empty_cache()
        # Wait for all operations to complete
        torch.cuda.synchronize()
