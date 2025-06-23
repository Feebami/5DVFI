import argparse
import os
import subprocess
import sys

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from dataset import video_dataset
from .train import VFIDDPM
from utility import utils

# Set high precision for matrix multiplications to improve numerical stability
torch.set_float32_matmul_precision('high')
# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # ------------------------------
    # Training Pipeline Setup
    # ------------------------------
    parser = argparse.ArgumentParser(description='Diffusion VFI Trainer')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Prepare data transformations
    transform = utils.get_transform(config.img_size)
    
    # Initialize dataset and dataloader
    data = video_dataset.UCF90KCombinedDataset(transform)
    dataloader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True
    )
    
    # Initialize model
    module = VFIDDPM(config)
    
    # Configure trainer
    trainer = L.Trainer(
        max_steps=config.base_steps,
        default_root_dir=f"{os.path.basename(args.config).split('.')[0]}_logs",
        precision='bf16-mixed',
        gradient_clip_val=1.0,
    )
    
    # Start training
    trainer.fit(module, dataloader)
    
    # Save final configuration
    config_filepath = os.path.join(trainer.logger.log_dir, 'config.yaml')
    OmegaConf.save(config, config_filepath)
    
    # Locate latest checkpoint file
    ckpt_folder = os.path.join(trainer.logger.log_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_folder, sorted(os.listdir(ckpt_folder))[-1])
    
    # Extract base config name for output directories
    config_name = os.path.basename(args.config).split('.')[0]

    # ------------------------------
    # Sample Generation & Evaluation
    # ------------------------------
    # Generate interpolated frames using trained model
    subprocess.run([
        'python', '-m', 'utility.make_diffused_samples',
        '--config', args.config,     # Configuration file
        '--ckpt', ckpt_path,         # Model checkpoint
        '--num_files', str(127),     # Number of samples to generate
        '--ddim_steps', str(8)       # DDIM sampling steps
    ])
    
    # Evaluate generated samples
    eval_dir = os.path.join('eval_samples', 'diffused', config_name)
    subprocess.run([
        'python', '-m', 'utility.eval_sample_dir',
        '--dir', eval_dir,           # Directory containing samples
    ])
