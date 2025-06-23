import argparse
import os
import subprocess
import sys

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import video_dataset
from .samplers import DDPMSampler
from .train import VFIDDPM
from unet import unet2d, unet3d
from utility import utils

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion VFI Trainer')
    parser.add_argument('--config', type=str, default='configs/winner8.yaml')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    transform = utils.get_transform(config.img_size)

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

    module = VFIDDPM(config)
    trainer = L.Trainer(
        max_steps=config.base_steps,
        default_root_dir=f"{os.path.basename(args.config).split('.')[0]}_logs",
        precision='bf16-mixed',
        gradient_clip_val=1.0,
    )

    trainer.fit(module, dataloader)

    config_filepath = os.path.join(trainer.logger.log_dir, 'config.yaml')
    OmegaConf.save(config, config_filepath)

    ckpt_folder = os.path.join(trainer.logger.log_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_folder, sorted(os.listdir(ckpt_folder))[-1])

    config_name = os.path.basename(args.config).split('.')[0]

    subprocess.run([
        'python', '-m', 'utility.make_diffused_samples',
        '--config', args.config,
        '--ckpt', ckpt_path,
        '--num_files', str(127),
        '--ddim_steps', str(8)
    ])

    eval_dir = os.path.join('eval_samples', 'diffused', config_name)
    subprocess.run([
        'python', '-m', 'utility.eval_sample_dir',
        '--dir', eval_dir,
    ])
