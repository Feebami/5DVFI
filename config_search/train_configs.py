import argparse
import gc
import os
import sys

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base_model.samplers import DDPMSampler
from base_model.train import VFIDDPM
from dataset import video_dataset
from unet import unet2d, unet3d
from utility import utils

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, required=True)
    parser.add_argument('--ckpt_version', type=int)
    args = parser.parse_args()

    for config_file in os.listdir('config_search/configs'):
        print('= '*16 + f'Training {config_file}' + ' ='*16)
        config = OmegaConf.load('config_search/configs/' + config_file)

        transform = utils.get_v90k_transform(config.img_size)
        data = video_dataset.V90KDataset(transform)

        dataloader = DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True
        )

        if args.ckpt_version is not None:
            ckpt_folder = os.path.join(
                config_file.split('.')[0] + '_logs',
                'lightning_logs',
                f'version_{args.ckpt_version}',
                'checkpoints'
            )
            ckpt_path = os.path.join(ckpt_folder, os.listdir(ckpt_folder)[0])
            print('Loading from checkpoint')
        
        module = VFIDDPM(config)

        trainer = L.Trainer(
            max_steps=config.base_steps,
            default_root_dir=f"{config_file.split('.')[0]}_logs",
            precision='bf16-mixed',
            gradient_clip_val=1.0, 
            max_time={'minutes': args.minutes}
        )

        trainer.fit(
            module, 
            dataloader,
            ckpt_path=ckpt_path if args.ckpt_version is not None else None
        )

        # === CLEANUP ===
        del module
        del trainer
        del dataloader
        del data

        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()