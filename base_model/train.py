import argparse
import os
import sys

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import video_dataset
from .samplers import DDPMSampler
from unet import unet2d, unet3d
from utility import utils

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VFIDDPM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_steps = config.n_steps
        self.max_lr = config.max_lr
        self.use_3d = config.use_3d
        if self.use_3d:
            self.model = unet3d.UNet3d(config.n_steps, config.emb_dim, config.channel_mult, config.attn_heads)
        else:
            self.model = unet2d.UNet(config.n_steps, config.emb_dim, config.channel_mult, config.attn_heads, in_channels=9)
        self.sampler = DDPMSampler(self.model)

    def add_noise(self, x, t):
        # x shape: [batch_size, channels, depth=3, height, width]
        # Extract frames
        frame1 = x[:, :, 0]  # First frame
        frame2 = x[:, :, 1]  # Middle frame
        frame3 = x[:, :, 2]  # Last frame
        
        # Add noise only to the middle frame
        noise = torch.randn_like(frame2)
        sqrt_alpha_cumprod = self.sampler.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_alpha_cumprod = self.sampler.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
        
        # Apply noise to middle frame
        noisy_frame2 = sqrt_alpha_cumprod * frame2 + sqrt_1m_alpha_cumprod * noise
        
        # Reconstruct the sequence with the noisy middle frame
        if self.use_3d:
            noisy_sequence = torch.stack([frame1, noisy_frame2, frame3], dim=2)
        else:
            noisy_sequence = torch.cat([frame1, noisy_frame2, frame3], dim=1)
        
        return noisy_sequence, noise
    
    def training_step(self, batch, batch_idx):
        # Log learning rate
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)

        # x shape: [batch_size, channels, depth=3, height, width]
        x = batch
        t = torch.randint(0, self.n_steps, (x.size(0),), device=device)
        x_t, noise = self.add_noise(x, t)
        noise_hat = self.model(x_t, t)
        loss = F.mse_loss(noise_hat, noise)
        self.log('loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.max_lr*0.1)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.config.base_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def forward(self, x, t):
        return self.model(x, t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion VFI Trainer')
    parser.add_argument('--config', type=str, default='configs.yaml')

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