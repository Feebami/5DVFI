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

# Set high precision for matrix multiplications
torch.set_float32_matmul_precision('high')
# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VFIDDPM(L.LightningModule):
    """Lightning Module for Video Frame Interpolation using DDPM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_steps = config.n_steps  # Diffusion steps
        self.max_lr = config.max_lr  # Maximum learning rate
        self.use_3d = config.use_3d  # 3D architecture flag
        
        # Initialize U-Net model based on architecture type
        if self.use_3d:
            self.model = unet3d.UNet3d(
                config.n_steps,
                config.emb_dim,
                config.channel_mult,
                config.attn_heads
            )
        else:
            self.model = unet2d.UNet(
                config.n_steps,
                config.emb_dim,
                config.channel_mult,
                config.attn_heads,
                in_channels=9  # 3 channels * 3 frames
            )
        
        # Initialize DDPM sampler
        self.sampler = DDPMSampler(self.model)

    def add_noise(self, x, t):
        """Add noise to middle frame in video sequence
        Args:
            x: Input tensor [B, C, D=3, H, W]
            t: Timestep tensor [B]
        Returns:
            noisy_sequence: Sequence with noisy middle frame
            noise: Ground truth noise added
        """
        # Extract frames from sequence
        frame1 = x[:, :, 0]  # First frame
        frame2 = x[:, :, 1]  # Middle frame (target)
        frame3 = x[:, :, 2]  # Last frame
        
        # Generate noise for middle frame
        noise = torch.randn_like(frame2)
        
        # Get schedule parameters for current timestep
        sqrt_alpha_cumprod = self.sampler.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_alpha_cumprod = self.sampler.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
        
        # Apply noise to middle frame using diffusion formula
        noisy_frame2 = sqrt_alpha_cumprod * frame2 + sqrt_1m_alpha_cumprod * noise
        
        # Reconstruct sequence with noisy middle frame
        if self.use_3d:
            noisy_sequence = torch.stack([frame1, noisy_frame2, frame3], dim=2)
        else:
            noisy_sequence = torch.cat([frame1, noisy_frame2, frame3], dim=1)
        
        return noisy_sequence, noise

    def training_step(self, batch, batch_idx):
        """Training step with learning rate logging
        Args:
            batch: Input video sequences [B, C, D=3, H, W]
            batch_idx: Batch index
        Returns:
            loss: MSE loss between predicted and actual noise
        """
        # Log current learning rate
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)

        x = batch  # Input sequence
        # Sample random timesteps for each sequence in batch
        t = torch.randint(0, self.n_steps, (x.size(0),), device=device)
        
        # Add noise to middle frame and get ground truth noise
        x_t, noise = self.add_noise(x, t)
        
        # Predict noise using model
        noise_hat = self.model(x_t, t)
        
        # Calculate MSE loss
        loss = F.mse_loss(noise_hat, noise)
        self.log('loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Initialize Adam optimizer with initial LR
        optimizer = Adam(self.model.parameters(), lr=self.max_lr * 0.1)
        
        # One-cycle learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.config.base_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def forward(self, x, t):
        """Forward pass through model"""
        return self.model(x, t)


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
