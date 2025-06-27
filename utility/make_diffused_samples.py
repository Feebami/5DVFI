import argparse
import os
import sys

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from dataset.video_dataset import VFIDatasetSmall
from base_model.samplers import DDIMSampler
from base_model.train import VFIDDPM
from utility import utils

if __name__ == '__main__':
    # Argument parser for command line configuration
    parser = argparse.ArgumentParser(description='Video Frame Interpolation using Diffusion Models')
    # Path to model configuration YAML file
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Model configuration file')
    # Path to model checkpoint file
    parser.add_argument('--ckpt', type=str, default='epoch=5-step=80000.ckpt',
                        help='Model checkpoint file')
    # Number of video files to process (default=127 for UCF-101 test set)
    parser.add_argument('--num_files', type=int, default=127,
                        help='Number of videos to interpolate')
    # DDIM sampling steps (fewer steps = faster but lower quality)
    parser.add_argument('--ddim_steps', type=int, default=16,
                        help='DDIM sampling steps')
    args = parser.parse_args()

    # Create output directory for interpolated videos
    sample_folder = os.path.join('eval_samples', 'diffused')
    os.makedirs(sample_folder, exist_ok=True)

    # Load model configuration from YAML file
    config = OmegaConf.load(args.config)

    # Get list of UCF-101 video file paths
    files = utils.get_ucf_file_paths(num_files=args.num_files)

    # Verify corresponding real videos exist for fair evaluation
    assert all(os.path.basename(f).split('.')[0] + '.mp4' in os.listdir('eval_samples/real') for f in files), \
        "Real videos missing for comparison"

    # Load pre-trained diffusion model
    module = VFIDDPM.load_from_checkpoint(args.ckpt, config=config)
    module.eval()  # Set to evaluation mode (no gradients)

    # Initialize DDIM sampler for accelerated generation
    sampler = DDIMSampler(module)

    # Process each video file with progress bar
    for file in tqdm(files, desc=os.path.basename(args.config).split('.')[0]):
        # Initialize video decoder
        decoder = utils.TrimmedDecoder(file)

        # Prepare image transformations (resize, normalize, etc.)
        transform = utils.get_transform(240)

        # Create dataset from video file
        data = VFIDatasetSmall(file, transform)
        # Configure dataloader for efficient batch processing
        dataloader = DataLoader(
            data,
            batch_size=32,                # Process 32 frames per batch
            shuffle=False,                # Maintain temporal order
            num_workers=4,                # Parallel data loading
            persistent_workers=True,      # Maintain workers between batches
            prefetch_factor=2,            # Preload next batches
            pin_memory=True               # Faster data transfer to GPU
        )

        diffusion_frames = []  # Stores generated frames
        # Process video in batches
        for batch in dataloader:
            # Extract first and third frames (t-1 and t+1)
            frame1s = batch[:, :, 0]  # Previous frame
            frame3s = batch[:, :, 2]  # Subsequent frame

            # Generate interpolated frames between frame1s and frame3s
            interps = sampler(frame1s, frame3s, eta=1.0, num_steps=args.ddim_steps)

            # Convert model output to RGB and move to CPU
            batch_frames = [utils.tensor_to_rgb(interp).cpu() for interp in interps]
            diffusion_frames.extend(batch_frames)

        # Combine all frames into single tensor [T, H, W, C]
        diffused_tensor = torch.stack(diffusion_frames)

        # Create output file path (same name as input with .mp4 extension)
        file_path = os.path.join(sample_folder, os.path.basename(file).split(".")[0] + '.mp4')

        # Write video file with fixed FPS
        write_video(
            file_path,
            diffused_tensor,
            fps=3,
            video_codec='h264'  # Standard video compression
        )