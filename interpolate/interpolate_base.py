import argparse
import os
import sys

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from base_model.samplers import DDIMSampler
from base_model.train import VFIDDPM
from dataset.video_dataset import VFIDatasetSmall
from utility import utils

# Set high precision for matrix multiplications to improve numerical stability
torch.set_float32_matmul_precision('high')
# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description='Diffusion VFI Sampler')
    # Path to model configuration YAML file
    parser.add_argument('--config', type=str, default='config.yaml')
    # Path to model checkpoint file
    parser.add_argument('--ckpt', type=str, default='epoch=5-step=80000.ckpt')
    # Path to input video file for interpolation
    parser.add_argument('--input_path', type=str, required=True)
    # Number of steps for DDIM sampling (fewer steps = faster but lower quality)
    parser.add_argument('--ddim_steps', type=int, default=32)
    # Directory to save output video
    parser.add_argument('--output_dir', type=str, default='samples')
    # Speed multiplier for output video (0.1 = 10% of original speed)
    parser.add_argument('--speed', type=float, default=0.1)

    args = parser.parse_args()

    # Load model configuration from YAML file
    config = OmegaConf.load(args.config)

    # Initialize video decoder for input video
    decoder = utils.TrimmedDecoder(args.input_path)
    # Determine smaller dimension for aspect ratio preservation
    small_dim = min(decoder.metadata.width, decoder.metadata.height)

    # Prepare image transformation pipeline (resize, normalize, etc.)
    transform = utils.get_transform(config.img_size, small_dim)

    # Create dataset from input video with transformations
    data = VFIDatasetSmall(args.input_path, transform)
    # DataLoader for efficient batching and loading
    dataloader = DataLoader(
        data,
        batch_size=16,               # Process 16 frames at a time
        shuffle=False,                # Maintain temporal order
        num_workers=4,                # Parallel data loading processes
        persistent_workers=True,      # Maintain workers between batches
        prefetch_factor=2,            # Preload next batches
        pin_memory=True               # Faster data transfer to GPU
    )

    # Load pre-trained diffusion model from checkpoint
    module = VFIDDPM.load_from_checkpoint(args.ckpt, config=config)
    module.eval()  # Set model to evaluation mode (no gradients)

    # Initialize DDIM sampler for accelerated generation
    sampler = DDIMSampler(module)

    frames = []  # List to store output frames
    # Process video in batches with progress bar
    for batch in tqdm(dataloader, desc='Interpolating...'):
        # Extract previous frames (t-1)
        prevs = batch[:, :, 0]
        # Extract subsequent frames (t+1)
        subseq = batch[:, :, 1]

        # Generate interpolated frames between prevs and subseq
        interps = sampler(prevs, subseq, eta=1.0, num_steps=args.ddim_steps)
        
        # Process each frame in batch
        for prev, interp in zip(prevs, interps):
            # Convert model output to RGB image (0-255 range)
            prev = utils.tensor_to_rgb(prev).cpu()
            interp = utils.tensor_to_rgb(interp).cpu()
            # Append original and interpolated frames
            frames.append(prev)
            frames.append(interp)

    # Combine all frames into single tensor [T, H, W, C]
    video_tensor = torch.stack(frames, dim=0)

    # Calculate output FPS:
    # Original FPS * 2 (since we insert interpolated frame between each original)
    normal_speed_fps = data.decoder.metadata.average_fps * 2
    # Apply speed multiplier (0.1 = 10% speed)
    fps = int(normal_speed_fps * args.speed)

    # Create output directory if missing
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename with parameters for traceability
    input_filename = os.path.basename(args.input_path).split('.')[0]
    output_path = os.path.join(
        args.output_dir,
        (
            f"test_{input_filename}_"          # Input filename
            f"{config.img_size}_"               # Output resolution
            f"FPS{fps}_"                        # Output framerate
            f"DDIM{args.ddim_steps}_"           # Sampling steps
            f"3D{config.use_3d}_"              # 3D architecture flag
            f"CM{config.channel_mult}_"         # Channel multiplier
            f"AH{config.attn_heads}.mp4"       # Attention heads
        )
    )

    # Write video to file with H.264 compression
    write_video(output_path, video_tensor, fps=fps, video_codec='h264')
