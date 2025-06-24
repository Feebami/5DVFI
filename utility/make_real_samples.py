import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from utility import utils
from dataset.video_dataset import VFIDatasetSmall

if __name__ == '__main__':
    # Get list of UCF-101 video file paths (default 127 files)
    file_paths = utils.get_ucf_file_paths(127)
    
    # Prepare image transformation pipeline (resize, normalize, etc.)
    transform = utils.get_transform(240)

    # Create directories to save videos
    os.makedirs('eval_samples/real', exist_ok=True)

    # Process each video file
    for i, file in enumerate(sorted(file_paths)):
        # Create dataset for video frames
        data = VFIDatasetSmall(file, transform)
        
        # DataLoader for efficient batch loading
        dataloader = DataLoader(
            data,
            batch_size=16,               # Process 16 frames per batch
            shuffle=False,                # Maintain temporal order
            num_workers=4,                # Parallel data loading
            persistent_workers=True,      # Keep workers alive between batches
            prefetch_factor=2,            # Preload batches
            pin_memory=True               # Faster GPU transfer
        )

        # Define output paths for videos
        real_path = f'eval_samples/real/{os.path.basename(file).split(".")[0]}.mp4'

        real_list = []  # Store real middle frames
        
        # Iterate over batches with progress bar
        for batch in tqdm(dataloader, desc=f'Processing video {i+1}/{len(file_paths)}: {os.path.basename(file)}'):
            # Extract middle frames
            frame2s = batch[:, :, 1]
            
            # Convert ground truth middle frames to RGB images
            real_frames = torch.stack([
                utils.tensor_to_rgb(frame2)
                for frame2 in frame2s
            ])
            
            # Append batch frames to lists
            real_list.append(real_frames)

        # Concatenate all batches into single tensors
        real_tensor = torch.cat(real_list)  # [T, H, W, C] for real frames

        # Write videos to disk with 3 FPS and H.264 codec
        write_video(real_path, real_tensor, fps=3, video_codec='h264')