from multiprocessing import Pool
import os

import numpy as np
import torch
from torchvision.io import write_jpeg

from utility.utils import TrimmedDecoder

def tripletize(video_path, folder):
    """Extract frame triplets from a video and save as JPEG images"""
    print(f'Processing {video_path}')
    # Initialize video decoder
    decoder = TrimmedDecoder(video_path)
    # Create subfolder using video filename (without extension)
    subfolder = os.path.basename(video_path).split('.')[0]
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
    
    # Iterate through all possible triplets in video
    for i in range(len(decoder) - 2):
        # Create zero-padded index for triplet folder
        subsubfolder = f'{i:05d}'
        # Extract 3 consecutive frames
        frames = decoder[i:i+3]
        
        # Skip triplets where first and second frames are identical
        # (avoids processing static scenes or duplicate frames)
        if torch.equal(frames[0], frames[1]):
            continue
            
        # Create directory for current triplet
        os.makedirs(os.path.join(folder, subfolder, subsubfolder), exist_ok=True)
        
        # Save each frame in triplet as JPEG
        for j, frame in enumerate(frames):
            write_jpeg(
                frame,  # Frame tensor (C, H, W)
                os.path.join(folder, subfolder, subsubfolder, f'{j}.jpg'),
                quality=100,  # Maximum quality to avoid compression artifacts
            )

# Get all video paths from UCF-101 dataset
filepaths = [
    os.path.join('UCF-101', folder, file)
    for folder in os.listdir('UCF-101')  # Action categories
    for file in os.listdir(os.path.join('UCF-101', folder))  # Videos in category
]

# Verify total video count (13320 total - 128 excluded = 13192)
assert len(filepaths) == 13320 - 128, "Unexpected number of video files"

# Randomly select 1024 videos for processing
np.random.seed(42)  # Ensure reproducibility
filepaths = np.random.choice(filepaths, size=1024, replace=False).tolist()

# Create output directory for triplets
folder = 'UCF-101_triplet'
os.makedirs(folder, exist_ok=True)

# Process videos in parallel using 6 workers
with Pool(processes=6) as pool:
    # Prepare tasks: (video_path, output_folder)
    tasks = [(file, folder) for file in sorted(filepaths)]
    # Distribute tasks across workers
    pool.starmap(tripletize, tasks)
