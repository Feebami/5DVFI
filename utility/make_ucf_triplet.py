import argparse
from multiprocessing import Pool
import os

import numpy as np
import torch
from torchvision.io import write_jpeg
from torchvision.transforms import v2, functional
from tqdm import tqdm

from utility.utils import TrimmedDecoder

def tripletize(video_path, folder):
    print(f'Processing {video_path}')
    decoder = TrimmedDecoder(video_path)
    subfolder = os.path.basename(video_path).split('.')[0]
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
    for i in range(len(decoder) - 2):
        subsubfolder = f'{i:05d}'
        frames = decoder[i:i+3]
        if torch.equal(frames[0], frames[1]):
            continue
        os.makedirs(os.path.join(folder, subfolder, subsubfolder), exist_ok=True)
        for j, frame in enumerate(frames):
            write_jpeg(
                frame,
                os.path.join(folder, subfolder, subsubfolder, f'{j}.jpg'),
                quality=100,
            )

filepaths = [
    os.path.join('UCF-101', folder, file)
    for folder in os.listdir('UCF-101')
    for file in os.listdir(os.path.join('UCF-101', folder))
]

assert len(filepaths) == 13320 - 128

# Get 1024 random files
np.random.seed(42)
filepaths = np.random.choice(filepaths, size=1024, replace=False).tolist()

folder = 'UCF-101_triplet'
os.makedirs(folder, exist_ok=True)

with Pool(processes=6) as pool:
    tasks = [(file, folder) for file in sorted(filepaths)]
    pool.starmap(tripletize, tasks)