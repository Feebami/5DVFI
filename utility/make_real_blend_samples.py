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
    file_paths = utils.get_ucf_file_paths(127)
    
    transform = utils.get_transform(192)

    os.makedirs(f'eval_samples/real', exist_ok=True)
    os.makedirs(f'eval_samples/blend', exist_ok=True)

    for i, file in enumerate(sorted(file_paths)):
        data = VFIDatasetSmall(file, transform)
        dataloader = DataLoader(
            data,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True
        )

        real_path = f'eval_samples/real/{os.path.basename(file).split(".")[0]}.mp4'
        blend_path = f'eval_samples/blend/{os.path.basename(file).split(".")[0]}.mp4'

        real_list = []
        blend_list = []
        for batch in tqdm(dataloader, desc=f'Processing video {i+1}/{len(file_paths)}: {os.path.basename(file)}'):
            frame1s = batch[:, :, 0]
            frame2s = batch[:, :, 1]
            frame3s = batch[:, :, 2]

            blends = (frame1s + frame3s) / 2.0
            
            real_frames = torch.stack([
                utils.tensor_to_rgb(frame2)
                for frame2 in frame2s
            ])
            
            blend_frames = torch.stack([
                utils.tensor_to_rgb(blend)
                for blend in blends
            ])
            real_list.append(real_frames)
            blend_list.append(blend_frames)

        real_tensor = torch.cat(real_list)
        blend_tensor = torch.cat(blend_list)

        write_video(real_path, real_tensor, fps=3, video_codec='h264')
        write_video(blend_path, blend_tensor, fps=3, video_codec='h264')