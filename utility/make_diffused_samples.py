import argparse
import os
import sys

from omegaconf import OmegaConf
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs.yaml')
    parser.add_argument('--ckpt', type=str, default='epoch=5-step=80000.ckpt')
    parser.add_argument('--num_files', type=int, default=127)
    parser.add_argument('--ddim_steps', type=int, default=32)
    args = parser.parse_args()

    sample_folder = os.path.join('eval_samples', 'diffused')
    os.makedirs(sample_folder, exist_ok=True)

    config = OmegaConf.load(args.config)

    files = utils.get_ucf_file_paths(num_files=args.num_files)

    assert all(os.path.basename(f).split('.')[0]+'.mp4' in os.listdir(f'eval_samples/real') for f in files), "Not all files are present in real files"

    module = VFIDDPM.load_from_checkpoint(args.ckpt, config=config)
    module.eval()

    sampler = DDIMSampler(module)

    for file in tqdm(files, desc=os.path.basename(args.config).split('.')[0]):
        decoder = utils.TrimmedDecoder(file)
        small_dim = min(decoder.metadata.width, decoder.metadata.height)

        transform = utils.get_transform(config.img_size)

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

        diffusion_frames = []
        for batch in dataloader:
            frame1s = batch[:, :, 0]
            frame3s = batch[:, :, 2]

            interps = sampler(frame1s, frame3s, eta=1.0, num_steps=args.ddim_steps)

            batch_frames = [sampler.tensor_to_rgb(interp).cpu() for interp in interps]
            diffusion_frames.extend(batch_frames)

        diffused_tensor = torch.stack(diffusion_frames)

        file_path = os.path.join(sample_folder, os.path.basename(file).split(".")[0]) + '.mp4'

        write_video(
            file_path,
            diffused_tensor,
            fps=3,
            video_codec='h264'
        )