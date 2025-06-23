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

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion VFI Sampler')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ckpt', type=str, default='epoch=5-step=80000.ckpt')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--ddim_steps', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='samples')
    parser.add_argument('--speed', type=float, default=0.1)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    decoder = utils.TrimmedDecoder(args.input_path)
    small_dim = min(decoder.metadata.width, decoder.metadata.height)

    transform = utils.get_transform(config.img_size, small_dim)

    data = VFIDatasetSmall(args.input_path, transform)
    dataloader = DataLoader(
        data,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True
    )

    module = VFIDDPM.load_from_checkpoint(args.ckpt, config=config)
    module.eval()

    sampler = DDIMSampler(module)

    frames = []
    for batch in tqdm(dataloader, desc='Interpolating...'):
        prevs = batch[:, :, 0]
        subseq = batch[:, :, 1]

        interps = sampler(prevs, subseq, eta=1.0, num_steps=args.ddim_steps)
        
        for prev, interp in zip(prevs, interps):
            prev = sampler.tensor_to_rgb(prev).cpu()
            interp = sampler.tensor_to_rgb(interp).cpu()
            frames.append(prev)
            frames.append(interp)

    video_tensor = torch.stack(frames, dim=0)

    normal_speed_fps = data.decoder.metadata.average_fps * 2
    fps = int(normal_speed_fps * args.speed)

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(
        args.output_dir,
        (
            f"test_"
            f"{args.input_path.split('/')[-1].split('.')[0]}_"
            f"{config.img_size}_"
            f"FPS{fps}_DDIM{args.ddim_steps}_"
            f"3D{config.use_3d}_CM{config.channel_mult}_AH{config.attn_heads}.mp4"
        )
    )

    write_video(output_path, video_tensor, fps=fps, video_codec='h264')