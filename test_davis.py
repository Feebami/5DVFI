import os
from statistics import mean
import sys

from omegaconf import OmegaConf
import torch
from torchvision.transforms import v2, functional as F
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchmetrics.image import FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchvision.io import decode_image
from tqdm import tqdm

from base_model.train import VFIDDPM
from base_model.samplers import DDIMSampler
from utility.utils import get_transform

# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DavisData(Dataset):
    """Dataset for UCF extraceted test sequences"""
    def __init__(self, folder, transform):
        super().__init__()
        # Base directory containing video triplets
        self.folder_path = 'davis-90/480p/' + folder
        self.transform = transform  # Image transformations
        # Build list of all sequence paths
        self.filepaths = [
            os.path.join(self.folder_path, file)
            for file in sorted(os.listdir(self.folder_path))
        ]
        
    def __len__(self):
        """Total number of video sequences"""
        return len(self.filepaths) - 2
    
    def __getitem__(self, idx):
        """Load and process a video triplet"""
        files = self.filepaths[idx:idx+3]
        imgs = [decode_image(file) for file in files]
        imgs = [self.transform(img) for img in imgs]
        return torch.stack(imgs, 1)
    
if __name__ == '__main__':
    transform = get_transform(480)
    config = OmegaConf.load('config.yaml')

    module = VFIDDPM.load_from_checkpoint('epoch=5-step=80000.ckpt', config=config)
    module.eval().to(device)

    sampler = DDIMSampler(module)

    psnr = PeakSignalNoiseRatio().to(device)  # PSNR: Higher is better
    fid = FrechetInceptionDistance(normalize=True).to(device)  # FID: Lower is better
    lpips = LearnedPerceptualImagePatchSimilarity().to(device)  # LPIPS: Lower is better

    psnr_list = []
    lpips_list = []
    total_frames = 0
    for folder in tqdm(os.listdir('davis-90/480p')):
        data = DavisData(folder, transform)
        dataloader = DataLoader(
            data,
            batch_size=8,                # Process 32 frames per batch
            shuffle=False,                # Maintain temporal order
            num_workers=4,                # Parallel data loading
            persistent_workers=True,      # Maintain workers between batches
            prefetch_factor=2,            # Preload next batches
            pin_memory=True  
        )


        # Initialize metric containers

        for batch in dataloader:
            batch = batch.to(device)
            i1 = batch[:, :, 0]
            i2 = batch[:, :, 1]
            i3 = batch[:, :, 2]

            interps = sampler(i1, i3, eta=1.0, num_steps=16).clamp(-1, 1)

            total_frames += len(batch)

            psnr_list.append(psnr(interps, i2).item())
            lpips_list.append(lpips(interps, i2).item())

            interps = interps * 0.5 + 0.5
            interps = interps.clamp(0, 1)

            i2 = i2 * 0.5 + 0.5
            i2 = i2.clamp(0, 1)

            # import matplotlib.pyplot as plt

            # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            # axs[0].imshow(interps[0].permute(1, 2, 0).cpu().detach().numpy())
            # axs[0].set_title('Interpolated')
            # axs[0].axis('off')
            # axs[1].imshow(i2[0].permute(1, 2, 0).cpu().detach().numpy())
            # axs[1].set_title('Ground Truth')
            # axs[1].axis('off')
            # plt.tight_layout()
            # plt.show()

            fid.update(i2, real=True)
            fid.update(interps, real=False)

    print('Mean PSNR:', mean(psnr_list))
    print('Mean LPIPS:', mean(lpips_list))
    print('FID:', fid.compute().item())
    print('Total frames:', total_frames)