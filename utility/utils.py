import os
from statistics import mean
import sys

import torch
from torchvision.transforms import v2, functional as F
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchmetrics.image import FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transform(img_size):
    return v2.Compose([
        CropAndResize(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def get_ucf_file_paths(num_files):
    filepaths = [os.path.join('UCF_test', file) for file in os.listdir('UCF_test')]
    
    assert len(filepaths) == 127, 'UCF101 test files not all present'

    return filepaths[:num_files]

def evaluate_samples(sample_folder_path):
    sample_files = os.listdir(sample_folder_path)
    transform = get_transform(192)
    psnr = PeakSignalNoiseRatio().to(device)
    fid = FrechetInceptionDistance().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity().to(device)
    psnr_list = []
    lpips_list = []
    total_frames = 0
    for file in tqdm(sample_files, desc=f'Evaluating Files in {sample_folder_path}'):
        file_path = os.path.join(sample_folder_path, file)
        data = EvalDataset(file_path)
        dataloader = DataLoader(
            data,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True
        )

        total_frames += len(data)

        for batch in dataloader:
            reals, fakes = batch
            reals, fakes = reals.to(device), fakes.to(device)

            psnr_list.append(psnr(fakes, reals).item())

            lpips_list.append(lpips(transform(fakes), transform(reals)).item())

            fid.update(reals, real=True)
            fid.update(fakes, real=False)

    return mean(psnr_list), mean(lpips_list), fid.compute().item(), total_frames

@torch.no_grad()
def tensor_to_rgb(tensor):
    # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # Clamp values to [0, 1] range
    tensor = tensor.clamp(0, 1)
    # Rearrange from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)
    # Scale to [0, 255] range
    tensor = (tensor * 255).to(torch.uint8)
    return tensor

class EvalDataset(Dataset):
    def __init__(self, fake_file_path):
        super().__init__()
        self.real_file_path = os.path.join('eval_samples', 'real_192', os.path.basename(fake_file_path))
        self.fake_file_path = fake_file_path
        self.real_decoder = TrimmedDecoder(self.real_file_path)
        self.fake_decoder = TrimmedDecoder(self.fake_file_path)
        self.real_frames = self.real_decoder[:]
        self.fake_frames = self.fake_decoder[:]
        self.real_frames = v2.Resize(self.fake_frames.shape[-1])(self.real_frames)
        assert self.real_frames.shape == self.fake_frames.shape, 'Mismatch frame shape in EvalDataset'
        assert len(self.real_frames) == len(self.fake_frames), 'Mismatch frames length in EvalDataset'

    def __len__(self):
        return len(self.real_frames)

    def __getitem__(self, index):
        return self.real_frames[index], self.fake_frames[index]

class CropAndResize(v2.Transform):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, img):
        h, w = img.shape[-2:]
        small_dim = min(h, w)
        img = F.center_crop(img, small_dim)
        img = F.resize(img, self.size)
        return img
    
class TrimmedDecoder():
    def __init__(self, filepath):
        self.decoder = VideoDecoder(filepath)
        self.metadata = self.decoder.metadata
        self.decoder = self._trim()

    def _trim(self):
        for i in range(len(self.decoder) - 1):
            if not torch.equal(self[i], self[i+1]):
                return self.decoder[i:]
            
    def __len__(self):
        return len(self.decoder) - 1
    
    def __getitem__(self, index):
        return self.decoder[index]