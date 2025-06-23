import os

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from utility.utils import TrimmedDecoder

class VFIDatasetSmall(Dataset):
    def __init__(self, filepath, transform, num_frames=None):
        super().__init__()
        self.filepath = filepath
        self.transform = transform
        self.num_frames = num_frames
        self.decoder = TrimmedDecoder(filepath)
        self.frames = self._load_video_frames()
        self.frames = [self.transform(frame) for frame in self.frames]
    
    def _load_video_frames(self):
        end_frame = self.num_frames if self.num_frames else -1
        frames = self.decoder[:end_frame]
        return frames
    
    def __len__(self):
        # Number of samples is number of frames minus 2 (for 3 sequential frames)
        return len(self.frames) - 2
    
    def __getitem__(self, idx):
        # Get 3 sequential frames
        frames = self.frames[idx:idx+3]
        # Stack frames along a new dimension
        frames = torch.stack(frames, dim=1)
        return frames
    
class V90KDataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.base_dir = 'vimeo_triplet/sequences'
        self.transform = transform
        self.filepaths = [os.path.join(self.base_dir, folder, subfolder)
                     for folder in sorted(os.listdir(self.base_dir))
                     for subfolder in sorted(os.listdir(os.path.join(self.base_dir, folder)))]
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        imgs = [os.path.join(self.filepaths[idx], file) for file in sorted(os.listdir(self.filepaths[idx]))]
        imgs = [decode_image(img) for img in imgs]
        imgs = [self.transform(img) for img in imgs]
        return torch.stack(imgs, dim=1)
    
class UCF90KCombinedDataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.ucf_base_dir = 'UCF-101_triplet'
        self.v90k_base_dir = 'vimeo_triplet/sequences'
        self.transform = transform
        self.ucf_files = [
            os.path.join(self.ucf_base_dir, folder, subfolder)
            for folder in sorted(os.listdir(self.ucf_base_dir))
            for subfolder in sorted(os.listdir(os.path.join(self.ucf_base_dir, folder)))
        ]
        self.v90k_files = [
            os.path.join(self.v90k_base_dir, folder, subfolder)
            for folder in sorted(os.listdir(self.v90k_base_dir))
            for subfolder in sorted(os.listdir(os.path.join(self.v90k_base_dir, folder)))
        ]
        self.files = self.ucf_files + self.v90k_files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        imgs = [os.path.join(file_path, file) for file in sorted(os.listdir(file_path))]
        imgs = [decode_image(img) for img in imgs]
        imgs = [self.transform(img) for img in imgs]
        return torch.stack(imgs, dim=1)