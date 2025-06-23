import os

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from utility.utils import TrimmedDecoder

class VFIDatasetSmall(Dataset):
    """Dataset for loading small video sequences from a single file"""
    def __init__(self, filepath, transform, num_frames=None):
        super().__init__()
        self.filepath = filepath  # Path to video file
        self.transform = transform  # Image transformations
        self.num_frames = num_frames  # Optional frame limit
        # Initialize video decoder
        self.decoder = TrimmedDecoder(filepath)
        # Load and transform video frames
        self.frames = self._load_video_frames()
        self.frames = [self.transform(frame) for frame in self.frames]
    
    def _load_video_frames(self):
        """Load frames from video using decoder"""
        # Determine end frame index (-1 = all frames)
        end_frame = self.num_frames if self.num_frames else -1
        # Load frames from decoder
        frames = self.decoder[:end_frame]
        return frames
    
    def __len__(self):
        """Number of valid triplets (3 consecutive frames)"""
        # Each sample requires 3 consecutive frames
        return len(self.frames) - 2
    
    def __getitem__(self, idx):
        """Get 3 consecutive frames as a tensor"""
        # Extract frame triplet [idx, idx+1, idx+2]
        frames = self.frames[idx:idx+3]
        # Stack along new dimension: [C, 3, H, W]
        frames = torch.stack(frames, dim=1)
        return frames
    
class V90KDataset(Dataset):
    """Dataset for Vimeo-90K triplet sequences"""
    def __init__(self, transform):
        super().__init__()
        # Base directory containing video triplets
        self.base_dir = 'vimeo_triplet/sequences'
        self.transform = transform  # Image transformations
        # Build list of all sequence paths
        self.filepaths = [
            os.path.join(self.base_dir, folder, subfolder)
            for folder in sorted(os.listdir(self.base_dir))
            for subfolder in sorted(os.listdir(os.path.join(self.base_dir, folder)))
        ]
        
    def __len__(self):
        """Total number of video sequences"""
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        """Load and process a video triplet"""
        # Get all images in sequence folder (sorted)
        imgs = [
            os.path.join(self.filepaths[idx], file) 
            for file in sorted(os.listdir(self.filepaths[idx]))
        ]
        # Decode images to tensors
        imgs = [decode_image(img) for img in imgs]
        # Apply transformations
        imgs = [self.transform(img) for img in imgs]
        # Stack along new dimension: [C, 3, H, W]
        return torch.stack(imgs, dim=1)
    
class UCF90KCombinedDataset(Dataset):
    """Combined dataset of UCF-101 and Vimeo-90K triplets"""
    def __init__(self, transform):
        super().__init__()
        # UCF-101 triplet directory
        self.ucf_base_dir = 'UCF-101_triplet'
        # Vimeo-90K triplet directory
        self.v90k_base_dir = 'vimeo_triplet/sequences'
        self.transform = transform  # Image transformations
        # Collect all UCF-101 sequence paths
        self.ucf_files = [
            os.path.join(self.ucf_base_dir, folder, subfolder)
            for folder in sorted(os.listdir(self.ucf_base_dir))
            for subfolder in sorted(os.listdir(os.path.join(self.ucf_base_dir, folder)))
        ]
        # Collect all Vimeo-90K sequence paths
        self.v90k_files = [
            os.path.join(self.v90k_base_dir, folder, subfolder)
            for folder in sorted(os.listdir(self.v90k_base_dir))
            for subfolder in sorted(os.listdir(os.path.join(self.v90k_base_dir, folder)))
        ]
        # Combine both datasets
        self.files = self.ucf_files + self.v90k_files

    def __len__(self):
        """Total number of sequences in combined dataset"""
        return len(self.files)
    
    def __getitem__(self, idx):
        """Load and process a video triplet from combined dataset"""
        # Get path to sequence folder
        file_path = self.files[idx]
        # Get all images in sequence (sorted)
        imgs = [
            os.path.join(file_path, file) 
            for file in sorted(os.listdir(file_path))
        ]
        # Decode images to tensors
        imgs = [decode_image(img) for img in imgs]
        # Apply transformations
        imgs = [self.transform(img) for img in imgs]
        # Stack along new dimension: [C, 3, H, W]
        return torch.stack(imgs, dim=1)
