import os
from statistics import mean
import sys

import torch
from torchvision.transforms import v2, functional as F
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchmetrics.image import FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from tqdm import tqdm

# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transform(img_size):
    """Create image transformation pipeline for preprocessing
    Args:
        img_size (int): Target image size for resizing
    Returns:
        v2.Compose: Composed transform pipeline
    """
    return v2.Compose([
        CropAndResize(img_size),  # Center crop to square and resize
        v2.ToDtype(torch.float32, scale=True),  # Convert to float32 in [0,1]
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1,1]
    ])

def get_ucf_file_paths(num_files):
    """Get paths to UCF-101 test videos
    Args:
        num_files (int): Number of files to return
    Returns:
        list: Paths to UCF-101 test videos
    """
    # List all files in UCF_test directory
    filepaths = [os.path.join('UCF_test', file) for file in os.listdir('UCF_test')]
    
    # UCF-101 test set should have exactly 127 videos
    assert len(filepaths) == 127, 'UCF101 test files not all present'

    return filepaths[:num_files]  # Return requested number of files

def evaluate_samples(sample_folder_path):
    """Evaluate generated video frames against ground truth
    Args:
        sample_folder_path (str): Path to directory with generated samples
    Returns:
        tuple: (mean_psnr, mean_lpips, fid_score, total_frames)
    """
    # List all files in sample directory
    sample_files = os.listdir(sample_folder_path)
    
    # Initialize evaluation metrics
    transform = get_transform(192)  # Transform for LPIPS compatibility
    psnr = PeakSignalNoiseRatio().to(device)  # PSNR: Higher is better
    fid = FrechetInceptionDistance().to(device)  # FID: Lower is better
    lpips = LearnedPerceptualImagePatchSimilarity().to(device)  # LPIPS: Lower is better
    
    # Initialize metric containers
    psnr_list = []
    lpips_list = []
    total_frames = 0
    
    # Process each sample file
    for file in tqdm(sample_files, desc=f'Evaluating Files in {sample_folder_path}'):
        file_path = os.path.join(sample_folder_path, file)
        
        # Create dataset with real and generated frames
        data = EvalDataset(file_path)
        
        # Configure dataloader for efficient batch processing
        dataloader = DataLoader(
            data,
            batch_size=16,                # Process 16 frames per batch
            shuffle=False,                # Maintain temporal order
            num_workers=4,                # Parallel data loading
            persistent_workers=True,      # Maintain workers between batches
            prefetch_factor=2,            # Preload next batches
            pin_memory=True               # Faster data transfer to GPU
        )

        total_frames += len(data)  # Accumulate total frame count

        # Process frames in batches
        for batch in dataloader:
            reals, fakes = batch
            reals, fakes = reals.to(device), fakes.to(device)

            # Calculate PSNR (pixel-level similarity)
            psnr_list.append(psnr(fakes, reals).item())
            
            # Calculate LPIPS (perceptual similarity)
            lpips_list.append(lpips(transform(fakes), transform(reals)).item())
            
            # Update FID statistics
            fid.update(reals, real=True)   # Add real images to FID distribution
            fid.update(fakes, real=False)  # Add generated images to FID distribution

    # Return computed metrics and total frame count
    return mean(psnr_list), mean(lpips_list), fid.compute().item(), total_frames

@torch.no_grad()
def tensor_to_rgb(tensor):
    """Convert normalized tensor to displayable RGB image
    Args:
        tensor (torch.Tensor): Normalized tensor in [-1,1] range
    Returns:
        torch.Tensor: RGB image tensor in [0,255] range (uint8)
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # Ensure values within valid range
    tensor = tensor.clamp(0, 1)
    # Change from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)
    # Convert to 8-bit unsigned integer
    tensor = (tensor * 255).to(torch.uint8)
    return tensor

class EvalDataset(Dataset):
    """Dataset for paired evaluation of real vs generated frames"""
    def __init__(self, fake_file_path):
        super().__init__()
        # Path to ground truth video
        self.real_file_path = os.path.join('eval_samples', 'real', os.path.basename(fake_file_path))
        # Path to generated video
        self.fake_file_path = fake_file_path
        
        # Initialize decoders with frame trimming
        self.real_decoder = TrimmedDecoder(self.real_file_path)
        self.fake_decoder = TrimmedDecoder(self.fake_file_path)
        
        # Load all frames
        self.real_frames = self.real_decoder[:]
        self.fake_frames = self.fake_decoder[:]
        
        # Resize real frames to match generated frame size
        self.real_frames = v2.Resize(self.fake_frames.shape[-1])(self.real_frames)
        
        # Ensure frame dimensions match
        assert self.real_frames.shape == self.fake_frames.shape, 'Mismatch frame shape in EvalDataset'
        assert len(self.real_frames) == len(self.fake_frames), 'Mismatch frames length in EvalDataset'

    def __len__(self):
        """Number of frames in dataset"""
        return len(self.real_frames)

    def __getitem__(self, index):
        """Get real and generated frame pair"""
        return self.real_frames[index], self.fake_frames[index]

class CropAndResize(v2.Transform):
    """Transform to center crop and resize images"""
    def __init__(self, size):
        super().__init__()
        self.size = size  # Target image size

    def __call__(self, img):
        """Apply center crop and resize"""
        h, w = img.shape[-2:]  # Get image dimensions
        small_dim = min(h, w)  # Determine smaller dimension
        img = F.center_crop(img, small_dim)  # Center crop to square
        img = F.resize(img, self.size)  # Resize to target size
        return img
    
class TrimmedDecoder():
    """Video decoder that trims initial duplicate frames"""
    def __init__(self, filepath):
        # Initialize video decoder
        self.decoder = VideoDecoder(filepath)
        # Store video metadata
        self.metadata = self.decoder.metadata
        # Trim duplicate frames from start
        self.decoder = self._trim()

    def _trim(self):
        """Remove duplicate frames at video start"""
        # Find first frame that differs from next frame
        for i in range(len(self.decoder) - 1):
            if not torch.equal(self[i], self[i+1]):
                # Return from first non-duplicate frame onward
                return self.decoder[i:]
            
    def __len__(self):
        """Number of frames after trimming"""
        return len(self.decoder) - 1
    
    def __getitem__(self, index):
        """Get frame at specified index"""
        return self.decoder[index]
