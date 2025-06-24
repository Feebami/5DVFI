import argparse
import os
import sys

from utility import utils  # Custom utility module for evaluation metrics

if __name__ == '__main__':
    # Argument parser setup for evaluation script
    parser = argparse.ArgumentParser(description='Evaluate generated video frames')
    # Directory containing generated samples to evaluate
    parser.add_argument('--dir', type=str, required=True,
                        help='Path to directory with generated samples')
    args = parser.parse_args()

    # Calculate evaluation metrics using utility function
    # Returns:
    #   mean_psnr  - Peak Signal-to-Noise Ratio (higher is better)
    #   mean_lpips - Learned Perceptual Image Patch Similarity (lower is better)
    #   fid        - Fréchet Inception Distance (lower is better)
    #   frames     - Total number of frames evaluated
    mean_psnr, mean_lpips, fid, frames = utils.evaluate_samples(args.dir)
    
    # Print evaluation results with visual separation
    print('= ' * 20 + f'Eval for {args.dir}' + ' =' * 20)
    print(f'Mean PSNR:      {mean_psnr:.4f}')   # Image quality metric (dB)
    print(f'Mean LPIPS:     {mean_lpips:.4f}')  # Perceptual similarity metric
    print(f'FID:            {fid:.4f}')         # Distribution similarity metric
    print(f'Total Frames:   {frames}')          # Number of frames processed
