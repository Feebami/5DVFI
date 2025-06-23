import argparse
import os
import sys

from utility import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    mean_psnr, mean_lpips, fid, frames = utils.evaluate_samples(args.dir)
    print('= ' * 20 + f'Eval for {args.dir}' + ' =' * 20)
    print(f'Mean PSNR:      {mean_psnr}')
    print(f'Mean LPIPS:     {mean_lpips}')
    print(f'FID:            {fid}')
    print(f'Total Frames:   {frames}')