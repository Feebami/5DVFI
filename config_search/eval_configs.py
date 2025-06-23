import argparse
from multiprocessing import Pool
import os
import subprocess
import sys

import pandas as pd

from utility import utils

def eval_config(config_path, ckpt_path, num_files):
    config_name = os.path.basename(config_path).split('.')[0]
    print(f'Evaluating {config_name} with checkpoint {ckpt_path}')

    # Make diffused samples
    subprocess.run([
        'python', '-m', 'utility.make_diffused_samples',
        '--config', config_path,
        '--ckpt', ckpt_path,
        '--num_files', str(num_files),
    ])

    # Evaluate samples
    eval_dir = os.path.join('eval_samples', 'diffused', config_name)
    subprocess.run([
        'python', '-m', 'utility.eval_sample_dir',
        '--dir', eval_dir,
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='config_search/configs')
    parser.add_argument('--num_files', type=int, required=True)
    parser.add_argument('--ckpt_version', type=int, default=0, help='Version of the checkpoint to use')
    args = parser.parse_args()

    config_files = sorted(os.listdir(args.config_dir))
    config_paths = [os.path.join(args.config_dir, file) for file in config_files]
    config_logs = [file.split('.')[0] + '_logs' for file in config_files]
    ckpt_folders = [log_folder + f'/lightning_logs/version_{args.ckpt_version}/checkpoints' for log_folder in config_logs]
    ckpt_paths = [os.path.join(ckpt_folder, sorted(os.listdir(ckpt_folder))[-1]) for ckpt_folder in ckpt_folders]

    args_list = list(zip(config_paths, ckpt_paths, [args.num_files] * len(config_paths)))


    with Pool(processes=2) as pool:
        pool.starmap(eval_config, args_list)