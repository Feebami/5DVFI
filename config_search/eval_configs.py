import argparse
from multiprocessing import Pool
import os
import subprocess
import sys

def eval_config(config_path, ckpt_path, num_files):
    # Extract base configuration name from file path (without extension)
    config_name = os.path.basename(config_path).split('.')[0]
    print(f'Evaluating {config_name} with checkpoint {ckpt_path}')

    # Generate interpolated frames using trained diffusion model
    subprocess.run([
        'python', '-m', 'utility.make_diffused_samples',
        '--config', config_path,      # Model configuration file
        '--ckpt', ckpt_path,          # Model checkpoint file
        '--num_files', str(num_files), # Number of samples to generate
    ])

    # Evaluate generated samples using quantitative metrics
    eval_dir = os.path.join('eval_samples', 'diffused', config_name)
    subprocess.run([
        'python', '-m', 'utility.eval_sample_dir',
        '--dir', eval_dir,            # Directory containing generated samples
    ])

if __name__ == '__main__':
    # Command-line argument parser configuration
    parser = argparse.ArgumentParser()
    # Directory containing model configuration files (YAML format)
    parser.add_argument('--config_dir', type=str, default='config_search/configs')
    # Number of sample files to generate per configuration
    parser.add_argument('--num_files', type=int, required=True)
    # Checkpoint version to use (0 = latest version)
    parser.add_argument('--ckpt_version', type=int, default=0, help='Version of the checkpoint to use')
    args = parser.parse_args()

    # Get sorted list of configuration files in directory
    config_files = sorted(os.listdir(args.config_dir))
    # Build full paths to each configuration file
    config_paths = [os.path.join(args.config_dir, file) for file in config_files]
    # Generate corresponding log directory names for each config
    config_logs = [file.split('.')[0] + '_logs' for file in config_files]
    # Build checkpoint folder paths based on Lightning versioning
    ckpt_folders = [
        log_folder + f'/lightning_logs/version_{args.ckpt_version}/checkpoints' 
        for log_folder in config_logs
    ]
    # Find latest checkpoint file in each checkpoint folder
    ckpt_paths = [
        os.path.join(ckpt_folder, sorted(os.listdir(ckpt_folder))[-1]) 
        for ckpt_folder in ckpt_folders
    ]

    # Prepare arguments for parallel processing: (config_path, ckpt_path, num_files)
    args_list = list(zip(config_paths, ckpt_paths, [args.num_files] * len(config_paths)))

    # Execute evaluations in parallel using 2 worker processes
    with Pool(processes=2) as pool:
        # Distribute evaluation tasks across workers
        pool.starmap(eval_config, args_list)
