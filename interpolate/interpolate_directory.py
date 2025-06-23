import argparse
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Interpolation on a Directory of Videos')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for file in os.listdir(args.input_dir):
        print(f'Interpolating {file}')
        subprocess.run([
            'python', '-m' 'interpolate.interpolate_base',
            '--input_path', os.path.join(args.input_dir, file),
            '--output_dir', args.output_dir
        ])