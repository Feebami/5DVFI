import argparse
import os
import subprocess

if __name__ == '__main__':
    # Argument parser for batch processing configuration
    parser = argparse.ArgumentParser(description='Perform Interpolation on a Directory of Videos')
    # Input directory containing video files to process
    parser.add_argument('--input_dir', type=str, required=True)
    # Output directory to save processed videos
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate through all files in input directory
    for file in os.listdir(args.input_dir):
        print(f'Interpolating {file}')
        
        # Execute interpolation command for current video file
        subprocess.run([
            'python', 
            '-m', 
            'interpolate.interpolate_base',  # Module containing interpolation script
            '--input_path', 
            os.path.join(args.input_dir, file),  # Full path to input video
            '--output_dir', 
            args.output_dir  # Directory to save processed video
        ])
