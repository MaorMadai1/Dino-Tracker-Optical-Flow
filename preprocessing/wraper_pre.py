import os
import subprocess

# Base directory containing all video directories
base_dir = "./dataset/tapvid-davis/davis_480"

# Path to the preprocessing script
preprocessing_script = "./preprocessing/main_preprocessing.py"

# Path to the config file
config_path = "./config/preprocessing.yaml"

# Iterate through all subdirectories in the base directory
for video_dir in sorted(os.listdir(base_dir)):
    video_dir_path = os.path.join(base_dir, video_dir)

    # Skip if it's not a directory
    if not os.path.isdir(video_dir_path):
        continue

    print(f"Processing video directory: {video_dir_path}", flush=True)

    # Run the preprocessing script for the current video directory
    args = [
        "python", preprocessing_script,
        "--config", config_path,
        "--data-path", video_dir_path
    ]
    print(f"Running command: {' '.join(args)}", flush=True)
    subprocess.run(args)