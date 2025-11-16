import os
import subprocess

# Base directory containing all video directories
base_dir = "./dataset/tapvid-davis/davis_480"

# Path to the train script
train_script = "./train.py"

# Path to the config file
config_path = "./config/train.yaml"

# Iterate through all subdirectories in the base directory
for video_dir in sorted(os.listdir(base_dir)):
    # Ensure the directory name is numeric and within the range 0-5
    if not video_dir.isdigit() or not (23 <= int(video_dir) <= 29):
        continue

    video_dir_path = os.path.join(base_dir, video_dir)

    # Skip if it's not a directory
    if not os.path.isdir(video_dir_path):
        continue

    print(f"Processing video directory: {video_dir_path}", flush=True)

    # Run the train script for the current video directory
    args = [
        "python", train_script,
        "--config", config_path,
        "--data-path", video_dir_path
    ]
    print(f"Running command: {' '.join(args)}", flush=True)
    subprocess.run(args)