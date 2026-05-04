import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
import yaml
from eval.metrics import compute_tapvid_metrics_for_video, compute_badja_metrics_for_video
from utils import get_dino_embed_dir


def eval_dataset(args):
    benchmark_data = pickle.load(open(args.benchmark_pickle_path, "rb"))

    # Resolve the per-layer subfolder used by the benchmark step (visualize_raw_heatmaps.py),
    # so eval reads from output_folder/davis_480/<id>/l{dino_layer}/...
    with open(args.config, "r") as _f:
        _preproc_cfg = yaml.safe_load(_f.read())
    dino_layer = _preproc_cfg["dino_layer"]
    embed_dir = get_dino_embed_dir(dino_layer)
    print(f"[eval_benchmark] dino_layer={dino_layer}, reading from subfolder='{embed_dir}'")

    metrics_list = []

        # Define the video sizes for each video index
    # video_sizes = {
    #     0: [854, 480], 1: [854, 480], 2: [854, 480], 3: [854, 480], 4: [854, 480],
    #     5: [854, 480], 6: [854, 480], 7: [854, 480], 8: [854, 480], 9: [854, 480],
    #     10: [854, 480], 11: [854, 480], 12: [854, 480], 13: [854, 480], 14: [854, 480],
    #     15: [854, 480], 16: [854, 480], 17: [854, 480], 18: [910, 480], 19: [1152, 480],
    #     20: [854, 480], 21: [854, 480], 22: [854, 480], 23: [854, 480], 24: [854, 480],
    #     25: [854, 480], 26: [854, 480], 27: [854, 480], 28: [854, 480], 29: [854, 480]
    # }
    #video_sizes = {19: [1152, 480], 18: [910, 480]}
    dataset_root = args.dataset_root_dir
    for video_idx_str in tqdm(os.listdir(dataset_root), desc="Evaluating dataset"):
        if video_idx_str.startswith("."):
            continue
        video_dir = os.path.join(dataset_root, video_idx_str, embed_dir)
        trajectories_dir = os.path.join(video_dir, "dinov2_grid_trajectory")
        occlusions_dir = os.path.join(video_dir, "dinov2_grid_occlusion")
        time_taken_dir = os.path.join(video_dir, "dinov2_grid_time_taken")
        video_idx = int(video_idx_str)

        # Get the video size for the current video index
        #pred_video_size = video_sizes.get(video_idx, [854, 476])  # Default to [854, 480] if not found

        if args.dataset_type == "tapvid":
            metrics = compute_tapvid_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                        model_occ_pred_dir=occlusions_dir,
                                                        video_idx=video_idx,
                                                        benchmark_data=benchmark_data,
                                                        pred_video_sizes=[854, 476],
                                                        optical_flow_opt=args.optical_flow_opt) # set to None to get the video size from the benchmark data
        elif args.dataset_type == "BADJA":
            metrics = compute_badja_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                      video_idx=video_idx,
                                                      benchmark_data=benchmark_data,
                                                      pred_video_sizes=[854, 480])
        else:
            raise ValueError("Invalid dataset type. Must be either tapvid or BADJA")
        
        # Load time taken data and compute average
        of_flag = 1 if args.optical_flow_opt else 0
        time_taken_file = os.path.join(time_taken_dir, f"time_taken_0_of{of_flag}.npy")
        if os.path.exists(time_taken_file):
            time_taken_data = np.load(time_taken_file)
            metrics["avg_time_per_point"] = float(np.mean(time_taken_data))
        else:
            metrics["avg_time_per_point"] = None
        
        metrics["video_idx"] = int(video_idx)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df['video_idx'] = metrics_df['video_idx'].astype(int)
    metrics_df.set_index('video_idx', inplace=True)
    metrics_df.sort_index(inplace=True)
    metrics_df.loc['average', :] = metrics_df.mean()
    metrics_df.to_csv(args.out_file)
    print("Total metrics:") 
    print(metrics_df.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root-dir", default="output_folder/davis_480", type=str)
    parser.add_argument("--benchmark-pickle-path", default="tapvid/tapvid_davis_data_strided.pkl", type=str)
    parser.add_argument("--out-file", default="output_folder/dinov2_metrics.csv", type=str)
    parser.add_argument("--dataset-type", default="tapvid", type=str, help="Dataset type: tapvid or BADJA")
    parser.add_argument("--optical-flow-opt", action="store_true", help="Whether optical flow optimization was used")
    parser.add_argument("--config", default="./config/preprocessing.yaml", type=str,
                        help="Preprocessing config (used to derive the DINO layer that selects the per-video subfolder)")
    args = parser.parse_args()
    eval_dataset(args)
