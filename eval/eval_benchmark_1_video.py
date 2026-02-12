import argparse
from tqdm import tqdm
import pandas as pd
import os
import pickle
from eval.metrics import compute_tapvid_metrics_for_video, compute_badja_metrics_for_video


def eval_single_video(args):
    benchmark_data = pickle.load(open(args.benchmark_pickle_path, "rb"))

    # Directly use the provided paths and video index for the single video evaluation
    trajectories_dir = args.trajectories_dir
    occlusions_dir = args.occlusions_dir
    video_idx = args.video_idx

    if args.dataset_type == "tapvid":
        metrics = compute_tapvid_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                    model_occ_pred_dir=occlusions_dir,
                                                    video_idx=video_idx,
                                                    benchmark_data=benchmark_data,
                                                    pred_video_sizes=[854, 476],
                                                    optical_flow_opt=args.optical_flow_opt)
    elif args.dataset_type == "BADJA":
        metrics = compute_badja_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                  video_idx=video_idx,
                                                  benchmark_data=benchmark_data,
                                                  pred_video_sizes=[854, 476])
    else:
        raise ValueError("Invalid dataset type. Must be either tapvid or BADJA")
    
    # Prepare results and save them to the output file
    metrics["video_idx"] = int(video_idx)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.set_index('video_idx', inplace=True)
    metrics_df.to_csv(args.out_file)
    print("Metrics for the single video:")
    print(metrics_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-dir", type=str,default="output_folder/davis_480/0/dinov2_sift_opt_grid_trajectory", help="Directory containing the trajectories")
    parser.add_argument("--occlusions-dir", type=str, default="output_folder/davis_480/0/dinov2_sift__opt_grid_occlusion", help="Directory containing the occlusions")
    parser.add_argument("--video-idx", type=int, default=0, help="Index of the video to evaluate")
    parser.add_argument("--benchmark-pickle-path", default="/home/dor.danino/dino-tracker/tapvid/tapvid_davis_data_strided.pkl", type=str)
    parser.add_argument("--out-file", default="dataset/tapvid-davis/davis_480/0/dinov2_sift_opt_comp_metrics.csv", type=str)
    parser.add_argument("--dataset-type", default="tapvid", type=str, help="Dataset type: tapvid or BADJA")
    parser.add_argument("--optical-flow-opt", action="store_true", help="Whether optical flow optimization was used")
    args = parser.parse_args()
    eval_single_video(args)
