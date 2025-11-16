import argparse
from tqdm import tqdm
import pandas as pd
import os
import pickle
from eval.metrics import compute_tapvid_metrics_for_video, compute_badja_metrics_for_video


def eval_dataset(args):
    benchmark_data = pickle.load(open(args.benchmark_pickle_path, "rb"))

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
        video_dir = os.path.join(dataset_root, video_idx_str)
        trajectories_dir = os.path.join(video_dir, "dinov2_Fit3D_argmax_grid_trajectory")
        occlusions_dir = os.path.join(video_dir, "dinov2_Fit3D_argmax_grid_occlusion")
        video_idx = int(video_idx_str)

        # Get the video size for the current video index
        #pred_video_size = video_sizes.get(video_idx, [854, 476])  # Default to [854, 480] if not found

        if args.dataset_type == "tapvid":
            metrics = compute_tapvid_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                        model_occ_pred_dir=occlusions_dir,
                                                        video_idx=video_idx,
                                                        benchmark_data=benchmark_data,
                                                        pred_video_sizes=[854, 476]) # set to None to get the video size from the benchmark data
        elif args.dataset_type == "BADJA":
            metrics = compute_badja_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                      video_idx=video_idx,
                                                      benchmark_data=benchmark_data,
                                                      pred_video_sizes=[854, 480])
        else:
            raise ValueError("Invalid dataset type. Must be either tapvid or BADJA")
        metrics["video_idx"] = int(video_idx)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('video_idx', inplace=True)
    metrics_df.loc['average', :] = metrics_df.mean()
    metrics_df.to_csv(args.out_file)
    print("Total metrics:") 
    print(metrics_df.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root-dir", default="output_folder/davis_480", type=str)
    parser.add_argument("--benchmark-pickle-path", default="tapvid/tapvid_davis_data_strided.pkl", type=str)
    parser.add_argument("--out-file", default="output_folder/dinov2_Fit3D_argmax_comp_metrics.csv", type=str)
    parser.add_argument("--dataset-type", default="tapvid", type=str, help="Dataset type: tapvid or BADJA")
    args = parser.parse_args()
    eval_dataset(args)
