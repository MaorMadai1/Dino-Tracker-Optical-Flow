import os
import pickle
import numpy as np 
# os.chdir("/home/projects/talide/narekt/projects/taming-dino2/")

import torch
from tqdm import tqdm
import yaml
from data.data_utils import frames_to_video
from data.tapvid import get_query_points_from_davis_config, get_video_config_by_video_id
from generate_supervision_mask import save_video_frames
from models.model import generate_heatmaps_for_query_point_tapnet
from train import OmniVideoTracker
import argparse
from models.utils import load_pre_trained_model, get_last_ckpt


device = "cuda:0"


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@torch.no_grad()
def run(args):
    davis_id = args.video_id
    training_cfg_path = args.train_cfg_path

    tracker_args = Namespace()
    tracker_args.config = training_cfg_path
    tracker_args.video_id = davis_id
    tracker_args.wandb_off = True
    tracker_args.debug_mode = False
    if args.remove_dino_bb_from_path:
        tracker_args.remove_dino_bb_from_path = args.remove_dino_bb_from_path

    omnivideo_tracker = OmniVideoTracker(tracker_args)
    model, _, _sch = omnivideo_tracker.get_model()
    if not args.use_raw_features:
        iter = get_last_ckpt(omnivideo_tracker.config["MODELS_FOLDER"]) if args.iter is None else args.iter
        model.refiner_head = load_pre_trained_model(
            torch.load(os.path.join(omnivideo_tracker.config["MODELS_FOLDER"], f"refiner_head_final_{iter}.pt")),
            model.refiner_head
        )
        model.tapnet_cnn = load_pre_trained_model(
            torch.load(os.path.join(omnivideo_tracker.config["MODELS_FOLDER"], f"tapnet_cnn_final_{iter}.pt")),
            model.tapnet_cnn
        )
    
    model.eval()
    if args.is_first:
        omnivideo_tracker.config["DAVIS_QUERY_POINTS_CONFIG_FILE"] = omnivideo_tracker.config["DAVIS_QUERY_POINTS_CONFIG_FILE"].replace("strided", "first")
    
    omnivideo_tracker.save_model_heatmaps(model=model, output_heatmap_path=args.out_heatmap_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cfg-path", type=str)
    parser.add_argument("--video-id", type=int, default=0)
    parser.add_argument("--out-heatmap-path", type=str, default=None)
    parser.add_argument("--use-raw-features", action="store_true", default=False)
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--is-first", action="store_true", default=False)
    parser.add_argument("--out-traj-postfix", type=str, default=None)

    parser.add_argument("--remove_dino_bb_from_path", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    run(args)
