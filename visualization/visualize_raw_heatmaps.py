# set working directory
import argparse
import os
from tkinter import W
import cv2
import torch
from torch.nn import CosineSimilarity
import yaml
from models.extractor import VitExtractor
from models.networks.tracker_head import TrackerHead
from data.dataset import LongRangeSampler
from optical_flow_opt.optical_flow_methods import predict_point_with_optical_flow, convert_bbox_to_feature_space, compute_correspondence_heatmap_with_search_region
import torchvision.transforms as T
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from pathlib import Path
from tracking_utils import overlay_heatmap_jpg, unravel_index, write_frame_number_on_image, concat_images_w, overlay_point
device = "cuda" if torch.cuda.is_available() else "cpu"
from data.data_utils import save_video_frames, frames_to_video, frames_to_video2, get_grid_query_points 
from data.tapvid import get_query_points_from_benchmark_config
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import gc
import time

imagenet_normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def load_video(video_folder: str, num_frames: int = 120, resize: tuple = None):
    """
    Loads video from folder, resizes frames as desired, and outputs video tensor.

    Args:
        video_folder (str): Folder containing all frames of video, ordered by frame index.
        num_frames (int): Number of frames to load from the video. Defaults to 120.
        resize (tuple): Desired (height, width) dimensions for resizing frames. If None, no resizing is applied.

    Returns:
        return_dict: Dictionary {"video": video tensor of shape: T x 3 x H' x W'}.
    """
    path = Path(video_folder)
    input_files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")))[:num_frames]
    num_frames = len(input_files)
    video = []

    for i, file in enumerate(input_files):
        frame = Image.open(str(file)).convert("RGB")  # Ensure frames are RGB
        if resize:
            frame = frame.resize(resize, Image.Resampling.LANCZOS)  # Resize frame if resize is specified
        video.append(T.ToTensor()(frame))  # Convert frame to tensor

    return_dict = {"video": torch.stack(video)}  # Stack frames into a single tensor
    return return_dict

@torch.no_grad()
def get_frame_features(frame, vit_extractor):
    n_layers = vit_extractor.get_n_layers()
    patch_size = vit_extractor.get_patch_size()

    h, w = (frame.shape[-2] // patch_size) * patch_size, (frame.shape[-1] // patch_size) * patch_size
    frame = T.CenterCrop((h, w))(frame)
    f = vit_extractor.get_feature_from_input(imagenet_normalize(frame), [n_layers - 1])
    # print(n_layers - 1)
    # f = vit_extractor.get_feature_from_input(imagenet_normalize(frame), [11])
    heads, n, d = f.shape

    f = f.permute(1, 0, 2)[1:]
    f = f.reshape(n-1, heads * d).t() # D x N
    n_patch_h = vit_extractor.get_height_patch_num(frame.shape)
    n_patch_w = vit_extractor.get_width_patch_num(frame.shape)
    f = f.reshape(heads * d, n_patch_h, n_patch_w) # D x H x W
    return f


@torch.no_grad()
def compute_correspondence_heatmap(source_feature, target_features, n_patch_h, n_patch_w, normalize_corr=True, normalize_spatially=True):
    """
    Returns:
        colormap: shape (n_patch_h n_patch_w 4), heatmap: shape (n_patch_h n_patch_w).
    """
    device = target_features.device  # Get the device of target_features
    source_feature = source_feature.to(device)  # Move source_feature to the same device

    time_start = time.time()
    if normalize_corr:
        heatmap = CosineSimilarity(dim=1)(source_feature, target_features)
    else:
        # compute dot product between source feature, shape 1 x C and target features N x C
        heatmap = torch.mm(source_feature, target_features.t())
    time_end = time.time()
    
    if normalize_spatially:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # normalize to [0,1]
    else:
        heatmap = torch.clamp(heatmap, 0) # lower bound 0
    heatmap = heatmap.reshape(n_patch_h, n_patch_w)
    
    cmap = cm.get_cmap('jet')
    heatmap_cpu = heatmap.detach().cpu().numpy()
    colormap = (cmap(heatmap_cpu) * 255).astype('uint8')
    return colormap, heatmap, time_end - time_start

def eff_compute_correspondence_heatmap(source_features, target_features, n_patch_h, n_patch_w, normalize_corr=True, normalize_spatially=True):
    """
    Computes correspondence heatmaps for multiple source features at once.
    
    Args:
        source_features: (num_points, C), feature vectors of source points.
        target_features: (H_f*W_f, C), feature vectors of the full target frame.
        n_patch_h: int, height of the feature map.
        n_patch_w: int, width of the feature map.
        normalize_corr: bool, whether to normalize cosine similarity.
        normalize_spatially: bool, whether to normalize heatmaps spatially.

    Returns:
        colormap: shape (num_points, n_patch_h, n_patch_w, 4) - Jet colormap visualization.
        heatmap: shape (num_points, n_patch_h, n_patch_w) - Similarity scores.
    """
    device = target_features.device  # Ensure device consistency
    source_features = source_features.to(device)

    if normalize_corr:
        similarity = CosineSimilarity(dim=2)(source_features[:, None, :], target_features[None, :, :])
    else:
        similarity = torch.mm(source_features, target_features.t())  # (num_points, H_f*W_f)

    if normalize_spatially:
        similarity = (similarity - similarity.min(dim=1, keepdim=True)[0]) / \
                     (similarity.max(dim=1, keepdim=True)[0] - similarity.min(dim=1, keepdim=True)[0] + 1e-6)  # Avoid division by zero
    else:
        similarity = torch.clamp(similarity, 0)

    heatmap = similarity.reshape(-1, n_patch_h, n_patch_w)  # (num_points, H_f, W_f)
    
    # Convert to colormap for visualization
    cmap = cm.get_cmap('jet')
    heatmap_cpu = heatmap.detach().cpu().numpy()
    colormap = (cmap(heatmap_cpu) * 255).astype('uint8')  # (num_points, H_f, W_f, 4)

    return colormap, heatmap

def extract_point_feature(point, frame_features, h, w):
    # samples shape: 1 x 1 x 1 x 2
    samples = point[:2][None, None, None, ...].detach().clone().float()
    #normalize between -1 and 1:
    samples[:, :, :, 0] = samples[:, :, :, 0] / (w - 1)
    samples[:, :, :, 1] = samples[:, :, :, 1] / (h - 1)
    samples[:, :, :, :] = samples[:, :, :, :] * 2 - 1
    #return the corresponding features vector for the normalized point, also does interpolation if needed.
    point_feature = torch.nn.functional.grid_sample(frame_features[None, ...], samples, align_corners=True)[:, :, 0, 0]
    return point_feature

def visualize_heatmaps_for_point(point, start_frame_idx, end_frame_idx, frames_path, vit_extractor = None, video_features=None, normalize_corr=True, normalize_spatially=True, occlusion_threshold=0.3, dynamic_bbox=False):
    frames = load_video(frames_path)["video"]
    _t, c, h, w = frames.shape

    heatmaps = []
    heatmap_imgs = []
    trajectory_imgs = []
    trajectory = []
    occlusion = []  # To store occlusion status for each point
    time_taken = 0
    point_frame_idx = int(point[2])
    point_frame = frames[point_frame_idx].cuda()

    # create an img with the query point on it, and write the frame num:
    point_img = overlay_point(T.ToPILImage()(point_frame.cpu()), point[0].cpu(), point[1].cpu())
    point_img = write_frame_number_on_image(point_img, point_frame_idx)

    # takes the frame features from frame(t=point[2]):
    #point_frame_features = get_frame_features(point_frame[None, ...], vit_extractor).detach().clone()
    point_frame_features = video_features[point_frame_idx]
    _, h_f, w_f = point_frame_features.shape
    point_frame_features = point_frame_features.to(device) #this dor add 
    point_feature = extract_point_feature(point, point_frame_features, h, w)
    last_point = (point[0], point[1])

    for t in range(start_frame_idx, end_frame_idx+1):
        frame = frames[t].cuda()
        of_success = False # we reset the optical flow sucees flag to false first.
        if args.optical_flow_opt and t > start_frame_idx:
            prev_frame = frames[t-1].cuda()
            _, bbox_pixels, of_success = predict_point_with_optical_flow(
                point=last_point,
                prev_frame=prev_frame,
                next_frame=frame,
                search_radius=50,
                dynamic_bbox=dynamic_bbox
            )

            if of_success:
                bbox_features = convert_bbox_to_feature_space(
                    bbox_pixels, (h, w), (h_f, w_f)
                )
                heatmap_img, heatmap, nearest_coord, time_taken = compute_correspondence_heatmap_with_search_region(
                    point_feature, video_features[t], bbox_features, h_f, w_f, normalize_corr=normalize_corr, normalize_spatially=normalize_spatially)

        if not of_success: # optical flow failed, use full frame search
            print(f"optical flow failed at frame {t}, using full frame search")
            frame_features = get_frame_features(frame[None, ...], vit_extractor) if video_features is None else video_features[t]
            c, h_f, w_f = frame_features.shape
            frame_features = frame_features.reshape(c, h_f * w_f).permute(1, 0) #each row is a feature vector for a pixel in the frame
            heatmap_img, heatmap, time_taken = compute_correspondence_heatmap(point_feature, frame_features, h_f, w_f, normalize_corr=normalize_corr, normalize_spatially=normalize_spatially)
            nearest_coord = unravel_index(heatmap.argmax(), heatmap.shape)  # h_f, w_f (y,x)

        # Get the heatmap value at the nearest_coord -saved for later, before normalization
        heatmap_value = heatmap[nearest_coord]

        # Scale coordinates from feature map size to full image size
        last_point = (int(nearest_coord[1] * h / h_f),int(nearest_coord[0] * w / w_f),)
        trajectory.append(last_point)
        trajectory_img = overlay_point(frame, last_point[0], last_point[1])
        heatmap_img = overlay_heatmap_jpg(T.ToPILImage()(frame.cpu()), heatmap_img)
        heatmap_img = write_frame_number_on_image(heatmap_img, t)
        heatmap_img = overlay_point(heatmap_img, x = last_point[0], y = last_point[1]) # this will add the max corespond point to the heatmap image 
        if t == point_frame_idx:
            heatmap_img = overlay_point(heatmap_img, point[0].cpu(), point[1].cpu(), r=3, c="green")
        heatmaps.append(heatmap)
        heatmap_imgs.append(heatmap_img)
        if t == point_frame_idx:
            trajectory_imgs.append(point_img)
        else:
            trajectory_imgs.append(trajectory_img)
        

        # If the heatmap value is below the occlusion threshold, set as occlusion
        if heatmap_value < occlusion_threshold:
            print(f"occlusion: True for t={t}")
            occlusion.append(True)  # Mark this point as occluded
            last_point = None  # Mark the coordinate as None (or use a placeholder value for occlusion)
        else:
            occlusion.append(False)  # Not occluded
        
        #trajectory = np.array(trajectory)  # Shape will be (num_frames, 2)
    
    return heatmap_imgs, heatmaps, trajectory_imgs, point_feature, point_img , trajectory , occlusion, time_taken
    # return heatmap_imgs, heatmaps, trajectory_imgs, point_feature, point_frame_features, point_frame

def return_heatmap_imgs(heatmap_imgs, trajectory_imgs):
    """
    Args:
        heatmap_imgs: list of PIL images
        trajectory_imgs: list of PIL images
    """
    # compute cosine 
    return concat_images_w(heatmap_imgs), concat_images_w(trajectory_imgs)

def heatmap_single_point(args):
    

    #vit_extractor = VitExtractor(args.dino_model, args.stride, device)
    my_features = torch.load(args.dino_embed_video_path)
    frames = load_video(args.FRAMES_PATH)["video"].cuda()
    # Automatically determine end_frame_idx
    args.end_frame_idx = frames.shape[0] - 1  # Set to the last frame index
    print(f"Automatically determined end_frame_idx: {args.end_frame_idx}")
    print(my_features.shape)
    # specfic interest point
    x, y, t = args.point
    point = torch.tensor([x, y, t]).cuda()
    print(f"point in heatmap_single_point: {point}")
    heatmap_imgs, heatmaps, trajectory_imgs, point_features, point_img ,_,_,time_taken = visualize_heatmaps_for_point(
        point, args.start_frame_idx, args.end_frame_idx, args.FRAMES_PATH,
        video_features = my_features,
        normalize_spatially=False,
        dynamic_bbox=args.dynamic_bbox
    )
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    # i add trajectory_imgs
    frames_path = save_video_frames(heatmap_imgs, output_path) 
    frames_to_video(frames_path, output_filename="heatmap", fps=5)
    frames_path = save_video_frames(trajectory_imgs, output_path)
    frames_to_video(frames_path, output_filename="trajectory", fps=5)
    # save point_img
    #point_img.save(os.path.join(output_path, "query_point.jpg"))
    print(f"time taken: {time_taken}")

def heatmap_grid_points(args):

    print("dino model: ", args.dino_model)
    vit_extractor = VitExtractor(args.dino_model, args.stride, device)
    my_features = torch.load(args.dino_embed_video_path)
    frames = load_video(args.FRAMES_PATH)["video"].cuda()
    # Automatically determine end_frame_idx
    args.end_frame_idx = frames.shape[0] - 1  # Set to the last frame index
    print(f"Automatically determined end_frame_idx: {args.end_frame_idx}")
    #video_features = [None for _ in range(frames.shape[0])]
    #for t in range(config['start_frame_idx'], config['end_frame_idx']+1):
     #   frame = frames[t].cuda()
      #  video_features[t] = get_frame_features(frame[None, ...], vit_extractor) 
    if args.seg_mask_path is not None:
        segm_mask = cv2.imread(str(args.seg_mask_path), cv2.IMREAD_GRAYSCALE)
    
    if segm_mask is None:
        raise ValueError(f"Failed to load segmentation mask from {args.seg_mask_path}. Check if the file exists and is readable.(you should load here only the first frame mask!)")
    
    segm_mask = torch.from_numpy(segm_mask).cuda().float().unsqueeze(0).unsqueeze(0) # 1 x 1 x H x W

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    query_points = get_grid_query_points((frames.shape[-2], frames.shape[-1]), segm_mask=segm_mask, device=frames.device , interval = 50)

    for query_point_idx, query_point in enumerate(query_points):
        heatmap_imgs, heatmaps, trajectory_imgs, point_features, point_img ,_,_,_ = visualize_heatmaps_for_point(
            query_point, args.start_frame_idx, args.end_frame_idx, args.FRAMES_PATH, vit_extractor,
            video_features = my_features, 
            normalize_spatially=True,
            dynamic_bbox=args.dynamic_bbox
            )
        frames_path = save_video_frames(heatmap_imgs, output_path)
        frames_to_video(frames_path, output_filename=f"heatmap_query_point_{query_point_idx}", fps=5)
        # frames_path = save_video_frames(trajectory_imgs, output_path)
        # frames_to_video(frames_path, output_filename=f"trajectory_query_point_{query_point_idx}", fps=5)
        # save point_img
        point_img.save(os.path.join(output_path, f"query_point_{query_point_idx}.jpg"))

def eff_unravel_index(indices, shape):
    """
    Converts flat indices into unraveled coordinates in a target shape.
    
    Args:
        indices: A tensor of flat indices.
        shape: The target shape as a tuple of integers.
    
    Returns:
        A tuple of tensors, each with the same shape as indices.
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(coord))

def eff2_heatmap_benchmark_grid_points(args):
    print("loading video features")
    my_features = torch.load(args.dino_embed_video_path)
    #frames = load_video(args.FRAMES_PATH)["video"].cuda()
    frames = load_video(video_folder = args.FRAMES_PATH , resize = (854, 476))["video"]

    args.end_frame_idx = frames.shape[0] - 1  # Set to the last frame index
    print(f"Automatically determined end_frame_idx: {args.end_frame_idx}")

    if args.seg_mask_path is not None:
        segm_mask = cv2.imread(str(args.seg_mask_path), cv2.IMREAD_GRAYSCALE)

    if segm_mask is None:
        raise ValueError(f"Failed to load segmentation mask from {args.seg_mask_path}. Check if the file exists and is readable. (you should load here only the first frame mask!)")

    segm_mask = torch.from_numpy(segm_mask).cuda().float().unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W

    os.makedirs(args.output_path, exist_ok=True)

    # Load all benchmark query points at once
    benchmark_query_points_dictionary = get_query_points_from_benchmark_config(
        args.benchmark_pickle_path, args.video_id, rescale_sizes=[frames.shape[-1], frames.shape[-2]]
    )

    print(f"Query Points Dictionary Keys: {benchmark_query_points_dictionary.keys()}")

    # Aggregate all points across frames
    all_points = []
    all_frame_idxs = []
    for frame_idx, query_points_list in benchmark_query_points_dictionary.items():
        all_points.append(torch.tensor(query_points_list))  # Collect query points
        all_frame_idxs.extend([frame_idx] * len(query_points_list))  # Keep track of frame indices

    all_points = torch.cat(all_points).cuda()  # (Total_points, 3)
    all_frame_idxs = torch.tensor(all_frame_idxs).cuda()  # (Total_points,)

    print(f"Total Query Points: {all_points.shape}")

    # Compute trajectory and occlusion for all points in one batch
    trajectory_query, occlusion_query = eff2_get_trajectory_and_occlusion(
        all_points, args.start_frame_idx, args.end_frame_idx, args.FRAMES_PATH, video_features=my_features, normalize_spatially=True
    )

    print(f"Computed Trajectories Shape: {trajectory_query.shape}")
    print(f"Computed Occlusion Shape: {occlusion_query.shape}")

    # Save trajectories and occlusions per frame
    os.makedirs(args.trajectory_output_path, exist_ok=True)
    os.makedirs(args.occlusion_output_path, exist_ok=True)

    # Assuming all_frame_idxs, trajectory_query, and occlusion_query are already defined
    trajectory_dict = {}  # Dictionary to store all trajectories per frame
    occlusion_dict = {}   # Dictionary to store all occlusions per frame

    # Iterate over each frame index
    for i, frame_idx in enumerate(all_frame_idxs.cpu().numpy()):
        # Append data to the trajectory and occlusion lists for the current frame index
        if frame_idx not in trajectory_dict:
            trajectory_dict[frame_idx] = []
        trajectory_dict[frame_idx].append(trajectory_query[i])  # Appending (50, 2) array

        if frame_idx not in occlusion_dict:
            occlusion_dict[frame_idx] = []
        occlusion_dict[frame_idx].append(occlusion_query[i])  # Assuming occlusion_query[i] is of appropriate shape

    # Now save all trajectories and occlusions at once for each frame
    for frame_idx in trajectory_dict:
        # Create filenames for each frame index
        trajectory_filename = os.path.join(args.trajectory_output_path, f"trajectories_{frame_idx}.npy")
        occlusion_filename = os.path.join(args.occlusion_output_path, f"occlusion_preds_{frame_idx}.npy")

        # Convert list of trajectories and occlusions for the frame index into numpy arrays
        # Convert the list of (50, 2) arrays into a (23, 50, 2) shape
        all_trajectories = np.stack(trajectory_dict[frame_idx], axis=0)  # This will stack them into shape (23, 50, 2)
        all_occlusions = np.stack(occlusion_dict[frame_idx], axis=0)  # Assuming occlusion data needs to be concatenated

        # Save the concatenated data
        np.save(trajectory_filename, all_trajectories)
        np.save(occlusion_filename, all_occlusions)

def get_subpixel_max_coords(heatmaps, h_f, w_f):
    """
    Get sub-pixel coordinates of maxima in heatmaps using quadratic interpolation.
    
    Args:
        heatmaps: Tensor of shape (num_points, h_f, w_f)
        h_f: height of the heatmap
        w_f: width of the heatmap
        
    Returns:
        nearest_coords: Tensor of shape (num_points, 2) with [y, x] subpixel coordinates
        heatmap_values: Tensor of shape (num_points,) with peak values from each heatmap
    """
    num_points = heatmaps.shape[0]

    # Flatten to get coarse max indices
    heatmap_values, max_indices = torch.max(heatmaps.view(num_points, -1), dim=1)
    coarse_y = max_indices // w_f
    coarse_x = max_indices % w_f

    # Pad heatmaps to handle edge max locations
    padded = torch.nn.functional.pad(heatmaps, (1, 1, 1, 1), mode='replicate')

    # Offsets for indexing into padded heatmaps
    px = coarse_x + 1
    py = coarse_y + 1

    center = padded[torch.arange(num_points), py, px]
    left   = padded[torch.arange(num_points), py, px - 1]
    right  = padded[torch.arange(num_points), py, px + 1]
    top    = padded[torch.arange(num_points), py - 1, px]
    bottom = padded[torch.arange(num_points), py + 1, px]

    # Compute x and y subpixel offsets
    eps = 1e-6  # for numerical stability
    denom_x = left - 2 * center + right + eps
    denom_y = top - 2 * center + bottom + eps
    x_offset = 0.5 * (left - right) / denom_x
    y_offset = 0.5 * (top - bottom) / denom_y

    # Clamp offsets to [-1, 1] to avoid wild values
    x_offset = torch.clamp(x_offset, -1.0, 1.0)
    y_offset = torch.clamp(y_offset, -1.0, 1.0)

    subpixel_x = coarse_x.float() + x_offset
    subpixel_y = coarse_y.float() + y_offset

    nearest_coords = torch.stack([subpixel_y, subpixel_x], dim=1)  # (num_points, 2)

    return nearest_coords, heatmap_values

def eff2_get_trajectory_and_occlusion(points, start_frame_idx, end_frame_idx, frames_path, vit_extractor=None, video_features=None, normalize_corr=True, normalize_spatially=True, occlusion_threshold=0.3):
    frames = load_video(video_folder = frames_path , resize = (854, 476))["video"]
    _t, c, h, w = frames.shape
    num_points = points.shape[0]

    print("Total points:", points.shape[0])
    point_frame_idxs = points[:, 2].long().cuda()  # Get frame indices for each point
    #print("Point Frame Indices:", point_frame_idxs)

    # Extract unique frame indices to optimize memory usage
    unique_frame_idxs, inverse_indices = torch.unique(point_frame_idxs, return_inverse=True)  # (num_unique_frames,)
    unique_frame_idxs = unique_frame_idxs.cpu()

    # Use video features only for unique frames (optimized memory usage)
    unique_frame_features = video_features[unique_frame_idxs].cuda()  # (num_unique_frames, C, H_f, W_f)
    print("unique_frame_features shape:", unique_frame_features.shape)

    # Normalize the point coordinates for grid sampling
    samples = points[:, :2].float()
    samples[:, 0] = samples[:, 0] / (w - 1) * 2 - 1  # Normalize X
    samples[:, 1] = samples[:, 1] / (h - 1) * 2 - 1  # Normalize Y
    samples = samples.view(-1, 1, 1, 2)  # (num_points, 1, 1, 2)

    # Initialize an empty list to collect the point features
    all_point_features = []

    # Process each sample individually
    for i in range(points.shape[0]):
        # Get the original frame index for the current point
        original_frame_idx = point_frame_idxs[i].item()

        # Find the corresponding unique frame index using the inverse mapping
        unique_frame_idx = inverse_indices[i].item()  # Get the mapped index for the current point

        # Extract the corresponding frame features for the current sample
        current_frame_features = unique_frame_features[unique_frame_idx].unsqueeze(0)  # Shape (1, C, H_f, W_f)

        # Prepare the grid sample
        sample = samples[i].unsqueeze(0)  # Shape (1, 1, 1, 2) for grid_sample
        
        # Use grid_sample to sample from the current frame features for the point
        point_frame_features = F.grid_sample(current_frame_features, sample, align_corners=True)  # Shape (1, C, 1, 1)
        
        # Flatten the point features and append to the list
        point_features = point_frame_features.view(-1)
        all_point_features.append(point_features)

    # Combine all point features into a tensor
    all_point_features = torch.stack(all_point_features)

    print("Extracted point features shape:", all_point_features.shape)
    trajectory, occlusion = [], []
    
    for t in tqdm(range(start_frame_idx, end_frame_idx + 1), desc="Processing frames"):
        frame = frames[t].cuda()
        frame_features = get_frame_features(frame[None, ...], vit_extractor) if video_features is None else video_features[t]

        c, h_f, w_f = frame_features.shape
        frame_features = frame_features.reshape(c, h_f * w_f).permute(1, 0)  # (H_f*W_f, C)

        #source_features = point_features: (num_points, C), feature vectors of source points.
        #target_features = frame_features: (H_f*W_f, C), feature vectors of the full target frame.
        # Compute heatmaps for all points at once
        _, heatmaps = eff_compute_correspondence_heatmap(all_point_features, frame_features, h_f, w_f, normalize_corr=normalize_corr, normalize_spatially=normalize_spatially)
    
        # # Find the max locations for all points at once
        #if want to use argmax 
        """
        heatmap_values, max_indices = torch.max(heatmaps.view(num_points, -1), dim=1) # max_indices is the index of the max value in the heatmap    
        #print("max_indices", max_indices)
        #print("max_indices shape", max_indices.shape)
        nearest_coords = eff_unravel_index(max_indices, (h_f, w_f))
        nearest_coords = torch.stack(nearest_coords, dim=1)
        #print("nearest_coords", nearest_coords)
        """

        # Get subpixel coordinates of maxima in heatmaps using quadratic interpolation
        nearest_coords, heatmap_values = get_subpixel_max_coords(heatmaps, h_f, w_f)  # Get subpixel coordinates
        # Check occlusion based on heatmap threshold
        is_occluded = heatmap_values < occlusion_threshold
        occlusion.append(is_occluded.cpu().numpy())

        # Convert nearest_coords from feature space to image space
        temp = nearest_coords.clone()  # Clone to avoid modifying the original tensor
        nearest_coords[:, 0] = (nearest_coords[:, 1] * w / w_f).int()  # Scale Y
        nearest_coords[:, 1] = (temp[:, 0] * h / h_f).int()  # Scale X
        #print("nearest_coords", nearest_coords)
        trajectory.append(nearest_coords.cpu().numpy())

    # Transpose trajectory to match the shape of ground truth trajectories
    trajectory = np.array(trajectory).transpose(1, 0, 2)  # Shape to: (num_points, num_frames, 2)
    occlusion = np.array(occlusion).transpose(1, 0)  # Shape to: (num_points, num_frames)

    return np.array(trajectory), np.array(occlusion)  # Shape: (num_points, num_frames, 2) and (num_points, num_frames)

def benchmark_grid_points(args):
    """Process benchmark query points for a video and generate heatmap visualizations."""
    
    # Load video features
    print("Loading video features...")
    my_features = torch.load(args.dino_embed_video_path)
    
    # Load video frames
    frames = load_video(video_folder=args.FRAMES_PATH, resize=(854, 476))["video"]
    
    # Automatically determine end frame
    args.end_frame_idx = frames.shape[0] - 1
    print(f"Video has {frames.shape[0]} frames (0 to {args.end_frame_idx})")
    
    # Create the benchmark query points dictionary
    benchmark_query_points_dictionary = get_query_points_from_benchmark_config(
        args.benchmark_pickle_path,
        args.video_id,
        rescale_sizes=[frames.shape[-1], frames.shape[-2]]  # [width, height]
    )
    
    # Keep only query points from frame 0
    if 0 in benchmark_query_points_dictionary:
        benchmark_query_points_dictionary = {0: benchmark_query_points_dictionary[0]}
    else:
        print("Warning: No query points found for frame 0")
        benchmark_query_points_dictionary = {}
    
    print(f"Query Points Dictionary Keys (frame indices): {benchmark_query_points_dictionary.keys()}")
    
    # Print summary of query points per frame
    for frame_idx, query_points_list in benchmark_query_points_dictionary.items():
        print(f"Frame {frame_idx}: {len(query_points_list)} query points")
    
    # Initialize dictionaries to store trajectories, occlusions, and time taken
    trajectory_dict = {}  # key: point index, value: trajectory list
    occlusion_dict = {}   # key: point index, value: occlusion list
    time_taken_dict = {}  # key: point index, value: time taken
    
    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.trajectory_output_path, exist_ok=True)
    os.makedirs(args.occlusion_output_path, exist_ok=True)
    os.makedirs(args.time_taken_output_path, exist_ok=True)
    
    # Iterate over all query points and compute trajectory/occlusion
    point_idx = 0
    for frame_idx, query_points_list in benchmark_query_points_dictionary.items():
        for query_point in tqdm(query_points_list, desc=f"Processing points from frame {frame_idx}"):
            # Convert query point to tensor
            query_point_tensor = torch.tensor(query_point).cuda()
            
            # Get trajectory and occlusion using visualize_heatmaps_for_point
            _, _, _, _, _, trajectory, occlusion, time_taken = visualize_heatmaps_for_point(
                point=query_point_tensor,
                start_frame_idx=args.start_frame_idx,
                end_frame_idx=args.end_frame_idx,
                frames_path=args.FRAMES_PATH,
                video_features=my_features,
                normalize_spatially=True,
                dynamic_bbox=args.dynamic_bbox
            )
            
            # Store trajectory, occlusion, and time taken in dictionaries
            trajectory_dict[point_idx] = trajectory
            occlusion_dict[point_idx] = occlusion
            time_taken_dict[point_idx] = time_taken
            
            point_idx += 1
    
    print(f"Processed {point_idx} query points")
    
    # Save all trajectories and occlusions
    # Convert dictionaries to arrays for saving
    all_trajectories = np.array([trajectory_dict[i] for i in range(len(trajectory_dict))])  # Shape: (num_points, num_frames, 2)
    all_occlusions = np.array([occlusion_dict[i] for i in range(len(occlusion_dict))])      # Shape: (num_points, num_frames)
    
    # Save trajectories
    if args.optical_flow_opt and args.dynamic_bbox:
        of_flag = 11
    elif args.optical_flow_opt:
        of_flag = 1
    else:
        of_flag = 0
    trajectory_filename = os.path.join(args.trajectory_output_path, f"trajectories_0_of{of_flag}.npy")
    np.save(trajectory_filename, all_trajectories)
    print(f"Saved trajectories to {trajectory_filename}, shape: {all_trajectories.shape}")
    
    # Save occlusions
    occlusion_filename = os.path.join(args.occlusion_output_path, f"occlusion_preds_0_of{of_flag}.npy")
    np.save(occlusion_filename, all_occlusions)
    print(f"Saved occlusions to {occlusion_filename}, shape: {all_occlusions.shape}")
    
    # Save time taken
    all_time_taken = np.array([time_taken_dict[i] for i in range(len(time_taken_dict))])  # Shape: (num_points,)
    time_taken_filename = os.path.join(args.time_taken_output_path, f"time_taken_0_of{of_flag}.npy")
    np.save(time_taken_filename, all_time_taken)
    print(f"Saved time taken to {time_taken_filename}, shape: {all_time_taken.shape}")


if __name__ == "__main__":
    # Set all parameters using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--start_frame_idx", type=int, default=0)
    parser.add_argument("--benchmark-pickle-path", default="./tapvid/tapvid_davis_data_strided.pkl", type=str)
    parser.add_argument("--dataset_root", type=str, default="dataset/tapvid-davis/davis_480", help="Root directory containing multiple video folders")
    parser.add_argument("--output_root", type=str, default="output_folder/davis_480/", help="Root directory for output files")
    #parser.add_argument("--point", type=str, default="benchmark_grid")
    parser.add_argument("--point", type=lambda s: s if s in ['grid', 'benchmark_grid'] else [int(v) for v in s.split(',')], 
                        default=[495, 150, 0]) # 290,313,0 for the car video, 495,150,0 for the first video
    parser.add_argument("--save_video", action="store_true", help="Save video")
    parser.add_argument("--optical_flow_opt", action="store_true", help="Use optical flow optimization")
    parser.add_argument("--video_id", type=int, default=0) # -1 for all videos, 0 for the first video, 1 for the second video, etc.
    parser.add_argument("--dynamic_bbox", action="store_true", help="Use dynamic bbox")
    args = parser.parse_args()

    # Iterate over all folders in the dataset root
    for folder_name in sorted(os.listdir(args.dataset_root)):
        folder_path = os.path.join(args.dataset_root, folder_name)

        if not os.path.isdir(folder_path):
            continue  # Skip files, only process directories

        video_id = folder_name  # Assuming the folder name corresponds to video_id
        if args.video_id != -1 and int(video_id) != args.video_id:
            continue

        output_video_folder = os.path.join(args.output_root, video_id)  # Ensure consistent output structure

        print(f"Processing folder: {folder_path} , with video_id: {video_id}")

        # Set folder-specific paths dynamically
        args.FRAMES_PATH = os.path.join(folder_path, "video")
        args.seg_mask_path = os.path.join(folder_path, "masks/00000.png")
        # this is NOT a correct name - @maor.madai - fix it later
        args.dino_embed_video_path = os.path.join(folder_path, "dino_giant_embeddings_l38/dino_embed_video.pt") # for dinov2 
        args.output_path = os.path.join(output_video_folder, "dinov2")
        args.trajectory_output_path = os.path.join(output_video_folder, "dinov2_grid_trajectory")
        args.occlusion_output_path = os.path.join(output_video_folder, "dinov2_grid_occlusion")
        args.time_taken_output_path = os.path.join(output_video_folder, "dinov2_grid_time_taken")
        # args.dino_embed_video_path = os.path.join(folder_path, "dinov3_small_embeddings_l11/dino_embed_video.pt") # for dinov3
        # args.output_path = os.path.join(output_video_folder, "dinov3")
        # args.trajectory_output_path = os.path.join(output_video_folder, "dinov3_grid_trajectory")
        # args.occlusion_output_path = os.path.join(output_video_folder, "dinov3_grid_occlusion")
        # args.video_id = int(video_id)

        # Create output directories if they don’t exist
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(args.trajectory_output_path, exist_ok=True)
        os.makedirs(args.occlusion_output_path, exist_ok=True)
        os.makedirs(args.time_taken_output_path, exist_ok=True)

        # Call the appropriate function based on the point type
        if args.point == 'grid':
            heatmap_grid_points(args)
        elif args.point == 'benchmark_grid':
            benchmark_grid_points(args)
        else:
            heatmap_single_point(args)

        print(f"Completed processing for: {folder_path}")
        # Clear memory
        if 'frames' in locals():
            del frames
        if 'my_features' in locals():
            del my_features
        torch.cuda.empty_cache()
        gc.collect()
