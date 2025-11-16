import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_lightning import seed_everything
from data.data_utils import frames_to_video2


class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


@torch.no_grad()
def plot_feats(
    frames_path: str, 
    lr_embeds_path: str, 
    hr_embeds_path: str, 
    lr_video_path: str, 
    hr_video_path: str,
    output_video_path: str
):
    """
    Visualizes PCA-transformed embeddings alongside video frames and heatmaps, then saves as MP4.
    """
    # Load embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_embeds = torch.load(lr_embeds_path, map_location=device)
    hr_embeds = torch.load(hr_embeds_path, map_location=device)
    
    assert len(lr_embeds.shape) == 4 and len(hr_embeds.shape) == 4  # (T, C, H, W)
    
    # Load frames
    frame_files = sorted([f for f in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, f))])
    frames = [T.ToTensor()(Image.open(os.path.join(frames_path, f))) for f in frame_files]
    
    num_frames = min(len(frames), lr_embeds.shape[0], hr_embeds.shape[0])
    
    # Load heatmap videos using OpenCV
    lr_video = cv2.VideoCapture(lr_video_path)
    hr_video = cv2.VideoCapture(hr_video_path)
    
    # Prepare for saving the output video
    output_frames = []
    
    for i in tqdm(range(num_frames), desc="Processing frames"):
        frame = frames[i].permute(1, 2, 0).detach().cpu().numpy()

        # Apply PCA separately for different channel sizes using the TorchPCA class
        assert len(frame.shape) == len(lr_embeds[i].shape) == len(hr_embeds[i].shape) == 3
        seed_everything(0)
        [lr_feats_pca, hr_feats_pca], _ = pca([lr_embeds[i].unsqueeze(0), hr_embeds[i].unsqueeze(0)])

        lr_pca = lr_feats_pca[0].permute(1, 2, 0).detach().cpu().numpy()
        hr_pca = hr_feats_pca[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Get corresponding frames from heatmap videos
        ret_lr, lr_heatmap_frame = lr_video.read()
        ret_hr, hr_heatmap_frame = hr_video.read()
        
        if not (ret_lr and ret_hr):
            break  # Stop if video ends
        
        lr_heatmap_frame = cv2.cvtColor(lr_heatmap_frame, cv2.COLOR_BGR2RGB)
        hr_heatmap_frame = cv2.cvtColor(hr_heatmap_frame, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(3, 2, figsize=(40, 35))

        # First row: Original Frame (centered title)
        axes[0, 0].imshow(frame)
        axes[0, 0].set_title("Original Frame", fontsize=40, loc='center')  # Set fontsize and center the title

        # Second row:lr_heatmap_frame, DinoV2 FeatUp Heatmap
        axes[1, 0].imshow(lr_heatmap_frame)
        axes[1, 0].set_title("DinoV2 Heatmap", fontsize=40)
        axes[1, 1].imshow(hr_heatmap_frame)
        axes[1, 1].set_title("DinoV2 FeatUp Heatmap", fontsize=40)

        # Third row: DinoV2 Features PCA, DinoV2 FeatUp Features PCA
        axes[2, 0].imshow(lr_pca)
        axes[2, 0].set_title("DinoV2 Features PCA", fontsize=40)
        axes[2, 1].imshow(hr_pca)
        axes[2, 1].set_title("DinoV2 FeatUp Features PCA", fontsize=40)

        for ax in axes.flatten():
            ax.axis("off")
        
        plt.tight_layout()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        output_frames.append(img_array)
        plt.close(fig)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    frames_to_video2(frames = output_frames,output_filename = "readyvideo" , fps=5, output_path = output_video_path)

print("Visualizing embeddings and heatmaps...")
plot_feats(
    frames_path = "dataset/horsejump/video", 
    lr_embeds_path = "dataset/horsejump/dino_embeddings/dino_embed_video.pt", 
    hr_embeds_path ="loftup/output/all_hr_feats.pt", 
    lr_video_path = "/home/dor.danino/dino-tracker/output_folder/horse/dinov2_small_grid/heatmap_query_point_0_fps_5.mp4", 
    hr_video_path ="/home/dor.danino/dino-tracker/output_folder/horse/dinov2_small_feat_up_grid/heatmap_query_point_0_fps_5.mp4",
    output_video_path ="loftup/output"
)