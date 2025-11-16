import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the .npy file
grid_trajectories_dino = np.load('/home/dor.danino/dino-tracker/dataset/tapvid-davis/davis_480/10/trajectories/trajectories_80.npy')
grid_occlusions_dino = np.load('/home/dor.danino/dino-tracker/dataset/tapvid-davis/davis_480/10/occlusions/occlusion_preds_80.npy')
grid_trajectories_my = np.load('/home/dor.danino/dino-tracker/output_folder/davis_480/10/grid_trajectory/trajectories_80.npy')
grid_occlusions_my = np.load('/home/dor.danino/dino-tracker/output_folder/davis_480/10/grid_occlusion/occlusion_preds_80.npy')
embed1= torch.load('dataset/tapvid-davis/davis_480/6/dino_small_embeddings/dino_embed_video.pt', map_location=torch.device('cpu'))
embed2= torch.load('dataset/tapvid-davis/davis_480/6/dino_embeddings/dino_embed_video.pt', map_location=torch.device('cpu'))
embed3= torch.load('output_folder/davis_480/6/dino_featup.pt', map_location=torch.device('cpu'))
embed4= torch.load('output_folder/davis_480/6/dinov2_Fit3D_embedded.pt', map_location=torch.device('cpu'))

# Print the shape of the .npy file
print(embed1.shape)
print(embed2.shape)
print(embed3.shape)
print(embed4.shape)

# print(grid_trajectories_my.shape)
# print(f"Keys in featupembed: {featupembed.keys()}")
# print(f"Keys in featupembed: {embed.keys()}")

from PIL import Image

# Base path for the frames
base_path = '/home/dor.danino/dino-tracker/dataset/tapvid-davis/davis_480'

# # Iterate through the folders (0, 1, 2, ...)
# for i in range(30):  # Adjust the range as needed
#     frame0_path = f'{base_path}/{i}/video/00000.jpg'
#     try:
#         # Load the image
#         frame0 = Image.open(frame0_path)

#         # Print the shape (size in PIL)
#         print(f"video {i} shape (width, height): {frame0.size}")
#     except FileNotFoundError:
#         print(f"Frame {i} not found at path: {frame0_path}")