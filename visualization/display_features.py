import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the DINO embeddings (video) from the .pt file, mapping to CPU
# dino_embed_video = torch.load(
#     'output_folder/super_res_features_frames.pt', 
#     map_location=torch.device('cpu')
# )
dino_embed_video = torch.load(
    'dataset/horsejump/dino_embeddings/dino_small_embed_video.pt', 
    map_location=torch.device('cpu')
)
# dino_embed_video = torch.load(
#     'dataset/horsejump/models/dino_tracker/tracker_head_10000.pt', 
#     map_location=torch.device('cpu')
# )


print(dino_embed_video.shape)
# Select a specific frame (e.g., the first frame)
frame = dino_embed_video[0]  # This is a tensor of shape (C', H', W')

# Reduce channels by averaging across the channel dimension
#print(frame.shape)
frame = frame.mean(dim=0)  # Shape becomes (H', W')

# Convert the tensor to a numpy array
frame = frame.detach().cpu().numpy()

# Normalize to [0, 1] range
frame = (frame - frame.min()) / (frame.max() - frame.min())
print(frame.shape)

# Save the frame as an image
plt.imshow(frame, cmap='viridis')  # Use a colormap for single-channel data
plt.title("Frame 0 (DINO features)")
plt.axis("off")  # Turn off axis labels
plt.savefig("Frame 0 (DINO features.png")  # Save the image to a file
