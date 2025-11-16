import os
import torch
import torchvision.transforms as T
from PIL import Image
from upsamplers import load_loftup_checkpoint, norm
from tqdm import tqdm

# --- Function to Process Frames and Save Combined hr_feats ---
def process_frames_and_save_combined_hr_feats(frames_folder, output_path, transform, model, upsampler, device):
    image_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.tiff'))])
    if not image_files:
        raise FileNotFoundError(f"No images found in directory {frames_folder}")

    all_hr_feats = []

    for image_file in tqdm(image_files, desc=f"Processing {frames_folder}"):
        image_path = os.path.join(frames_folder, image_file)
        
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)

            # Extract low-resolution features
            with torch.no_grad():
                lr_feats = model.get_intermediate_layers(image_tensor, reshape=True)[0]  # 1, 384, 14, 14

                # Upsample to high-resolution features
                hr_feats = upsampler(lr_feats, image_tensor).to(device)

            print(f"{image_file}: hr_feats shape = {hr_feats.shape}, device = {hr_feats.device}")
            all_hr_feats.append(hr_feats.cpu())  # Move to CPU to save memory
            
            # Free memory
            del image_tensor, lr_feats, hr_feats
            torch.cuda.synchronize()  # Ensure all operations are complete

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    if all_hr_feats:
        # Combine all hr_feats into a single tensor
        combined_hr_feats = torch.cat(all_hr_feats, dim=0)
        torch.save(combined_hr_feats, output_path)
        print(f"Saved hr_feats to {output_path}, shape: {combined_hr_feats.shape}")
    else:
        print("No hr_feats found.")

# --- Main Script ---
if __name__ == "__main__":
    # Configuration
    image_folder = "dataset/tapvid-davis/davis_480/29/video"
    output_path = "output_folder/davis_480/29/loftup_embedded.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and upsampler setup
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    upsampler_path = "loftup/loftup_dinov2b.ckpt"
    upsampler = load_loftup_checkpoint(upsampler_path, 768, lr_pe_type="sine").to(device)

    # Image transformation
    w_new = 854 - 70-70-70
    h_new = 476 - 42-42-42
    transform = T.Compose([T.Resize((h_new, w_new)), T.ToTensor(), norm])

    # Process frames and save combined hr_feats
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_frames_and_save_combined_hr_feats(image_folder, output_path, transform, model, upsampler, device)

    # export PYTHONPATH=$PYTHONPATH:/home/dor.danino/dino-tracker/loftup
    # export PYTHONPATH=`pwd`:$PYTHONPATH
    # conda activate loftup-cuda12
