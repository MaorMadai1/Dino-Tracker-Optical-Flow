import gc
import torch
from einops import rearrange

from models.extractor import VitExtractor
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage



@torch.no_grad()
def get_dino_features_video(video, facet='tokens', device: str = 'cuda:0', model_name="dinov2_vitb14", stride=7):
    """
    Args:
        video (torch.tensor): Tensor of the input video, of shape: T x 3 x H x W.
            T- number of frames. C- number of RGB channels (most likely 3), W- width, H- height.
        device (str, optional):indicating device type. Defaults to 'cuda:0'.

    Returns:
        dino_keys_video: DINO keys from last layer for each frame. Shape: (T x C x H//8 x W//8).
            T- number of frames. C - DINO key embedding dimension for patch.
    """
    dino_extractor = VitExtractor(model_name=model_name, device=device, stride=stride)
    dino_extractor = dino_extractor.eval().to(device)
    imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ph = dino_extractor.get_height_patch_num(video[[0]].shape)
    pw = dino_extractor.get_width_patch_num(video[[0]].shape)
    dino_embedding_dim = dino_extractor.get_embedding_dim(model_name)
    n_layers = dino_extractor.get_n_layers()
    dino_features_video = torch.zeros(size=(video.shape[0], dino_embedding_dim, ph, pw))
    for i in range(video.shape[0]):
        dino_input = imagenet_norm(video[[i]]).to(device)
        if facet == 'tokens':
            features = dino_extractor.get_feature_from_input(dino_input, layers=[n_layers - 1])
        elif facet == 'keys':
            features = dino_extractor.get_keys_from_input(dino_input, layers=[n_layers - 1])
        else:
            raise NotImplementedError(f"facet {facet} not implemented")
        features = rearrange(features[:, 1:, :], "heads (ph pw) ch -> (ch heads) ph pw", ph=ph, pw=pw)
        dino_features_video[i] = features.cpu()
    # interpolate to the original video length
    del dino_extractor
    torch.cuda.empty_cache()
    gc.collect()
    return dino_features_video


def bilinear_interpolate_video(video:torch.tensor, points:torch.tensor, h:int, w:int, t:int, normalize_h=False, normalize_w=False, normalize_t=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x T x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: B x 3.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x 1 x B x 1.
    """
    samples = points[None, None, :, None].detach().clone() # expand shape B x 3 TO (1 x 1 x B x 1 x 3), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] * 2 - 1  # normalize to [-1,1]
    if normalize_t:
        if t > 1:
            samples[:, :, :, :, 2] = samples[:, :, :, :, 2] / (t - 1)  # normalize to [0,1]
        samples[:, :, :, :, 2] = samples[:, :, :, :, 2] * 2 - 1  # normalize to [-1,1]
    return torch.nn.functional.grid_sample(video, samples, align_corners=True, padding_mode ='border') # points out-of bounds are padded with border values


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append((index % dim).item())
        index = index // dim
    return tuple(reversed(out))


def overlay_heatmap_jpg(image_pil, heatmap, upsample_mode="bilinear"):
    pil_upsample = Image.BILINEAR if upsample_mode == "bilinear" else Image.NEAREST
    heatmap_pil = Image.fromarray(heatmap)
    heatmap_pil = heatmap_pil.convert('RGB')
    heatmap_pil = heatmap_pil.resize(image_pil.size, resample=pil_upsample)

    alpha = 0.5
    overlay = Image.blend(image_pil, heatmap_pil, alpha)

    return overlay


def overlay_bounding_box(image, center_coordinate, box_size):
    """
    Overlay a bounding box on the image.

    Parameters:
    - image: PIL Image object
    - center_coordinate: Tuple (x, y) representing the center coordinate of the bounding box
    - box_size: Tuple (width, height) representing (half) the size of the bounding box

    Returns:
    - PIL Image object with the bounding box overlaid
    """
    draw = ImageDraw.Draw(image)

    # Calculate the coordinates of the bounding box
    x_center, y_center = center_coordinate
    width = box_size
    height = box_size
    x1 = x_center - width
    y1 = y_center - height
    x2 = x_center + width
    y2 = y_center + height

    # Draw the bounding box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return image


def overlay_point(image_tensor, x, y, r=6, c="red"):
    # Check if the image is already a PIL image or a tensor
    if isinstance(image_tensor, torch.Tensor):
        to_pil_image = ToPILImage()
        image_pil = to_pil_image(image_tensor)
    else:
        image_pil = image_tensor  # If it's already a PIL image, use it directly

    # Create a new ImageDraw object
    draw = ImageDraw.Draw(image_pil)

    # Draw a red dot at the specified coordinate
    draw.ellipse((x-r, y-r, x+r, y+r), fill=c)

    return image_pil

def overlay_cross(image_pil, x, y, r=6, c="green"):
    # Create a new ImageDraw object
    draw = ImageDraw.Draw(image_pil)

    # Draw a red cross at the specified coordinate
    draw.line([(x-r, y-r), (x+r, y+r)], width=r, fill=c)
    draw.line([(x-r, y+r), (x+r, y-r)], width=r, fill=c)

    return image_pil



def write_frame_number_on_image(image, frame_idx: int, color=(255, 255, 255)):
    """draws frame number on top left corner of image and returns it"""
    # create ImageDraw object
    draw = ImageDraw.Draw(image)
    # draw text on top right corner
    text = "Frame: {}".format(frame_idx)
    draw.text((0, 0), text, fill=color, )
    return image

def write_text_on_image(image, text, color=(255, 255, 255)):
    """draws frame number on top right corner of image and returns it"""
    # create ImageDraw object
    draw = ImageDraw.Draw(image)
    pixel_margin = len(text) * 6
    # draw text on top right corner
    draw.text((image.size[0]-pixel_margin, 0), text, fill=color, )
    return image


def concat_images_w(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im


def concat_images_h(images):
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    return new_im
