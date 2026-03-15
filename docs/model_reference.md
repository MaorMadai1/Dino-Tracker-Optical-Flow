# DINO Models Reference Guide

## DINOv2 Models

DINOv2 comes in 4 different model sizes, offering trade-offs between accuracy and computational cost:

### 1. DINOv2-Small (ViT-S/14)
- **Model name**: `dinov2_vits14`
- **Parameters**: ~22M
- **Feature dimension**: 384
- **Layers**: 12
- **Patch size**: 14×14
- **Best for**: Fast inference, limited GPU memory

### 2. DINOv2-Base (ViT-B/14)
- **Model name**: `dinov2_vitb14`
- **Parameters**: ~86M
- **Feature dimension**: 768
- **Layers**: 12
- **Patch size**: 14×14
- **Best for**: Good balance between speed and accuracy

### 3. DINOv2-Large (ViT-L/14)
- **Model name**: `dinov2_vitl14`
- **Parameters**: ~300M
- **Feature dimension**: 1024
- **Layers**: 24
- **Patch size**: 14×14
- **Best for**: Higher accuracy, more semantic features

### 4. DINOv2-Giant (ViT-g/14) 🔥
- **Model name**: `dinov2_vitg14`
- **Parameters**: ~1.1B
- **Feature dimension**: 1536
- **Layers**: 40
- **Patch size**: 14×14
- **Best for**: Best accuracy, research purposes

### Additional Variants
- `dinov2_vits8` / `dinov2_vitb8` - Smaller patch size (8×8) for higher resolution features

---

## DINOv3 Models

DINOv3 offers more model variants and includes ConvNeXt architectures:

### Vision Transformer (ViT) Models

#### 1. DINOv3-Small (ViT-S/16)
- **Model name**: `dinov3_vits16`
- **Parameters**: ~22M
- **Feature dimension**: 384
- **Layers**: 12
- **Heads**: 6
- **Patch size**: 16×16

#### 2. DINOv3-Small Plus (ViT-S/16+)
- **Model name**: `dinov3_vits16plus`
- **Parameters**: ~22M
- **Feature dimension**: 384
- **FFN layer**: SwiGLU (improved)

#### 3. DINOv3-Base (ViT-B/16)
- **Model name**: `dinov3_vitb16`
- **Parameters**: ~86M
- **Feature dimension**: 768
- **Layers**: 12
- **Heads**: 12

#### 4. DINOv3-Large (ViT-L/16)
- **Model name**: `dinov3_vitl16`
- **Parameters**: ~304M
- **Feature dimension**: 1024
- **Layers**: 24
- **Heads**: 16

#### 5. DINOv3-Large Plus (ViT-L/16+)
- **Model name**: `dinov3_vitl16plus`
- **Parameters**: ~304M
- **Feature dimension**: 1024
- **FFN layer**: SwiGLU

#### 6. DINOv3-Huge Plus (ViT-H/16+)
- **Model name**: `dinov3_vith16plus`
- **Parameters**: ~840M
- **Feature dimension**: ~1280

#### 7. DINOv3-Giant (ViT-7B/16) 🚀
- **Model name**: `dinov3_vit7b16`
- **Parameters**: ~7B (7 billion!)
- **Feature dimension**: 4096
- **Layers**: 40
- **Heads**: 32
- **FFN layer**: SwiGLU64

### ConvNeXt Models

#### 8. DINOv3-ConvNeXt-Tiny
- **Model name**: `dinov3_convnext_tiny`
- **Best for**: Dense prediction tasks

#### 9. DINOv3-ConvNeXt-Small
- **Model name**: `dinov3_convnext_small`
- **Parameters**: ~50M

#### 10. DINOv3-ConvNeXt-Base
- **Model name**: `dinov3_convnext_base`
- **Parameters**: ~89M

#### 11. DINOv3-ConvNeXt-Large
- **Model name**: `dinov3_convnext_large`

---

## Key Differences: DINOv2 vs DINOv3

| Feature | DINOv2 | DINOv3 |
|---------|--------|--------|
| **Patch Size** | 14×14 (mainly) | 16×16 |
| **Largest Model** | 1.1B (Giant) | 7B (7 Billion!) |
| **Position Encoding** | Learned absolute | RoPE (Rotary) |
| **FFN Options** | MLP only | MLP + SwiGLU variants |
| **ConvNeXt variants** | ❌ No | ✅ Yes |
| **Text alignment** | ❌ No | ✅ Yes (dino.txt) |

---

## Layer Selection Guidelines

- **Early layers** (0-10): Low-level features (edges, textures)
- **Middle layers** (11-25): Mid-level features (object parts, shapes)
- **Late layers** (26+): High-level semantic features (object identity)

### Common Configurations:
- **Point tracking**: Deep layers (e.g., layer 38 for vitg14) - semantic invariance
- **Segmentation/masks**: Middle layers (e.g., layer 23) - spatial precision

### Recommended Layers for DINOv2 ViT-S/14 (`dinov2_vits14`):

Since ViT-S/14 has **12 layers total (0-11)**, here are the recommended layers:

1. **Point Tracking / Feature Matching** (Recommended):
   - **Layer 11** (last layer) - Most semantic, best for tracking invariance
   - **Layer 10** - Alternative if layer 11 is too abstract
   - **Layers 9-11** - Can average multiple deep layers for robustness

2. **Segmentation / Dense Prediction**:
   - **Layers 8-10** - Balance between semantic and spatial information
   - **Layer 9** - Good middle-ground option

3. **General Feature Extraction**:
   - **Layer 11** - Default choice, highest semantic level
   - **Layers 10-11** - Average for more robust features

**Note**: Your current config uses `dino_layer: 11`, which is the recommended last layer for point tracking tasks. This provides maximum semantic invariance, which is crucial for tracking points across frames.

---

## "Plus" Variants in DINOv3

Models with "plus" suffix use **SwiGLU** (Swish-Gated Linear Units) instead of standard MLP:
- ✅ Better performance
- ✅ More efficient training
- ⚠️ Slightly higher computational cost

---

## Comparison Table

| Model | Name | Params | Feature Dim | Layers | Patch | Version |
|-------|------|--------|-------------|--------|-------|---------|
| ViT-S/14 | `dinov2_vits14` | 22M | 384 | 12 | 14 | v2 |
| ViT-B/14 | `dinov2_vitb14` | 86M | 768 | 12 | 14 | v2 |
| ViT-L/14 | `dinov2_vitl14` | 300M | 1024 | 24 | 14 | v2 |
| ViT-g/14 | `dinov2_vitg14` | 1.1B | 1536 | 40 | 14 | v2 |
| ViT-S/16 | `dinov3_vits16` | 22M | 384 | 12 | 16 | v3 |
| ViT-B/16 | `dinov3_vitb16` | 86M | 768 | 12 | 16 | v3 |
| ViT-L/16 | `dinov3_vitl16` | 304M | 1024 | 24 | 16 | v3 |
| ViT-7B/16 | `dinov3_vit7b16` | 7B | 4096 | 40 | 16 | v3 |

---

## Usage in Your Project

Your current config (`config/preprocessing.yaml`) uses:
```yaml
dino_model_name: dinov2_vitg14  # DINOv2-Giant
dino_layer: 38                   # Layer 38 out of 40
dino_stride: 7                   # Feature extraction stride
```

This is the most powerful DINOv2 model with very deep semantic features!

