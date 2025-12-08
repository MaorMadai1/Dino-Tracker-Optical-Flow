# Optical Flow Methods for Point Tracking Optimization

This directory contains optical flow implementations to optimize point tracking by narrowing the search space.

## Overview

Instead of comparing query point features to **all** locations in each frame, we use optical flow to predict where points likely moved, then only search in a local region around that prediction.

**Benefits:**
- ⚡ **Faster**: Compare to ~100s of locations instead of 8,000+
- 🎯 **More Robust**: Optical flow handles large motions well
- 🔍 **Focused**: Computation where it matters most

---

## Method 1: OpenCV Lucas-Kanade (Sparse) ⭐ **Recommended to Start**

**Best for:** Tracking specific query points (your use case!)

### Installation
```bash
# Already in requirements.txt
pip install opencv-python
```

### Basic Usage

```python
import cv2
import numpy as np

def predict_point_with_optical_flow(point, prev_frame, next_frame, search_radius=20):
    """
    Use optical flow to predict where a point moved.
    
    Args:
        point: (x, y) tuple - query point in previous frame
        prev_frame: numpy array (H, W, 3) - RGB previous frame
        next_frame: numpy array (H, W, 3) - RGB next frame
        search_radius: int - search radius around prediction (pixels)
    
    Returns:
        predicted_point: (x, y) - predicted location in next frame
        search_region: (x_min, y_min, x_max, y_max) - bounding box for feature search
        status: bool - True if tracking succeeded
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    
    # Prepare points for Lucas-Kanade (needs specific format)
    points = np.array([[[point[0], point[1]]]], dtype=np.float32)
    
    # Compute optical flow
    predicted, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,          # Previous frame (grayscale)
        next_gray,          # Next frame (grayscale)
        points,             # Points to track
        None,               # Next points (None = compute)
        winSize=(21, 21),   # Search window size
        maxLevel=3,         # Pyramid levels for multi-scale
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    if status[0] == 1:  # Successfully tracked
        pred_x, pred_y = predicted[0][0]
        
        # Define search region
        x_min = max(0, int(pred_x - search_radius))
        y_min = max(0, int(pred_y - search_radius))
        x_max = min(next_frame.shape[1], int(pred_x + search_radius))
        y_max = min(next_frame.shape[0], int(pred_y + search_radius))
        
        return (pred_x, pred_y), (x_min, y_min, x_max, y_max), True
    else:
        return None, None, False  # Tracking failed
```

### Example

```python
# Load frames
prev_frame = load_frame("frame_000.jpg")  # (476, 854, 3)
next_frame = load_frame("frame_001.jpg")

# Query point
point = (427, 238)

# Predict with optical flow
predicted_point, search_region, success = predict_point_with_optical_flow(
    point, prev_frame, next_frame, search_radius=30
)

if success:
    print(f"Point moved from {point} to {predicted_point}")
    print(f"Search region: {search_region}")
    # Now compare DINO features only within search_region
```

### Parameters to Tune

| Parameter | Default | Description | When to Increase | When to Decrease |
|-----------|---------|-------------|------------------|------------------|
| `winSize` | (21, 21) | Search window | Large motions | Small motions, speed |
| `maxLevel` | 3 | Pyramid levels | Large motions | Speed, small motions |
| `search_radius` | 20 | Feature search radius | Uncertain motion | Speed, confident flow |

---

## Method 2: OpenCV Farneback (Dense)

**Best for:** When you need flow at every pixel

### Usage

```python
def compute_dense_optical_flow(prev_frame, next_frame):
    """
    Compute dense optical flow for entire frame.
    
    Args:
        prev_frame: (H, W, 3) RGB frame
        next_frame: (H, W, 3) RGB frame
    
    Returns:
        flow: (H, W, 2) where [:,:,0]=dx, [:,:,1]=dy
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,      # Previous frame
        next_gray,      # Next frame
        None,           # Flow (None = compute new)
        pyr_scale=0.5,  # Pyramid scale (< 1)
        levels=3,       # Number of pyramid layers
        winsize=15,     # Averaging window size
        iterations=3,   # Iterations at each level
        poly_n=5,       # Polynomial expansion neighborhood
        poly_sigma=1.2, # Gaussian std for polynomial expansion
        flags=0
    )
    
    return flow

def get_flow_at_point(flow, point):
    """Extract flow vector at specific point."""
    x, y = int(point[0]), int(point[1])
    dx, dy = flow[y, x]  # Note: indexing is [y, x] for numpy
    return dx, dy
```

### Example

```python
# Compute dense flow
flow = compute_dense_optical_flow(prev_frame, next_frame)  # (476, 854, 2)

# Get flow at query point
point = (427, 238)
dx, dy = get_flow_at_point(flow, point)

predicted_x = point[0] + dx
predicted_y = point[1] + dy
print(f"Point moved by ({dx:.2f}, {dy:.2f}) pixels")
```

---

## Method 3: RAFT (PyTorch, State-of-the-art) 🔥

**Best for:** Highest accuracy, handles large motions and occlusions

### Installation

```bash
pip install torch torchvision
```

### Usage

```python
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large

class RAFTFlowPredictor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = raft_large(pretrained=True).to(device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @torch.no_grad()
    def compute_flow(self, frame1, frame2):
        """
        Compute optical flow using RAFT.
        
        Args:
            frame1: numpy array (H, W, 3) or PIL Image
            frame2: numpy array (H, W, 3) or PIL Image
        
        Returns:
            flow: numpy array (H, W, 2)
        """
        # Prepare inputs
        if isinstance(frame1, np.ndarray):
            frame1 = Image.fromarray(frame1)
            frame2 = Image.fromarray(frame2)
        
        img1 = self.transform(frame1).unsqueeze(0).to(self.device)
        img2 = self.transform(frame2).unsqueeze(0).to(self.device)
        
        # Compute flow
        flow_predictions = self.model(img1, img2)
        flow = flow_predictions[-1][0]  # (2, H, W)
        
        # Convert to numpy
        flow = flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        return flow
    
    def get_flow_at_point(self, flow, point):
        """Extract flow at specific point."""
        x, y = int(point[0]), int(point[1])
        dx, dy = flow[y, x]
        return dx, dy
```

### Example

```python
# Initialize predictor
flow_predictor = RAFTFlowPredictor(device='cuda')

# Compute flow
flow = flow_predictor.compute_flow(prev_frame, next_frame)

# Get prediction for query point
point = (427, 238)
dx, dy = flow_predictor.get_flow_at_point(flow, point)

predicted_point = (point[0] + dx, point[1] + dy)
```

### Performance Comparison

| Method | Speed (FPS) | Accuracy | GPU Memory | Use Case |
|--------|-------------|----------|------------|----------|
| Lucas-Kanade | ~100 | Good | Low | Sparse points, real-time |
| Farneback | ~30 | Good | Low | Dense flow, CPU-friendly |
| RAFT | ~5-10 | Excellent | High | Research, offline processing |

---

## Integration with DINO Feature Tracking

### Complete Workflow

```python
def track_point_with_optical_flow(
    point,
    point_feature,
    prev_frame,
    next_frame,
    next_frame_features,
    search_radius_pixels=30
):
    """
    Track a point using optical flow + DINO features.
    
    Args:
        point: (x, y) in prev_frame
        point_feature: (1, C) DINO feature at point
        prev_frame: (H, W, 3) RGB frame
        next_frame: (H, W, 3) RGB frame
        next_frame_features: (C, H_f, W_f) DINO features
        search_radius_pixels: search radius around optical flow prediction
    
    Returns:
        best_match: (x, y) in next_frame
        confidence: similarity score
    """
    # Step 1: Predict with optical flow
    predicted_point, search_region, success = predict_point_with_optical_flow(
        point, prev_frame, next_frame, search_radius_pixels
    )
    
    if not success:
        # Fall back to full frame search if optical flow fails
        search_region = (0, 0, next_frame.shape[1], next_frame.shape[0])
        predicted_point = point
    
    x_min, y_min, x_max, y_max = search_region
    
    # Step 2: Convert to feature space
    H, W = next_frame.shape[:2]
    C, H_f, W_f = next_frame_features.shape
    
    x_min_feat = int(x_min * W_f / W)
    x_max_feat = int(x_max * W_f / W)
    y_min_feat = int(y_min * H_f / H)
    y_max_feat = int(y_max * H_f / H)
    
    # Step 3: Extract local features
    local_features = next_frame_features[:, y_min_feat:y_max_feat, x_min_feat:x_max_feat]
    local_features = local_features.reshape(C, -1).T  # (N_local, C)
    
    # Step 4: Compute similarity
    from torch.nn import CosineSimilarity
    similarities = CosineSimilarity(dim=1)(point_feature, local_features)
    
    # Step 5: Find best match
    best_idx = similarities.argmax()
    best_confidence = similarities[best_idx].item()
    
    # Convert back to image coordinates
    local_h = y_max_feat - y_min_feat
    local_w = x_max_feat - x_min_feat
    best_y_feat = best_idx // local_w + y_min_feat
    best_x_feat = best_idx % local_w + x_min_feat
    
    best_x = best_x_feat * W / W_f
    best_y = best_y_feat * H / H_f
    
    return (best_x, best_y), best_confidence
```

### Speedup Analysis

```python
# Without optical flow:
# Compare to ALL locations: 8,296 comparisons per frame

# With optical flow (search_radius=30):
# Feature map: 68×122, stride=7
# Search region in features: ~8×8 = 64 locations
# Speedup: 8,296 / 64 = ~130x faster!
```

---

## Choosing the Right Method

### Decision Tree

```
Do you need real-time performance?
├─ YES → Use Lucas-Kanade (sparse)
│   └─ Tracking specific query points? → ✅ PERFECT FIT
│
└─ NO → Do you need highest accuracy?
    ├─ YES → Use RAFT (deep learning)
    │   └─ Have GPU? → ✅ Use RAFT
    │   └─ CPU only? → Use Farneback
    │
    └─ NO → Use Farneback (good balance)
```

### Recommendations by Use Case

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| **Your project: TapVid tracking** | Lucas-Kanade | Sparse points, fast, good enough |
| Research paper (best results) | RAFT | Highest accuracy |
| Real-time demo | Lucas-Kanade | Fastest |
| Dense correspondence | RAFT or Farneback | Need flow everywhere |

---

## Advanced: Handling Edge Cases

### 1. Optical Flow Failure Detection

```python
def is_flow_reliable(error, threshold=10.0):
    """Check if optical flow is reliable."""
    return error[0] < threshold

# Usage
predicted, status, error = cv2.calcOpticalFlowPyrLK(...)
if status[0] == 1 and is_flow_reliable(error):
    # Use optical flow prediction
else:
    # Fall back to full frame search
```

### 2. Multi-Scale Search

```python
def adaptive_search_radius(point_velocity, base_radius=20, max_radius=50):
    """
    Adapt search radius based on point velocity.
    Fast moving points need larger search regions.
    """
    velocity_magnitude = np.sqrt(point_velocity[0]**2 + point_velocity[1]**2)
    radius = min(base_radius + int(velocity_magnitude * 0.5), max_radius)
    return radius
```

### 3. Temporal Consistency

```python
# Use previous predictions to inform current search
predicted_points_history = []

# If point velocity is consistent, trust optical flow more
if len(predicted_points_history) >= 3:
    velocities = np.diff(predicted_points_history, axis=0)
    velocity_std = np.std(velocities, axis=0)
    
    if np.all(velocity_std < 5.0):  # Smooth motion
        search_radius = 15  # Can use smaller radius
    else:  # Erratic motion
        search_radius = 40  # Use larger radius
```

---

## Performance Tips

1. **Precompute Flow**: If tracking many points in same frame pair, compute flow once
   ```python
   flow = compute_dense_flow(prev_frame, next_frame)  # Once
   for point in points:
       dx, dy = get_flow_at_point(flow, point)  # Reuse flow
   ```

2. **Batch Processing**: Process multiple points together
   ```python
   points = np.array([[x1, y1], [x2, y2], ...])
   predicted_points, status, error = cv2.calcOpticalFlowPyrLK(
       prev_gray, next_gray, points.reshape(-1, 1, 2), None
   )
   ```

3. **Resolution Tradeoff**: Downsample for speed, upsample results
   ```python
   # Compute flow at half resolution
   scale = 0.5
   flow = compute_flow(downsample(prev), downsample(next))
   flow = flow / scale  # Scale flow vectors
   ```

---

## References

- **Lucas-Kanade**: [OpenCV Documentation](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)
- **Farneback**: [OpenCV Documentation](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af)
- **RAFT**: ["RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"](https://arxiv.org/abs/2003.12039)
- **TapVid Benchmark**: [Paper](https://arxiv.org/abs/2211.03726)

---

## Next Steps

1. **Start Simple**: Implement Lucas-Kanade first
2. **Measure Impact**: Compare tracking speed with/without optical flow
3. **Tune Parameters**: Adjust `search_radius` for your dataset
4. **Upgrade if Needed**: Switch to RAFT if accuracy is insufficient

Good luck with your optical flow optimization! 🚀

