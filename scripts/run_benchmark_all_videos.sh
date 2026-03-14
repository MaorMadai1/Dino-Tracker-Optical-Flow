#!/bin/bash

# Run benchmark grid points for all DAVIS videos (3 to 75)

echo "=========================================="
echo "Running WITHOUT optical flow optimization"
echo "=========================================="

for video_id in $(seq 0 29); do
    echo "Processing video $video_id without optical flow..."
    python visualization/visualize_raw_heatmaps.py --point=benchmark_grid --video_id=$video_id
done

echo "=========================================="
echo "Running WITH optical flow optimization"
echo "=========================================="

for video_id in $(seq 0 29); do
    echo "Processing video $video_id with optical flow..."
    python visualization/visualize_raw_heatmaps.py --point=benchmark_grid --video_id=$video_id --optical_flow_opt
done

echo "=========================================="
echo "All videos processed!"
echo "=========================================="
