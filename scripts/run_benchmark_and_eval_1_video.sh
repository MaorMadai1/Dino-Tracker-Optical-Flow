#!/bin/bash

# Run visualization and evaluation for a single DAVIS video
# Usage: ./run_benchmark_and_eval_1_video.sh --video_id=0 [--optical_flow_opt] [--dynamic_bbox]

# Default values
VIDEO_ID=""
OPTICAL_FLOW_OPT_FLAG=""
DYNAMIC_BBOX_FLAG=""

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --video_id=*)
            VIDEO_ID="${arg#*=}"
            shift
            ;;
        --optical_flow_opt)
            OPTICAL_FLOW_OPT_FLAG="--optical_flow_opt"
            shift
            ;;
        --dynamic_bbox)
            DYNAMIC_BBOX_FLAG="--dynamic_bbox"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 --video_id=<id> [--optical_flow_opt] [--dynamic_bbox]"
            exit 1
            ;;
    esac
done

# Check if video_id is provided
if [ -z "$VIDEO_ID" ]; then
    echo "Error: --video_id is required"
    echo "Usage: $0 --video_id=<id> [--optical_flow_opt] [--dynamic_bbox]"
    exit 1
fi

# Configuration
DATASET_ROOT="dataset/tapvid-davis/davis_480"
OUTPUT_ROOT="output_folder/davis_480"
BENCHMARK_PICKLE="tapvid/tapvid_davis_data_strided.pkl"
DATASET_TYPE="tapvid"

# Build output paths for evaluation
# Note: Folder names don't change with optical flow - only the filenames inside do (of0 vs of1)
TRAJECTORIES_DIR="${OUTPUT_ROOT}/${VIDEO_ID}/dinov2_grid_trajectory"
OCCLUSIONS_DIR="${OUTPUT_ROOT}/${VIDEO_ID}/dinov2_grid_occlusion"

# Determine output file suffix based on flags
if [ -n "$OPTICAL_FLOW_OPT_FLAG" ] && [ -n "$DYNAMIC_BBOX_FLAG" ]; then
    SUFFIX="_opt_dynamic"
elif [ -n "$OPTICAL_FLOW_OPT_FLAG" ]; then
    SUFFIX="_opt"
else
    SUFFIX=""
fi

OUT_FILE="${OUTPUT_ROOT}/${VIDEO_ID}/dinov2_grid${SUFFIX}_metrics.csv"

echo "=========================================="
echo "Processing video ${VIDEO_ID}"
echo "Optical flow optimization: $([ -n "$OPTICAL_FLOW_OPT_FLAG" ] && echo "ENABLED" || echo "DISABLED")"
echo "Dynamic bbox: $([ -n "$DYNAMIC_BBOX_FLAG" ] && echo "ENABLED" || echo "DISABLED")"
echo "=========================================="

# Step 1: Run visualization/benchmark
echo ""
echo "Step 1: Running benchmark visualization..."
echo "Command: python visualization/visualize_raw_heatmaps.py --point=benchmark_grid --video_id=${VIDEO_ID} ${OPTICAL_FLOW_OPT_FLAG} ${DYNAMIC_BBOX_FLAG}"
python visualization/visualize_raw_heatmaps.py \
    --point=benchmark_grid \
    --video_id=${VIDEO_ID} \
    ${OPTICAL_FLOW_OPT_FLAG} \
    ${DYNAMIC_BBOX_FLAG}

# Check if visualization was successful
if [ $? -ne 0 ]; then
    echo "Error: Visualization failed for video ${VIDEO_ID}"
    exit 1
fi

echo ""
echo "Visualization completed successfully!"

# Step 2: Run evaluation
echo ""
echo "Step 2: Running evaluation..."
echo "Trajectories: ${TRAJECTORIES_DIR}"
echo "Occlusions: ${OCCLUSIONS_DIR}"
echo "Output file: ${OUT_FILE}"

python eval/eval_benchmark_1_video.py \
    --trajectories-dir "${TRAJECTORIES_DIR}" \
    --occlusions-dir "${OCCLUSIONS_DIR}" \
    --video-idx ${VIDEO_ID} \
    --benchmark-pickle-path "${BENCHMARK_PICKLE}" \
    --out-file "${OUT_FILE}" \
    --dataset-type "${DATASET_TYPE}" \
    ${OPTICAL_FLOW_OPT_FLAG} \
    ${DYNAMIC_BBOX_FLAG}

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed for video ${VIDEO_ID}"
    exit 1
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
echo "Results saved to: ${OUT_FILE}"
