#!/bin/bash
# Run the full DINO-Tracker pipeline end-to-end for all videos:
#   1) Extract DINO embeddings        -> dataset/.../<id>/l<N>/dino_embed_video.pt
#   2) Run benchmark (visualize)      -> output_folder/davis_480/<id>/l<N>/dinov2_grid_*
#   3) Evaluate all videos            -> output_folder/eval_results/all_videos_*_l<N>.csv
#
# Stops immediately if any step fails. The DINO layer (and therefore the
# l<N> subfolder used everywhere) is read once from config/preprocessing.yaml.
#
# Usage: scripts/run_full_pipeline.sh
#        CONFIG=./config/other.yaml scripts/run_full_pipeline.sh

set -euo pipefail

CONFIG="${CONFIG:-./config/preprocessing.yaml}"
DINO_LAYER=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['dino_layer'])")
if [ -z "$DINO_LAYER" ]; then
    echo "Failed to read dino_layer from ${CONFIG}"
    exit 1
fi

echo "=========================================="
echo "Full pipeline for dino_layer=${DINO_LAYER} (l${DINO_LAYER})"
echo "Config: ${CONFIG}"
echo "=========================================="

step() {
    echo ""
    echo "------------------------------------------"
    echo "STEP $1: $2"
    echo "------------------------------------------"
}

PIPELINE_START=$(date +%s)

step 1 "Save DINO embeddings"
STEP_START=$(date +%s)
python3 preprocessing/save_dino_embed_video.py
echo "Step 1 finished in $(( $(date +%s) - STEP_START ))s"

step 2 "Run benchmark for all videos"
STEP_START=$(date +%s)
# Invoke via `bash` so the shebang/line-endings of the called script don't matter.
bash scripts/run_benchmark_all_videos.sh
echo "Step 2 finished in $(( $(date +%s) - STEP_START ))s"

step 3 "Run evaluation for all videos"
STEP_START=$(date +%s)
bash scripts/run_eval_all_videos.sh
echo "Step 3 finished in $(( $(date +%s) - STEP_START ))s"

echo ""
echo "=========================================="
echo "Pipeline completed successfully in $(( $(date +%s) - PIPELINE_START ))s"
echo "Layer:    l${DINO_LAYER}"
echo "Results:  output_folder/eval_results/all_videos_no_opt_l${DINO_LAYER}.csv"
echo "          output_folder/eval_results/all_videos_with_opt_l${DINO_LAYER}.csv"
echo "=========================================="
