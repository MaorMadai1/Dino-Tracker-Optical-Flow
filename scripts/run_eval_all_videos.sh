#!/bin/bash

# Run eval_benchmark.py for all DAVIS videos with and without optical flow optimization

# Configuration
CONFIG="./config/preprocessing.yaml"
DATASET_ROOT="output_folder/davis_480"
BENCHMARK_PICKLE="tapvid/tapvid_davis_data_strided.pkl"
OUTPUT_DIR="output_folder/eval_results"
DATASET_TYPE="tapvid"

# Read dino_layer from the YAML so the CSV filename advertises which layer it covers,
# and so eval reads from the matching l{dino_layer} subfolder produced by the benchmark step.
DINO_LAYER=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['dino_layer'])")
if [ -z "$DINO_LAYER" ]; then
    echo "Failed to read dino_layer from ${CONFIG}"
    exit 1
fi
echo "Using dino_layer=${DINO_LAYER} (l${DINO_LAYER})"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Output files for combined results (per-layer so different layers don't overwrite each other)
OUTPUT_NO_OPT="$OUTPUT_DIR/all_videos_no_opt_l${DINO_LAYER}.csv"
OUTPUT_WITH_OPT="$OUTPUT_DIR/all_videos_with_opt_l${DINO_LAYER}.csv"

echo "=========================================="
echo "Running WITHOUT optical flow optimization"
echo "=========================================="

python eval/eval_benchmark.py \
    --dataset-root-dir "$DATASET_ROOT" \
    --benchmark-pickle-path "$BENCHMARK_PICKLE" \
    --out-file "$OUTPUT_NO_OPT" \
    --dataset-type "$DATASET_TYPE" \
    --config "$CONFIG"

echo "=========================================="
echo "Running WITH optical flow optimization"
echo "=========================================="

python eval/eval_benchmark.py \
    --dataset-root-dir "$DATASET_ROOT" \
    --benchmark-pickle-path "$BENCHMARK_PICKLE" \
    --out-file "$OUTPUT_WITH_OPT" \
    --dataset-type "$DATASET_TYPE" \
    --optical-flow-opt \
    --config "$CONFIG"

echo "=========================================="
echo "All videos evaluated!"
echo "=========================================="
echo "Results saved to:"
echo "  - Without optical flow: $OUTPUT_NO_OPT"
echo "  - With optical flow:    $OUTPUT_WITH_OPT"
