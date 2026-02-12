#!/bin/bash

# Run eval_benchmark.py for all DAVIS videos with and without optical flow optimization

# Configuration
DATASET_ROOT="output_folder/davis_480"
BENCHMARK_PICKLE="tapvid/tapvid_davis_data_strided.pkl"
OUTPUT_DIR="output_folder/eval_results"
DATASET_TYPE="tapvid"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Output files for combined results
OUTPUT_NO_OPT="$OUTPUT_DIR/all_videos_no_opt.csv"
OUTPUT_WITH_OPT="$OUTPUT_DIR/all_videos_with_opt.csv"

echo "=========================================="
echo "Running WITHOUT optical flow optimization"
echo "=========================================="

python eval/eval_benchmark.py \
    --dataset-root-dir "$DATASET_ROOT" \
    --benchmark-pickle-path "$BENCHMARK_PICKLE" \
    --out-file "$OUTPUT_NO_OPT" \
    --dataset-type "$DATASET_TYPE"

echo "=========================================="
echo "Running WITH optical flow optimization"
echo "=========================================="

python eval/eval_benchmark.py \
    --dataset-root-dir "$DATASET_ROOT" \
    --benchmark-pickle-path "$BENCHMARK_PICKLE" \
    --out-file "$OUTPUT_WITH_OPT" \
    --dataset-type "$DATASET_TYPE" \
    --optical-flow-opt

echo "=========================================="
echo "All videos evaluated!"
echo "=========================================="
echo "Results saved to:"
echo "  - Without optical flow: $OUTPUT_NO_OPT"
echo "  - With optical flow:    $OUTPUT_WITH_OPT"
