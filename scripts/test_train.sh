#!/bin/bash
# Smoke test for train_gold_style.py

# Use a small number of steps
MAX_STEPS=2
OUTPUT_DIR="tmp_test_output"

# Clean up previous run
rm -rf $OUTPUT_DIR

echo "Starting smoke test..."
python3 scripts/train_gold_style.py \
    --student "Qwen/Qwen2.5-1.5B-Instruct" \
    --teacher "Qwen/Qwen2.5-1.5B-Instruct" \
    --max-steps $MAX_STEPS \
    --logging-steps 1 \
    --output-dir $OUTPUT_DIR \
    --assistant-only-loss \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --trust-remote-code

if [ $? -eq 0 ]; then
    echo "Smoke test PASSED!"
else
    echo "Smoke test FAILED!"
    exit 1
fi
