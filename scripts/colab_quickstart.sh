#!/bin/bash
# Quickstart script for Google Colab
# Installs dependencies and runs a smoke test

set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running quick correctness test..."
python -m pytest tests/test_deterministic_mode.py::TestDeterministicMode::test_duplication_detection_disabled_in_deterministic_mode -v

echo "Running smoke benchmark..."
python -m scripts.benchmark_specdec_vs_vanilla \
    --model_name sshleifer/tiny-gpt2 \
    --device cpu \
    --impl torch \
    --batch_size 1 \
    --max_tokens 16 \
    --k 4 \
    --num_prompts 2 \
    --output_csv benchmarks/raw/colab_smoke.csv

echo "✅ Quickstart complete!"





