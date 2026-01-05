# Google Colab Setup Guide

## Quick Setup

### 1. Clone Repository (dev branch)
```bash
!git clone -b dev https://github.com/GogoRit/llm-inference-lab.git
%cd llm-inference-lab
```

### 2. Install Dependencies
```bash
!pip install -r requirements.txt

# Create benchmarks directory structure
!mkdir -p benchmarks/raw benchmarks/summary
```

### 3. Optional: Install CUDA-appropriate PyTorch for T4 GPU
If you're using a T4 GPU in Colab, you may need to install the correct PyTorch version:

```bash
# For CUDA 11.8 (common in Colab)
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify Installation
```bash
# Quick correctness test
!python -m pytest tests/test_deterministic_mode.py::TestDeterministicMode::test_duplication_detection_disabled_in_deterministic_mode -v
```

## Running Benchmarks

### Single Benchmark Run
```bash
!python -m scripts.benchmark_specdec_vs_vanilla \
    --model_name gpt2 \
    --device cuda \
    --impl torch \
    --batch_size 2 \
    --max_tokens 64 \
    --k 4 \
    --num_prompts 4 \
    --output_csv benchmarks/raw/colab_smoke.csv
```

### Quick Grid Run (Small)
```bash
!python -m scripts.run_benchmark_grid \
    --device cuda \
    --impl torch \
    --output_csv benchmarks/raw/colab_grid.csv \
    --repeats 1 \
    --experiment_set tiny
```

### Full Grid Run (GPT-2)
```bash
!python -m scripts.run_benchmark_grid \
    --device cuda \
    --impl torch \
    --output_csv benchmarks/raw/colab_gpt2_grid.csv \
    --repeats 3 \
    --experiment_set gpt2
```

### Analyze Results
```bash
!python -m scripts.analyze_benchmarks \
    --aggregated_csv benchmarks/raw/colab_gpt2_grid_aggregated.csv
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` (e.g., `--batch_size 1`)
- Reduce `max_tokens` (e.g., `--max_tokens 32`)
- Use smaller model (e.g., `--model_name sshleifer/tiny-gpt2`)

### Import Errors
- Ensure you're in the repo directory: `%cd llm-inference-lab`
- Check Python path: `import sys; print(sys.path)`
- Reinstall: `!pip install -e .` (if setup.py exists)

### Slow Performance
- Colab free tier has limited GPU time
- Use `--experiment_set tiny` for quick tests
- Reduce `--repeats` to 1 for faster runs

