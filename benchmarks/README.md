# Benchmark Results

This directory contains benchmark results from systematic experiments comparing speculative decoding vs vanilla decoding.

## Directory Structure

- `raw/` - Raw CSV files with individual run results
- `summary/` - Aggregated CSV files and Markdown reports

## Experiment Matrix

### GPT-2 Experiments (for T4/MPS)

**Models:**
- `gpt2` → `gpt2` (perfect draft)
- `distilgpt2` → `gpt2` (draft → target)

**Grid:**
- `batch_size`: [1, 2, 4, 8]
- `k`: [1, 2, 4, 8]
- `max_tokens`: 64
- `num_prompts`: 32

### Tiny Model Experiments (for CPU sanity checks)

**Models:**
- `sshleifer/tiny-gpt2` → `sshleifer/tiny-gpt2` (perfect draft)

**Grid:**
- `batch_size`: [1, 2, 4]
- `k`: [1, 2, 4]
- `max_tokens`: 64
- `num_prompts`: 32

## Running Benchmarks

### CPU Sanity Check

```bash
python -m scripts.run_benchmark_grid \
  --device cpu \
  --impl torch \
  --experiment_set tiny \
  --output_csv benchmarks/raw/cpu_tiny_grid.csv \
  --repeats 2 \
  --output_md benchmarks/summary/cpu_tiny_benchmarks.md
```

### T4 (CUDA)

```bash
python -m scripts.run_benchmark_grid \
  --device cuda \
  --impl torch \
  --experiment_set gpt2 \
  --output_csv benchmarks/raw/t4_gpt2_grid.csv \
  --repeats 3 \
  --output_md benchmarks/summary/t4_gpt2_benchmarks.md
```

### Mac (MPS)

```bash
python -m scripts.run_benchmark_grid \
  --device mps \
  --impl torch \
  --experiment_set gpt2 \
  --output_csv benchmarks/raw/mps_gpt2_grid.csv \
  --repeats 3 \
  --output_md benchmarks/summary/mps_gpt2_benchmarks.md
```

## Analyzing Results

Use the analysis script to interpret aggregated CSVs:

```bash
python -m scripts.analyze_benchmarks \
  --aggregated_csv benchmarks/raw/t4_gpt2_grid_aggregated.csv \
  --device cuda \
  --model_name gpt2
```

The analysis script will show:
- Configuration summary
- Detailed tables per config
- Speedup patterns (vs k, vs batch_size)
- Which configs have speedup > 1.0 (SpecDec wins)

## File Naming Convention

- Raw CSVs: `{device}_{model}_grid.csv`
- Aggregated CSVs: `{device}_{model}_grid_aggregated.csv`
- Markdown reports: `{device}_{model}_benchmarks.md`





