# Zero Copy Speculative Decoding Engine

A research-oriented inference engine for speculative decoding of large language models, optimized for memory-constrained hardware such as NVIDIA T4 and Apple Silicon.

## Quick Start

### Installation

```bash
pip install -e .
```

### Running Benchmarks

The canonical way to run benchmarks is using the `run_bench.py` script with a YAML configuration:

```bash
python scripts/run_bench.py --config configs/t4_gpt2_distil.yaml
```

This will:
- Run baseline (non-speculative) and speculative decoding experiments
- Use batch mode with ring-buffer KV cache (zero-copy path)
- Write results to `results/<run_id>/` (summary.json, summary.csv, system.json)
- Print a summary table with throughput, latency, and acceptance rates

### Example Configurations

- `configs/t4_gpt2_distil.yaml`: GPT2 + DistilGPT2 on T4
- `configs/t4_llama32.yaml`: Llama 3.2 3B + 1B on T4

You can customize any config by editing the YAML file or creating your own.

## Core Architecture

The engine implements three core ideas:

1. **Zero-copy ring buffer for KV cache**: Pre-allocated memory arena avoids repeated tensor allocation and concatenation.

2. **Pointer-based rollback**: O(1) rejection cost by updating integer pointers instead of slicing tensors.

3. **Parallel verification**: Constant-time verification across K by processing draft tokens in a single forward pass.

## Repository Layout

- `scripts/run_bench.py`: Canonical benchmark runner
- `configs/`: YAML configuration files for experiments
- `src/specdec/`: Core speculative decoding implementation
- `src/kernels/`: CUDA and Triton kernel implementations
- `tests/`: Unit and integration tests
- `archive/`: Archived legacy scripts, results, and documentation

## Results

Results are written to `results/<run_id>/`:
- `summary.json`: Aggregated results with summary statistics
- `summary.csv`: Detailed per-iteration results
- `system.json`: System configuration and metadata

The summary table printed at the end shows:
- Throughput (tokens/sec) for baseline and each K value
- Latency (ms) breakdown
- Acceptance rates for speculative decoding
- Timing breakdown (draft, verify, KV append)

## Deterministic Mode

Enable deterministic mode for correctness verification:

```yaml
deterministic: true
```

In deterministic mode with `draft_model == base_model` and `do_sample=False`, speculative decoding must produce bit-identical output to vanilla greedy decoding.

## Configuration Options

Key configuration options in YAML files:

- `base_model`: Base model name (required)
- `draft_model`: Draft model name (set to `null` for baseline-only)
- `device`: Device to use (`cuda`, `mps`, `cpu`, `auto`)
- `dtype`: Data type (`float16`, `float32`, `auto`)
- `batch_size`: Batch size for processing
- `max_tokens`: Maximum tokens to generate
- `iterations`: Number of iterations per configuration
- `k_values`: List of K values to test (e.g., `[1, 2, 3, 4]`)
- `kv_append_enabled`: Enable ring-buffer KV cache (default: `true`)
- `draft_force_hf_generate`: Force HF generate() path (default: `false`)
- `run_baseline`: Whether to run baseline comparison (default: `true`)

## Citation

If you use this repository in academic work, please cite:

```
LLM Inference Lab. A transparent research grade inference runtime for reproducible speculative decoding experiments. Repository available at https://github.com/GogoRit/llm-inference-lab
```

## License

MIT License. See [LICENSE](LICENSE) for details.
