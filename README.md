# LLM Inference Lab

LLM Inference Lab is a transparent, research-grade inference runtime for reproducible speculative decoding experiments across CPU, MPS, and CUDA.

## Strategy Note

As of Phase 3C.5, LLM Inference Lab follows an MPS-first optimization approach. All new features are validated locally on Apple Silicon (MPS) before limited CUDA (Kaggle/A100) benchmarking. This ensures efficient use of GPU credits and rapid iteration on Mac hardware.

The long-term goal is to provide open, reproducible baselines for speculative decoding comparable to vLLM and TensorRT-LLM—bridging academic transparency with production performance.

## What's New

### Phase 4A: In Progress – Performance Optimization and Stabilization

Phase 4A focuses on optimizing hot paths and reducing overhead for paper submission readiness. Key optimizations include removing debug print overhead, eliminating unnecessary CPU-GPU synchronizations, and consolidating redundant validation code.

**Completed Optimizations**:
- Hot path logging removed (gated with `SPECDEC_DEBUG` flag)
- Async synchronization optimized for true stream overlap
- Token validation consolidated into centralized helper
- Partial KV cache reuse enhanced
- **CPU-GPU Memory Optimization**: Removed unnecessary `.item()` calls, deferred CPU transfers, GPU-only validation
- **Batch Processing (Phase 4A.1)**: Attention mask support, optimized padding, reduced CPU-GPU syncs

**Phase 4A Results** (Tesla T4, 64 tokens, 800 samples):
- Throughput: 5.88 tok/s (stable vs Phase 3D baseline 5.97 tok/s)
- Acceptance: 39.8% (maintained)
- Success Rate: 100% (800/800)
- **Finding**: Hot-path optimizations maintained stability; verified no regression

**Phase 4A.1 Batch Processing Results** (Tesla T4, 64 tokens, 200 samples × 4 K values):
- **Individual Throughput**: 5.76±0.15 tok/s (K=1: 5.92±0.26, K=2-4: 5.69-5.73 tok/s)
- **Batch Aggregate**: ~58 tok/s (10 prompts, working as expected)
- **Acceptance Rate**: 39.815% (consistent across all K values)
- **Success Rate**: 100% (800/800 samples)
- **Stream Overlap**: ~47-58ms savings per step (draft: ~51ms, verify: ~88ms)
- **GPU Utilization**: 12.2% (low compute utilization - compute-bound, not memory-bound)
- **Finding**: Batch processing functional; individual throughput matches single-prompt baseline; GPU compute is bottleneck

**New (2025-11-18)**: Added Kaggle Tesla T4 experiments with `meta-llama/Llama-3.2-3B` + `meta-llama/Llama-3.2-1B` (K=1). On T4, throughput peaks at batch size 1 (8.45 tok/s) and drops at batch sizes 2 and 4, highlighting that speculative decoding does not trivially benefit from larger batches on small GPUs. KV append was disabled in these runs (`SPECDEC_ENABLE_KV_APPEND=0`), and baseline mode with `draft_model=None` is still under repair (see Current Limitations).

### Phase 3D: Complete – Production-Ready Pipeline Validated

Phase 3D represents the transition from kernel-level optimization (Phase 3C) to full GPU runtime optimization. All Phase 3D features have been validated on Apple Silicon MPS and Tesla T4 CUDA hardware. The speculative pipeline now completes full K-sweeps on production hardware without CUDA asserts, with KV-cache reuse properly gated on full acceptance to maintain consistency.

**Latest Achievements**:
- GPU Optimization: CUDA stream overlap, graph capture, structured profiling
- Deterministic Reproducibility: Centralized seeding utility with environment flags
- Structured Metrics: Per-step GPU event timers with JSON output
- Dry-Run Profiling: Latency-only profiling without model compute
- Enhanced Memory Tracking: CUDA and MPS memory stats integration
- MPS Validation Complete: All Phase 3D features validated on Apple Silicon

**Phase 3D Features**:
- Deterministic Seeding: `SPECDEC_DETERMINISTIC=1` for reproducible benchmarks
- Structured Profiler: `SPECDEC_PROFILE=1` for per-step event timing
- CUDA Graph Capture: `SPECDEC_CUDA_GRAPH=1` for steady decoding optimization
- Stream Overlap: `SPECDEC_PARALLEL_STREAMS=1` with event-based synchronization
- Dry-Run Mode: `SPECDEC_DRY_RUN=1` for latency profiling without model load

**Current Status**:
- MPS: 80 runs, 100% success, all Phase 3D features functional
- CUDA: T4 K-sweep complete (800 samples, 100% success), A100/H100 validation planned
- Test Health: GitHub CI passing (black/isort/flake8 with exit-zero, mypy with ignore-missing-imports, pytest -k "not gpu")
- Unit Tests: 14 tests passing (2 skipped on non-CUDA)

## Project Phases

| Phase | Status | Description | Key Features |
|-------|--------|-------------|--------------|
| **1A** | Complete | Baseline runner | HuggingFace OPT-125M, CPU/MPS support |
| **1B** | Complete | Benchmark client | HTTP client, dual-mode benchmarking |
| **1C** | Complete | CI/CD pipeline | Automated testing, code quality |
| **2A** | Complete | Speculative decoding | Draft-and-verify pipeline |
| **2B** | Complete | Advanced policies | Acceptance policies, adaptive K controllers |
| **2C** | Complete | Advanced modes | Medusa-lite, EAGLE-lite implementations |
| **3A** | Complete | Local optimization | Mixed precision, profiling, K-sweep analysis |
| **3B** | Complete | GPU pre-check | CUDA readiness, memory-safe loading |
| **3C** | Complete | CUDA kernels | Custom kernels, registry, detailed metrics |
| **3C.5** | Complete | KV cache integration | Efficient token appending without recomputation |
| **3D** | Complete | GPU optimization | Stream overlap, graph capture, structured profiling (CUDA validation complete) |
| **4A** | In Progress | Performance optimization | Hot path optimization, CPU-GPU sync reduction, batch processing (Phase 4A.1) |
| **4B** | Planned | Advanced quantization | INT8/INT4 quantization techniques |
| **4C** | Planned | Layer/model parallelism | Multi-GPU support for 7B+ models |
| **4D** | Planned | Speculative tree decoding | Tree-based acceptance strategies |

## Performance Results

### 1. Phase 3D – GPT2 / DistilGPT2 (Historical Baseline)

**MPS Results** (Apple Silicon, GPT2-124M + DistilGPT2, 32 tokens, 5 iterations):

| K Value | Throughput (tok/s) | Acceptance Rate | Latency (ms) | Memory (MB) |
|---------|-------------------|-----------------|--------------|-------------|
| K=1 | 9.12±0.45 | 18.2±8.1% | 3,510±173 | 275±12 |
| K=2 | 9.23±0.38 | 16.8±7.2% | 3,470±143 | 280±15 |
| K=3 | 9.15±0.42 | 17.1±6.8% | 3,500±161 | 285±18 |
| **K=4** | **9.55±0.51** | **16.9±7.5%** | **3,350±179** | **290±20** |

**CUDA T4 Results** (Tesla T4, GPT2 + DistilGPT2):

**Run #1**: 32 tokens × 100 iterations, fp16

[Results folder](docs/results/2025-10-30-T4-Phase3D-Run1-32tok-100iter-fp16/)

| K | Latency (ms mean ± std) | Throughput (tok/s mean ± std) | Acceptance (%) mean ± std |
|---|-------------------------|--------------------------------|----------------------------|
| 1 | 3743.07 ± 1031.39 | 17.24 ± 4.96 | 21.38 ± 12.43 |
| 2 | 3904.52 ± 908.77 | 17.66 ± 5.67 | 22.72 ± 14.20 |
| 3 | 3916.34 ± 976.50 | 17.53 ± 5.96 | 22.28 ± 15.06 |
| 4 | 3870.67 ± 1012.93 | 17.22 ± 4.96 | 21.88 ± 12.71 |

Summary: ~17.4 tok/s average across K=1–4 (approximately 1.8–2.0× vs MPS), best acceptance ~22.7% at K=2, 100% success rate, no OOM.

**Run #2**: 64 tokens × 200 samples, fp16

| K | Latency (ms mean ± std) | Throughput (tok/s mean ± std) | Acceptance (%) mean ± std | Success Rate |
|---|-------------------------|--------------------------------|----------------------------|--------------|
| 1 | 177.3 ± 3.6 | 5.64 ± 0.11 | 39.8 ± 0.0 | 100% (200/200) |
| 2 | 170.9 ± 9.1 | 5.87 ± 0.29 | 39.8 ± 0.0 | 100% (200/200) |
| 3 | 167.9 ± 4.4 | 5.96 ± 0.15 | 39.8 ± 0.0 | 100% (200/200) |
| 4 | 167.5 ± 3.8 | 5.97 ± 0.13 | 39.8 ± 0.0 | 100% (200/200) |

Summary: Full K-sweep completed successfully with ~6 tok/s throughput and 39.8% acceptance rate at K=4. All 800 runs (200 per K) completed without CUDA asserts. KV-cache reuse is now properly gated on full acceptance to maintain consistency. Production-ready pipeline validated.

### 2. Phase 4A – Llama 3.2 on Tesla T4 (Kaggle, 2025-11-18)

**Configuration**: Base `meta-llama/Llama-3.2-3B`, Draft `meta-llama/Llama-3.2-1B`, K=1, CUDA (Tesla T4), fp16

| Batch size | Max tokens | Throughput (tok/s) | Acceptance | Notes |
|------------|------------|-------------------|------------|-------|
| 1 | 64 | 8.45 ± 1.68 | 0.858 ± 0.179 | Best throughput, high acceptance |
| 2 | 64 | 4.98 ± 0.81 | 0.649 ± 0.072 | Lower throughput, acceptance drop |
| 4 | 32 | 4.68 ± 0.77 | 0.618 ± 0.129 | Needed reduced max_tokens |

**Baseline (Non-speculative, Llama 3.2 3B only)**: 16.99 ± 0.27 tok/s (64 tokens, BS=1, 10 samples)

**Takeaways**:

On T4, speculative decoding for Llama 3.2 prefers small batch sizes. Throughput peaks at batch size 1 (8.45 tok/s) with high acceptance (~86%), then drops at batch sizes 2 and 4. Bigger batches hurt acceptance, and the sequential verify loop prevents converting larger batch compute into more tokens per second. 

**Notably**, the non-speculative baseline (16.99 tok/s) is approximately 2× faster than speculative decoding at batch size 1, suggesting that on T4 hardware, the overhead of running both base and draft models plus verification outweighs the benefits for this model pairing. These results complement the earlier GPT2 experiments and show model and hardware dependence.

**Key Findings**:
- Best per-sequence throughput at batch size 1 (8.45 tok/s)
- Throughput and acceptance both degrade at larger batch sizes
- KV append was disabled (`SPECDEC_ENABLE_KV_APPEND=0`) in these runs
- All results are T4-only; A100/H100 runs are planned for future work

See [docs/progress.md](docs/progress.md) for detailed logs, tables, and analysis.

## Quick Start

### Local MPS Run

```bash
# Activate environment
source env/bin/activate

# Quick K-sweep test
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 --draft-model distilgpt2 \
  --max-tokens 32 --iterations 5 --device mps \
  --output-dir results_mps --no-plots

# With detailed metrics
SPECDEC_DETAILED_METRICS=1 python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 --draft-model distilgpt2 \
  --max-tokens 32 --iterations 5 --device mps \
  --output-dir results_detailed --no-plots
```

### CUDA Run (Kaggle/Cloud) - Phase 3D

```bash
# Set environment variables for Phase 3D optimization
export SPECDEC_AMP=1
export SPECDEC_DTYPE=float16
export SPECDEC_DETAILED_METRICS=1
export SPECDEC_DETERMINISTIC=1
export SPECDEC_PROFILE=1
export SPECDEC_CUDA_GRAPH=1
export SPECDEC_PARALLEL_STREAMS=1
export SPECDEC_SYNC_MODE=event

# Run comprehensive K-sweep
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 --draft-model distilgpt2 \
  --max-tokens 64 --iterations 100 --device cuda \
  --deterministic \
  --output-dir results_cuda --no-plots
```

### CUDA Run (Kaggle/Cloud) - Phase 4A Llama 3.2

```bash
# Set environment variables for Llama 3.2 T4 experiments
export SPECDEC_FORCE_PYTORCH_BACKEND=1
export SPECDEC_BATCH_SIZE=1
export SPECDEC_PARALLEL_STREAMS=1
export SPECDEC_DTYPE=float16
export SPECDEC_ENABLE_KV_APPEND=0
export SPECDEC_CUDA_GRAPH=0

# Run Llama 3.2 K=1 sweep
python scripts/comprehensive_k_sweep.py \
  --base-model "meta-llama/Llama-3.2-3B" \
  --draft-model "meta-llama/Llama-3.2-1B" \
  --max-tokens 64 \
  --iterations 1 \
  --device cuda \
  --max-k 1 \
  --output-dir ./docs/results/2025-11-18-llama32-t4-batch1 \
  --no-plots
```

### Basic Usage

```bash
# Smoke test
python scripts/dev/smoke_cuda.py

# Speculative decoding
python -m src.specdec.run_specdec --prompt "Hello world!" --max-tokens 32

# Local baseline
python -m src.server.local_baseline --prompt "Hello world!"
```

## Architecture Overview

### Core Components

- **Speculative Pipeline**: Draft-and-verify with advanced policies
- **Kernel System**: CUDA/Triton/PyTorch backends with registry
- **Metrics System**: Detailed profiling and memory tracking
- **Scheduler**: Multi-stream verification and batching
- **Optimization**: Mixed precision, memory management
- **KV Cache Integration**: Efficient token appending without recomputation

### Kernel Backends

The kernel system provides a priority-based registry for selecting optimized implementations:

- **CUDA**: Custom kernels for `verify_prefix` and `kv_append` (priority 100)
- **Triton**: Python-based GPU kernels with fallbacks (priority 50)
- **PyTorch**: Reference implementations for all devices (priority 10)

### KV Cache Integration

The speculative decoding pipeline includes efficient KV cache management to reduce redundant computation.

**What it does:**
- When the base model accepts tokens from the draft, their key-value states are cached
- Subsequent verifications reuse cached KV states instead of recomputing from scratch
- Reduces redundant computation and improves latency

**When it is active:**
- Enabled by default (`SPECDEC_ENABLE_KV_APPEND=1`)
- Works with HuggingFace causal language models that support `past_key_values`
- Automatically disabled for models that do not support KV caching

**Kernel backends:**
- **CUDA**: Optimized coalesced memory operations (priority 100)
- **Triton**: Python-based GPU kernels (priority 50)
- **PyTorch**: `torch.cat` fallback for all devices (priority 10)

**Usage:**
```bash
# Enable KV cache (default)
SPECDEC_ENABLE_KV_APPEND=1 python scripts/comprehensive_k_sweep.py ...

# Disable KV cache
SPECDEC_ENABLE_KV_APPEND=0 python scripts/comprehensive_k_sweep.py ...
```

**Metrics tracked:**
- `kv_appended_tokens_total`: Total tokens appended to cache
- `kv_append_time_ms`: Time spent in KV append operations
- `kv_append_enabled`: Whether KV caching is active
- `kv_append_backend`: Which kernel backend is used (cuda/triton/torch)

## Environment Flags

| Flag | Description | Default |
|------|-------------|---------|
| `SPECDEC_AMP=1/0` | Enable/disable mixed precision | Auto |
| `SPECDEC_DTYPE=float16/bfloat16/float32` | Override dtype | Auto |
| `SPECDEC_DETAILED_METRICS=1` | Enable detailed profiling | Off |
| `SPECDEC_ENABLE_KV_APPEND=1/0` | Enable KV cache appending | On |
| `SPECDEC_FORCE_PY=1` | Skip kernel compilation | Off |
| `SPECDEC_DETERMINISTIC=1` | Enable deterministic seeding | Off |
| `SPECDEC_PROFILE=1` | Enable structured event profiling | Off |
| `SPECDEC_CUDA_GRAPH=1` | Enable CUDA graph capture | Off |
| `SPECDEC_PARALLEL_STREAMS=1` | Enable CUDA stream overlap | On |
| `SPECDEC_SYNC_MODE=event/barrier` | Sync mode: event or barrier | event |
| `SPECDEC_DRY_RUN=1` | Run latency-only profiling | Off |
| `SPECDEC_DEBUG=1` | Enable debug logging (gated print statements) | Off |

## Testing

```bash
# Run all tests
pytest tests/ -k "not gpu" -q

# Run specific test categories
pytest tests/test_kernels_verify.py -v
pytest tests/test_metrics_profiler.py -v

# Run CI/CD pipeline
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/ tests/ scripts/
```

## Project Structure

```
llm-inference-lab/
├── src/
│   ├── kernels/          # Custom CUDA/Triton kernels
│   ├── metrics/          # Profiling and metrics
│   ├── scheduler/        # Multi-stream verification
│   ├── specdec/          # Speculative decoding pipeline
│   └── optimization/     # Mixed precision, memory
├── tests/                # Comprehensive test suite
├── scripts/              # Utility and benchmark scripts
├── configs/              # Configuration files
└── docs/                 # Documentation and progress
```

## Current Limitations

### Baseline Mode Bug

- **Status: FIXED (2025-11-18)**: The baseline mode bug has been resolved. `SpeculativePipeline._check_compatibility()` now properly handles `draft_lm=None` by checking for baseline mode and skipping draft model compatibility checks.

- **Baseline results available**: Non-speculative baseline runs are now functional. For Llama 3.2 3B on T4 (64 tokens, BS=1), baseline achieves 16.99 ± 0.27 tok/s, approximately 2× faster than speculative decoding (8.45 tok/s) at the same batch size.

### Hardware Limitations

- **All new Llama 3.2 results are T4-only**: All Phase 4A Llama 3.2 experiments were conducted on Tesla T4 (Kaggle). No A100/H100 runs have been performed yet for Llama 3.2; they remain part of future work.

### KV Cache Status

- **KV append disabled in new runs**: All new Llama 3.2 T4 runs used `SPECDEC_ENABLE_KV_APPEND=0`. These represent "pure" speculative results without KV cache reuse optimizations. CUDA KV cache tests for Llama 3.2 are future work.

For detailed findings and limitations, see [docs/progress.md](docs/progress.md).

## Next Steps

### Phase 3D: CUDA Validation (Complete)

CUDA validation completed successfully on Kaggle T4 hardware. Full K-sweep runs completed without CUDA asserts:

1. **T4 Validation** (Complete):
   - Run #1: 32 tokens × 100 iterations (Complete)
   - Run #2: 64 tokens × 200 samples (Complete)
   - All 800+ samples succeeded with 100% success rate
   - KV-cache consistency fixes validated
   
2. **A100 Validation** (Planned):
   - Single-session benchmarks with full Phase 3D features
   - Expected throughput: ~60 tok/s
   
3. **H100 Validation** (Planned):
   - High-end GPU validation
   - Expected throughput: ~100+ tok/s

**Phase 4A.1: Batch Processing Optimization** (Profiling Complete):
- Attention mask support (skip padding computation)
- Optimized padding operations (torch.nn.functional.pad)
- CPU-GPU memory optimization (reduced syncs and transfers)
- Debug logging gated with SPECDEC_DEBUG flag
- Batch vs single-prompt throughput profiling (completed - 5.76±0.15 tok/s individual, ~58 tok/s aggregate)
- Dynamic batching strategies (remaining)
- GPU compute optimization - Phase 1 (verify-loop kernel optimized, redundant syncs removed)

**Remaining Roadmap**:
- **Phase 4A.1**: GPU compute optimization - Phase 2 (profile utilization improvement, further optimizations)
- **Dynamic batching**: Group sequences by length to reduce padding waste
- **Larger GPU validation**: A100/H100 benchmarking for higher throughput targets
- **Tokenizer-aware optimizations**: Reduce CPU-bound overhead
- **Phase 4B**: Advanced quantization (INT8/INT4) for memory efficiency
- **Phase 4C**: Multi-GPU support for 7B+ models
- **Phase 4D**: Speculative tree decoding for improved acceptance rates

### Phase 4B: Advanced Quantization (Planned)

INT8/INT4 quantization techniques for memory-constrained deployments.

### Phase 4C: Layer/Model Parallelism (Planned)

Multi-GPU support for 7B+ models with tensor and pipeline parallelism.

### Phase 4D: Speculative Tree Decoding (Planned)

Tree-based acceptance strategies for improved draft token utilization.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the CI/CD pipeline
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

For research citation or academic reference, please cite this repository as:

```
LLM Inference Lab (2025). A transparent, research-grade inference runtime for reproducible 
speculative decoding experiments. https://github.com/GogoRit/llm-inference-lab
```

---

**Status**: Phase 4A In Progress – Llama 3.2 T4 Batch Scaling Experiments Complete (2025-11-18)
