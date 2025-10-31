# LLM Inference Lab

A comprehensive toolkit for optimizing Large Language Model inference with speculative decoding, custom CUDA kernels, and advanced performance techniques.

> **Strategy Note**: As of Phase 3C.5, LLM Inference Lab follows an **MPS-first optimization approach**.
> All new features are validated locally on Apple Silicon (MPS) before limited CUDA (Kaggle/A100) benchmarking.
> This ensures efficient use of GPU credits and rapid iteration on Mac hardware.

## What's New (Phase 3D Complete)

**Latest Achievements**:
- **GPU Optimization**: CUDA stream overlap, graph capture, structured profiling
- **Deterministic Reproducibility**: Centralized seeding utility with environment flags
- **Structured Metrics**: Per-step GPU event timers with JSON output
- **Dry-Run Profiling**: Latency-only profiling without model compute
- **Enhanced Memory Tracking**: CUDA and MPS memory stats integration
- **MPS Validation Complete**: All Phase 3D features validated on Apple Silicon

**Phase 3D Features**:
- **Deterministic Seeding**: `SPECDEC_DETERMINISTIC=1` for reproducible benchmarks
- **Structured Profiler**: `SPECDEC_PROFILE=1` for per-step event timing
- **CUDA Graph Capture**: `SPECDEC_CUDA_GRAPH=1` for steady decoding optimization
- **Stream Overlap**: `SPECDEC_PARALLEL_STREAMS=1` with event-based synchronization
- **Dry-Run Mode**: `SPECDEC_DRY_RUN=1` for latency profiling without model load

**Performance Results** (GPT2-124M + DistilGPT2, MPS, Phase 3D):
- **Throughput**: ~11.5 tok/s average across K=1-4
- **Validation Status**: 80 runs, 100% success, all Phase 3D features functional
- **Unit Tests**: 14 tests passing (2 skipped on non-CUDA)
- **Ready for CUDA**: All features validated on MPS before GPU runs

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

### CUDA Run (Kaggle/Cloud)
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
  --max-tokens 32 --iterations 5 --device cuda \
  --output-dir results_cuda --no-plots
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
| **3D** | In Progress | GPU optimization | Stream overlap, graph capture, structured profiling (CUDA validation pending) |
| **4A** | Planned | Batch-level processing | Multi-prompt batch processing (after Phase 3D CUDA validation) |
| **4C** | Future | Layer/model parallelism | Multi-GPU support for 7B+ models (after Phase 3D CUDA validation) |
| **4D** | Future | Speculative tree decoding | Tree-based acceptance strategies (after Phase 3D CUDA validation) |

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

## Performance Results

### Latest K-Sweep Results (Phase 3C.4)
**MPS Performance** (gpt2 + distilgpt2, 32 tokens, 5 iterations):

| K Value | Throughput (tok/s) | Acceptance Rate | Latency (ms) | Memory (MB) |
|---------|-------------------|-----------------|--------------|-------------|
| K=1 | 9.12±0.45 | 18.2±8.1% | 3,510±173 | 275±12 |
| K=2 | 9.23±0.38 | 16.8±7.2% | 3,470±143 | 280±15 |
| K=3 | 9.15±0.42 | 17.1±6.8% | 3,500±161 | 285±18 |
| **K=4** | **9.55±0.51** | **16.9±7.5%** | **3,350±179** | **290±20** |

**Key Achievements**:
- **Peak Performance**: K=4 achieves 9.55 tok/s
- **Stable Across K**: Consistent ~9 tok/s performance
- **100% Success Rate**: All 200 test runs completed successfully
- **Memory Efficient**: ~275MB peak usage maintained

### Technical Improvements Validated
- **Mixed Precision**: AMP enabled on MPS with float16
- **Memory-Safe Loading**: Device-aware model loading
- **Kernel Registry**: Priority-based backend selection
- **Detailed Metrics**: Optional profiling system
- **CUDA Graph Capture**: Performance optimization ready
- **CI/CD Pipeline**: All checks passing (Black, isort, flake8, mypy, pytest)

### CUDA (T4) Performance – 2025-10-30

**Run #1**: 32 tokens × 100 iterations | [Results folder](docs/results/2025-10-30-T4/2025-10-30-T4_k32_i100_fp16/)

| K | Latency (ms mean ± std) | Throughput (tok/s mean ± std) | Acceptance (%) mean ± std |
|---|-------------------------|--------------------------------|----------------------------|
| 1 | 3743.07 ± 1031.39 | 17.24 ± 4.96 | 21.38 ± 12.43 |
| 2 | 3904.52 ± 908.77 | 17.66 ± 5.67 | 22.72 ± 14.20 |
| 3 | 3916.34 ± 976.50 | 17.53 ± 5.96 | 22.28 ± 15.06 |
| 4 | 3870.67 ± 1012.93 | 17.22 ± 4.96 | 21.88 ± 12.71 |

Summary: ~17.4 tok/s average across K=1–4 (≈1.8–2.0× vs MPS), best acceptance ≈22.7% at K=2, 100% success, no OOM.

**Run #2**: 64 tokens × 100 iterations (deterministic) | **In Progress**

Expected improvements: longer sequences amortize overhead; deterministic mode ensures reproducibility. Results to be added upon completion. A100/H100 validation to follow.

## Architecture

### Core Components
- **Speculative Pipeline**: Draft-and-verify with advanced policies
- **Kernel System**: CUDA/Triton/PyTorch backends with registry
- **Metrics System**: Detailed profiling and memory tracking
- **Scheduler**: Multi-stream verification and batching
- **Optimization**: Mixed precision, memory management
- **KV Cache Integration**: Efficient token appending without recomputation

### Kernel Backends
- **CUDA**: Custom kernels for `verify_prefix` and `kv_append`
- **Triton**: Python-based GPU kernels with fallbacks
- **PyTorch**: Reference implementations for all devices

### KV Cache Integration

The speculative decoding pipeline now includes efficient KV cache management:

**What it does:**
- When the base model accepts tokens from the draft, their key-value states are cached
- Subsequent verifications reuse cached KV states instead of recomputing from scratch
- Reduces redundant computation and improves latency

**When it's active:**
- Enabled by default (`SPECDEC_ENABLE_KV_APPEND=1`)
- Works with HuggingFace causal language models that support `past_key_values`
- Automatically disabled for models that don't support KV caching

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

## Next Steps

**Phase 3D - CUDA Validation** (MPS-first strategy):
1. **MPS Validation** (Local):
   - Validate all Phase 3C features on Apple Silicon
   - Establish performance baselines (~9 tok/s)
   - Verify KV cache integration with on/off comparison
2. **CUDA Validation** (Limited GPU credits):
   - Single-session T4/A100 benchmarks
   - Controlled GPU credit usage
   - Compare against MPS baselines

**Future Phases**:
- Multi-GPU scaling and distributed inference
- Advanced quantization techniques
- Cloud deployment and monitoring

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the CI/CD pipeline
5. Submit a pull request

---

**Status**: Phase 3C Complete - Ready for CUDA validation in Phase 3D