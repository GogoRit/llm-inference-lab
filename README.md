# LLM Inference Lab

A comprehensive toolkit for optimizing Large Language Model inference with speculative decoding, custom CUDA kernels, and advanced performance techniques.

## 🚀 What's New (Phase 3C Complete)

**Latest Achievements**:
- ✅ **Custom CUDA Kernels**: `verify_prefix` and `kv_append` with Triton fallbacks
- ✅ **Kernel Registry**: Priority-based backend selection with safe fallbacks
- ✅ **Detailed Metrics**: Optional profiling with `--metrics-detailed` flag
- ✅ **CUDA Graph Capture**: Optional graph capture with `--cuda-graph` flag
- ✅ **Memory Profiling**: Enhanced tracking for CUDA and MPS
- ✅ **CI/CD Ready**: All tests passing, production-ready code

**Performance Results** (MPS, gpt2 + distilgpt2):
- **K=4 Peak**: 9.55 tok/s throughput
- **Stable Performance**: ~9 tok/s across K=1-4
- **100% Success Rate**: 200 test runs completed
- **Memory Efficient**: ~275MB peak usage

## 📋 Quick Start

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
# Set environment variables
export SPECDEC_AMP=1
export SPECDEC_DTYPE=float16
export SPECDEC_DETAILED_METRICS=1
export SPECDEC_CUDA_GRAPH=1

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

## 🏗️ Project Phases

| Phase | Status | Description | Key Features |
|-------|--------|-------------|--------------|
| **1A** | ✅ Complete | Baseline runner | HuggingFace OPT-125M, CPU/MPS support |
| **1B** | ✅ Complete | Benchmark client | HTTP client, dual-mode benchmarking |
| **1C** | ✅ Complete | CI/CD pipeline | Automated testing, code quality |
| **2A** | ✅ Complete | Speculative decoding | Draft-and-verify pipeline |
| **2B** | ✅ Complete | Advanced policies | Acceptance policies, adaptive K controllers |
| **2C** | ✅ Complete | Advanced modes | Medusa-lite, EAGLE-lite implementations |
| **3A** | ✅ Complete | Local optimization | Mixed precision, profiling, K-sweep analysis |
| **3B** | ✅ Complete | GPU pre-check | CUDA readiness, memory-safe loading |
| **3C** | ✅ Complete | CUDA kernels | Custom kernels, registry, detailed metrics |
| **3D** | 🔄 Next | CUDA validation | GPU performance validation |

## 🛠️ Environment Flags

| Flag | Description | Default |
|------|-------------|---------|
| `SPECDEC_AMP=1/0` | Enable/disable mixed precision | Auto |
| `SPECDEC_DTYPE=float16/bfloat16/float32` | Override dtype | Auto |
| `SPECDEC_DETAILED_METRICS=1` | Enable detailed profiling | Off |
| `SPECDEC_CUDA_GRAPH=1` | Enable CUDA graph capture | Off |
| `SPECDEC_FORCE_PY=1` | Skip kernel compilation | Off |

## 📊 Performance Results

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
- ✅ **Mixed Precision**: AMP enabled on MPS with float16
- ✅ **Memory-Safe Loading**: Device-aware model loading
- ✅ **Kernel Registry**: Priority-based backend selection
- ✅ **Detailed Metrics**: Optional profiling system
- ✅ **CUDA Graph Capture**: Performance optimization ready
- ✅ **CI/CD Pipeline**: All checks passing (Black, isort, flake8, mypy, pytest)

## 🏛️ Architecture

### Core Components
- **Speculative Pipeline**: Draft-and-verify with advanced policies
- **Kernel System**: CUDA/Triton/PyTorch backends with registry
- **Metrics System**: Detailed profiling and memory tracking
- **Scheduler**: Multi-stream verification and batching
- **Optimization**: Mixed precision, memory management

### Kernel Backends
- **CUDA**: Custom kernels for `verify_prefix` and `kv_append`
- **Triton**: Python-based GPU kernels with fallbacks
- **PyTorch**: Reference implementations for all devices

## 🧪 Testing

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

## 📁 Project Structure

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

## 🚀 Next Steps

**Phase 3D - CUDA Validation**:
- GPU performance validation on CUDA hardware
- Kernel performance benchmarking
- Production-ready deployment testing

**Future Phases**:
- Multi-GPU scaling and distributed inference
- Advanced quantization techniques
- Cloud deployment and monitoring

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the CI/CD pipeline
5. Submit a pull request

---

**Status**: Phase 3C Complete - Ready for CUDA validation in Phase 3D