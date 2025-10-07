# LLM Inference Lab

A comprehensive toolkit for optimizing and benchmarking Large Language Model inference performance with advanced techniques including speculative decoding, custom CUDA kernels, and intelligent batching.

## Overview

This project provides a comprehensive toolkit for LLM inference optimization research:

**Current Capabilities**:
- **Local Baseline Runner**: HuggingFace OPT-125M with CPU/MPS support
- **HTTP Client**: OpenAI-compatible client for vLLM server integration
- **Speculative Decoding**: Draft-and-verify pipeline with advanced strategies
- **Advanced Draft Modes**: Medusa-lite and EAGLE-lite implementations
- **Acceptance Policies**: Longest prefix, confidence threshold, top-k agreement, typical acceptance
- **Adaptive Controllers**: Fixed and adaptive K controllers for draft token management
- **Performance Optimization**: Mixed precision, gradient checkpointing, and tokenizer caching
- **Comprehensive Profiling**: PyTorch Profiler integration with memory tracking
- **Dual-Mode Benchmarking**: Statistical performance analysis (local vs HTTP vs specdec)
- **Professional Logging**: Structured logging and configuration management
- **Code Quality**: Comprehensive linting and type checking

**Future Research Areas**:
- Custom CUDA kernels and quantization
- Intelligent request batching and scheduling
- Multi-GPU scaling and distributed inference
- Advanced model optimization techniques

## What's New (2025-10-06)

**Phase 3C.4 Complete - MPS Final Validation**:
- **Kernel Registry**: Safe backend selection with priority-based fallbacks
- **Detailed Metrics**: Optional performance profiling with `--metrics-detailed` flag
- **CUDA Graph Capture**: Optional graph capture with `--cuda-graph` flag (CUDA only)
- **Memory Profiling**: Enhanced memory tracking for CUDA and MPS
- **Documentation**: Updated with latest MPS performance results
- **CI Ready**: All tests passing, ready for CUDA validation

**Environment Flags**:
- `SPECDEC_AMP=1/0`: Enable/disable mixed precision
- `SPECDEC_DTYPE=float16/bfloat16/float32`: Override dtype
- `SPECDEC_DETAILED_METRICS=1`: Enable detailed profiling
- `SPECDEC_CUDA_GRAPH=1`: Enable CUDA graph capture
- `SPECDEC_FORCE_PY=1`: Skip kernel compilation, use PyTorch fallbacks

## Latest Results: Comprehensive K-Sweep Analysis

**Phase 3A Complete** - Local Performance Optimization with Statistical Validation:

| K Value | Latency (ms) | Throughput (tok/s) | Acceptance Rate | Status |
|---------|--------------|-------------------|-----------------|---------|
| K=1 | 5,554±891 | 5.90±1.12 | 15.6±8.8% | Baseline |
| **K=2** | **5,141±776** | **6.35±1.04** | **15.6±7.5%** | **⭐ OPTIMAL** |
| K=3 | 5,311±884 | 6.23±1.38 | 16.5±9.5% | Good |
| K=4 | 5,392±1,092 | 5.95±1.24 | 16.6±10.2% | Diminishing returns |

**Key Achievements:**
- **100 samples per K value** (10 iterations × 10 prompts)
- **1.23x speedup** with K=2 vs baseline
- **Statistical robustness** with proper error bars
- **Research-grade documentation** with visualizations
- **Ready for Phase 3B** (GPU scaling and kernel optimizations)

## Setup

### Prerequisites

**Local Development**:
- Python 3.8+
- Mac with Apple Silicon (for MPS acceleration) or CPU-only setup
- 4GB+ RAM for model loading

**Future Cloud Deployment**:
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with compute capability 7.0+ (A100/H100 recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/GogoRit/llm-inference-lab.git
cd llm-inference-lab

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r env/requirements.txt

# Install in development mode
pip install -e .
```

### Environment Setup

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Usage

### Quick Start

**Local MPS Quick Run**:
```bash
# Activate environment
source env/bin/activate

# Run comprehensive K-sweep on MPS
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 --draft-model distilgpt2 \
  --max-tokens 32 --iterations 5 --device mps \
  --output-dir results_mps_quick --no-plots

# Run with detailed metrics
SPECDEC_DETAILED_METRICS=1 python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 --draft-model distilgpt2 \
  --max-tokens 32 --iterations 5 --device mps \
  --output-dir results_mps_detailed --no-plots
```

**CUDA Quick Run (Kaggle)**:
```bash
# Set environment variables
export SPECDEC_AMP=1
export SPECDEC_DTYPE=float16
export SPECDEC_DETAILED_METRICS=1
export SPECDEC_CUDA_GRAPH=1

# Run comprehensive K-sweep on CUDA
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 --draft-model distilgpt2 \
  --max-tokens 32 --iterations 5 --device cuda \
  --output-dir results_cuda_quick --no-plots

# Artifacts will be saved to results_cuda_quick/
```

**Basic Usage**:
```bash
# Local baseline inference
python -m src.server.local_baseline --prompt "Hello, world!"

# Speculative decoding
python -m src.specdec.run_specdec --prompt "Explain KV cache simply." --max-tokens 64 --verbose

# Smoke test
python scripts/dev/smoke_cuda.py
python -m src.specdec.run_specdec --config configs/specdec.yaml --prompt "Test" --seed 42

# Performance optimization and profiling
python -m src.specdec.run_specdec --prompt "Test" --profile --profile-dir traces/
python -m src.benchmarks.run_bench --mode specdec --profile --iterations 10
python -m src.specdec.run_specdec --prompt "Test" --disable-optimization  # Baseline comparison
```

### Configuration

The system supports YAML configuration files for easy parameter management:

```yaml
# configs/baseline.yaml
model: "facebook/opt-125m"
max_new_tokens: 48
temperature: 0.7
do_sample: true
device_priority: ["mps", "cuda", "cpu"]
```

### Speculative Decoding

Speculative decoding uses a small draft model to propose multiple tokens, then verifies them with the base model for faster inference. The implementation features advanced acceptance policies, adaptive K controllers, and comprehensive instrumentation for research and production use cases.

**FakeLM Mode (Testing & Development)**:
```bash
# Default mode - memory safe, deterministic testing
python -m src.specdec.run_specdec --prompt "Explain KV cache simply." --max-tokens 64 --impl fake

# With custom parameters and policies
python -m src.specdec.run_specdec --prompt "Test" --K 8 --policy conf_threshold --tau 0.7 --impl fake

# Using adaptive K controller
python -m src.specdec.run_specdec --prompt "Test" --adaptive-K --min-k 1 --max-k 8 --impl fake

# Using configuration file
python -m src.specdec.run_specdec --config configs/specdec.yaml --prompt "Test" --impl fake
```

**Hugging Face Mode (Real Inference)**:
```bash
# Real models with tiny configurations for memory safety
python -m src.specdec.run_specdec --prompt "Test real inference" --max-tokens 20 --impl hf --force-device cpu

# With different base and draft models
python -m src.specdec.run_specdec --prompt "Test" --base-model sshleifer/tiny-gpt2 --draft-model sshleifer/tiny-gpt2 --impl hf

# Using HF-specific configuration
python -m src.specdec.run_specdec --config configs/specdec_hf.yaml --prompt "Test" --impl hf
```

**Advanced Features**:

**Acceptance Policies**:
- `longest_prefix` (default): Accept longest matching prefix
- `conf_threshold`: Accept tokens above confidence threshold (--tau)
- `topk_agree`: Accept tokens in base model's top-k predictions (--k)
- `typical`: Accept tokens above typical probability (--p)

**K Controllers**:
- `fixed`: Fixed K value (--K)
- `adaptive`: Adaptive K based on acceptance rates (--adaptive-K, --min-k, --max-k, --target-acceptance)

**Architecture**:
1. **Draft Model**: Small model proposes up to K tokens (controlled by K controller)
2. **Verification**: Base model verifies proposals with selected acceptance policy
3. **Acceptance**: Policy determines how many proposed tokens to accept
4. **Fallback**: If no tokens accepted, generate one token with base model
5. **Repeat**: Continue until max_tokens or stop condition

**Implementation Modes**:
- **FakeLM Mode** (`--impl fake`): Deterministic testing without memory issues, 100% acceptance rate
- **HF Mode** (`--impl hf`): Real models (sshleifer/tiny-gpt2) with shared tokenizers and memory safety

### Performance Optimization

The system includes comprehensive performance optimization tools for CPU/MPS development:

**Mixed Precision and Memory Optimization**:
- Automatic device-aware dtype selection (float16 on MPS, bfloat16 on CUDA)
- Gradient checkpointing for memory efficiency
- Memory efficient attention where supported
- Tokenizer caching and batched processing

**Comprehensive Profiling**:
- PyTorch Profiler integration with Chrome trace export
- Memory tracking with RSS monitoring
- Per-operation timing with statistical analysis
- Device-specific performance metrics

**Usage Examples**:
```bash
# Enable profiling for performance analysis
python -m src.specdec.run_specdec --prompt "Test" --profile --profile-dir traces/

# Benchmark with profiling enabled
python -m src.benchmarks.run_bench --mode specdec --profile --iterations 10

# Compare optimized vs baseline performance
python -m src.specdec.run_specdec --prompt "Test" --profile
python -m src.specdec.run_specdec --prompt "Test" --disable-optimization --profile
```

### Benchmarking Different Endpoints

The benchmark harness supports local baseline, HTTP server, and speculative decoding:

```bash
# Compare local vs HTTP performance
python -m src.benchmarks.run_bench --prompt "Compare performance" --mode local --iterations 5
python -m src.benchmarks.run_bench --prompt "Compare performance" --mode http --host 127.0.0.1 --port 8000 --iterations 5

# Benchmark speculative decoding
python -m src.benchmarks.run_bench --mode specdec --prompt "Why speculative decoding helps?" --iterations 3

# Compare speculative decoding vs baseline
python -m src.benchmarks.run_bench --mode specdec --compare-baseline --prompt "Test comparison" --iterations 3

# Benchmark with perplexity evaluation (HF mode only)
python -m src.benchmarks.run_bench --mode specdec --eval-perplexity --prompt "Quality test" --iterations 3

# Use different configuration files
python -m src.benchmarks.run_bench --config configs/baseline.yaml --prompt "Test" --mode local
python -m src.benchmarks.run_bench --config configs/vllm.yaml --prompt "Test" --mode http
python -m src.benchmarks.run_bench --config configs/specdec.yaml --prompt "Test" --mode specdec
```

### Programmatic Usage

```python
from src.server.local_baseline import LocalBaselineRunner
from src.server.ping_vllm import VLLMPingClient
from src.benchmarks.run_bench import BenchmarkRunner
from src.specdec.pipeline import SpeculativePipeline

# Local baseline inference
runner = LocalBaselineRunner(config_path="configs/baseline.yaml")
result = runner.run("Hello, world!", max_new_tokens=50)
print(f"Generated: {result['text']}")
print(f"Latency: {result['latency_ms']:.2f} ms")

# Speculative decoding inference (vanilla mode)
pipeline = SpeculativePipeline(config_path="configs/specdec.yaml")
result = pipeline.generate("Explain KV cache simply.", max_tokens=64)
print(f"Generated: {result['text']}")
print(f"Latency: {result['latency_ms']:.2f} ms")
print(f"Acceptance Rate: {result['acceptance_rate']:.3f}")

# Advanced draft modes
# Medusa-lite: Multiple prediction heads
pipeline_medusa = SpeculativePipeline(draft_mode="medusa")
result = pipeline_medusa.generate("Explain KV cache simply.", max_tokens=64)

# EAGLE-lite: Hidden state extrapolation
pipeline_eagle = SpeculativePipeline(draft_mode="eagle")
result = pipeline_eagle.generate("Explain KV cache simply.", max_tokens=64)

# HTTP server inference
client = VLLMPingClient(host="127.0.0.1", port=8000, config_path="configs/vllm.yaml")
if client.ping():
    result = client.generate("Hello, world!", max_tokens=50)
    if result["success"]:
        print(f"Generated: {result['generated_text']}")
        print(f"Latency: {result['latency_ms']:.2f} ms")

# Performance benchmarking
benchmark = BenchmarkRunner(config_path="configs/baseline.yaml", mode="local")
stats = benchmark.run_benchmark("Hello, world!", iterations=10)
benchmark.print_summary(stats)

# Speculative decoding benchmarking with comparison
specdec_benchmark = BenchmarkRunner(config_path="configs/specdec.yaml", mode="specdec", compare_baseline=True)
stats = specdec_benchmark.run_benchmark("Why speculative decoding helps?", iterations=5)
specdec_benchmark.print_summary(stats)
```

### Performance Metrics

**Current Baseline Performance** (Apple Silicon MPS):
- **Model**: facebook/opt-125m (125M parameters)
- **Latency**: ~1.15s per 48 tokens
- **Throughput**: ~41 tokens/second
- **Memory Usage**: ~500MB model loading
- **Device**: MPS acceleration on Apple Silicon

**Speculative Decoding Performance**:

| Implementation | Latency | Throughput | Memory | Use Case |
|---------------|---------|------------|--------|----------|
| **FakeLM Mode** | 0.42ms (4 tokens) | 9,430 tokens/sec | Minimal | Testing, CI/CD |
| **HF Mode** | 44.62ms (4 tokens) | 89.65 tokens/sec | ~500MB | Real inference |

**Advanced Draft Modes** (Real Performance):

**MPS Performance** (facebook/opt-125m + distilgpt2, 10 tokens):
| Draft Mode | Latency (ms) | Tokens/sec | Acceptance Rate | Speedup | Description |
|------------|-------------|------------|-----------------|---------|-------------|
| **Vanilla** | 7,315 | 1.37 | 0.00 | 1.0x | Traditional draft model approach |
| **Medusa-lite** | 14,088 | 0.71 | 0.00 | 0.52x | Multiple prediction heads |
| **EAGLE-lite** | 2,837 | 3.53 | 0.05 | 2.58x | Hidden state extrapolation |

**Tiny CPU Performance** (sshleifer/tiny-gpt2, 24 tokens):
| Draft Mode | Latency (ms) | Tokens/sec | Acceptance Rate | Speedup | Description |
|------------|-------------|------------|-----------------|---------|-------------|
| **Vanilla** | 101.7 | 236.0 | 1.00 | 1.0x | Traditional draft model approach |

**Implementation Details**:
- **FakeLM Mode**: Deterministic testing without memory issues, 100% acceptance rate
- **HF Mode**: Real models (sshleifer/tiny-gpt2), CPU-optimized, shared tokenizers
- **Architecture**: Dependency injection with dual-mode support
- **Memory Safety**: 500MB limits, MPS cleanup, dtype guards (float16 on MPS, float32 on CPU)

**Performance Optimization Results** (Comprehensive K-Sweep Analysis):

**Comprehensive K-Sweep (100 samples per K value)**:
- **K=1**: 5,554±891ms latency, 5.90±1.12 tok/s, 15.6±8.8% acceptance
- **K=2**: 5,141±776ms latency, 6.35±1.04 tok/s, 15.6±7.5% acceptance ⭐ **OPTIMAL**
- **K=3**: 5,311±884ms latency, 6.23±1.38 tok/s, 16.5±9.5% acceptance
- **K=4**: 5,392±1,092ms latency, 5.95±1.24 tok/s, 16.6±10.2% acceptance

**Fake Models (Testing/Development)**:
- **Throughput**: 896.45 ± 45.97 tokens/sec (3 iterations average)
- **Latency**: 53.64 ± 2.82 ms for 48 tokens
- **Memory Usage**: ~279MB peak memory with profiling
- **Acceptance Rate**: 100% (perfect speculative decoding)

**Real HF Models (Production - Compatible Architecture)**:
- **Baseline (No Opt)**: 5.16±0.92 tokens/sec, 520MB peak memory, 0% acceptance rate
- **Optimized (Mixed Prec)**: 5.82±0.94 tokens/sec, 500MB peak memory, 0% acceptance rate
- **K=2 (Optimal)**: 6.35±1.04 tokens/sec, 540MB peak memory, 15.6±7.5% acceptance rate
- **Optimization Impact**: 1.13x speedup, 4% memory reduction, 1.23x speedup with K=2
- **Development vs Production**: 255x faster with fake models for testing

**Benchmarking Capabilities**:
- Statistical analysis (mean, median, std deviation)
- Multi-mode comparison (local vs HTTP vs specdec)
- Baseline comparison with speedup metrics
- Comprehensive profiling with memory tracking
- Configurable warmup and iteration counts
- Professional logging and error handling

### Phase 3A: Local Performance Optimization (COMPLETED)

**Comprehensive K-Sweep Analysis:**
- **Test Suite**: 10 diverse prompts across different domains
- **Sample Size**: 100 samples per K value (10 iterations × 10 prompts)
- **Statistical Robustness**: All metrics include mean±std for significance
- **Optimal Configuration**: K=2 provides best throughput (6.35±1.04 tok/s)
- **Acceptance Rate**: Consistent ~15.6% across all K values
- **Memory Scaling**: Linear increase with K (520MB → 580MB for K=4)

**Key Achievements:**
- **Performance Optimization**: Mixed precision (fp16), tokenizer caching, gradient checkpointing
- **Comprehensive Profiling**: PyTorch Profiler integration with memory tracking
- **Statistical Validation**: 10-iteration K-sweep with 100 samples per K value
- **Research Foundation**: Clean architecture ready for Phase 3B (GPU scaling)
- **Documentation**: Professional-grade results with visualizations and interpretation

### Development Notes

**Local Development**: This project is designed for local development on CPU/MPS (Apple Silicon) systems. Cloud GPUs (A100/H100) will be used later for final benchmarking and paper-quality results.

## Project Structure

```
llm-inference-lab/
├── src/                    # Main source code
│   ├── server/            # API & serving infrastructure
│   ├── scheduler/         # Request batching & routing
│   ├── specdec/          # Speculative decoding implementation
│   ├── kernels/          # Custom CUDA kernels
│   └── benchmarks/       # Performance testing tools
├── tests/                 # Test suite
├── configs/               # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── env/                   # Environment configuration
```

## Development Roadmap

| Phase | Feature | Status | Description |
|-------|---------|--------|-------------|
| 1A | Baseline runner | Complete | HuggingFace OPT-125M with CPU/MPS support |
| 1B | Benchmark client | Complete | HTTP client + dual-mode benchmarking (local/HTTP) |
| 1C | CI/CD sanity | Complete | Automated testing and code quality checks |
| 2A | Speculative decoding | Complete | CPU/MPS draft-and-verify pipeline with benchmarking |
| 2B | Advanced specdec | Complete | Acceptance policies, adaptive K controllers, instrumentation |
| 2C | Advanced specdec | Complete | Medusa/EAGLE techniques and optimizations |
| 3A | Local optimization | Complete | Mixed precision, profiling, K-sweep analysis (CPU/MPS) |
| 3B | GPU scaling | Next | CUDA optimization, kernel improvements, A100/H100 testing |
| 4 | Multi-GPU scaling | Future | Distributed inference and load balancing |
| 5 | Cloud deployment | Future | Production deployment and monitoring |

### Current Focus
- **Phase 3A**: Complete - Local performance optimization with comprehensive K-sweep analysis
- **Phase 3B**: Next - GPU scaling and kernel optimizations for CUDA/A100/H100
- **Research Ready**: Clean architecture with statistical validation and professional documentation

### Future Phases
- **Phase 3**: Quantization techniques and memory optimization
- **Phase 4**: Multi-GPU scaling and distributed inference
- **Phase 5**: Cloud deployment for final benchmarking

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of PyTorch and Transformers
- Inspired by vLLM and TensorRT-LLM
- CUDA kernel optimizations based on FlashAttention

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.
