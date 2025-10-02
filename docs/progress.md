# LLM Inference Lab - Development Progress

**Project**: LLM Inference Lab  
**Repository**: https://github.com/GogoRit/llm-inference-lab  
**Objective**: Build a comprehensive toolkit for optimizing and benchmarking Large Language Model inference performance

---

## Project Overview

This document tracks the systematic development of an LLM inference optimization framework, documenting methodology, results, and insights that may contribute to research publications. The project follows a phase-based approach to ensure robust, incremental development.

### Key Research Areas
- **Speculative Decoding**: Implementation and optimization of draft model techniques
- **Custom CUDA Kernels**: High-performance attention and matrix operations
- **Intelligent Batching**: Dynamic request scheduling and load balancing
- **Performance Benchmarking**: Comprehensive latency and throughput analysis
- **Multi-GPU Scaling**: Distributed inference optimization

---

## Phase 1: Baseline Infrastructure

### Phase 1A: Local Baseline Runner + CPU/MPS Smoke Tests (COMPLETED)

**Objective**: Establish a working baseline with Hugging Face Transformers that can run on Mac (CPU/MPS) for local development and validation.

#### Implementation Details

**Local Baseline Runner** (`src/server/local_baseline.py`)
- **Model**: facebook/opt-125m (125M parameters)
- **Device Selection**: Auto-detects MPS (Apple Silicon) > CUDA > CPU
- **Architecture**: Hugging Face Transformers with PyTorch backend
- **Interface**: CLI with argparse for prompt and token count control

**Key Technical Decisions**:
1. **Model Choice**: OPT-125M selected for fast loading and reasonable performance
2. **Device Priority**: MPS > CUDA > CPU for optimal Mac compatibility
3. **Precision**: Float32 for maximum compatibility across devices
4. **Generation Parameters**: Temperature=0.7, do_sample=True for balanced creativity

#### Results and Performance

**Test Environment**:
- **Hardware**: Mac with Apple Silicon (MPS acceleration)
- **Software**: Python 3.9.6, PyTorch 2.8.0, Transformers 4.56.2
- **Virtual Environment**: Isolated development environment

**Performance Metrics**:
```
Device: mps
Latency: 6084.37 ms
Prompt: "Hello, world!"
Generated tokens: 20
Tokens per second: ~3.3
```

**Analysis**:
- MPS acceleration successfully utilized
- Initial latency indicates room for optimization
- Model loading time not separated from inference time (future improvement)

#### Testing Infrastructure

**Smoke Tests** (`tests/test_cpu_smoke.py`):
- PyTorch import and basic tensor operations
- Device availability detection (CPU/MPS/CUDA)
- Transformers library functionality
- LocalBaselineRunner import and initialization
- MPS-specific operations (marked with @pytest.mark.gpu)
- Model loading simulation with actual HF model

**Test Results**:
```
12 passed, 2 deselected, 3 warnings in 2.91s
```

**CI/CD Integration**:
- Tests properly filtered with `pytest -k "not gpu"`
- GPU tests marked and excluded from CPU-only CI runs
- Virtual environment setup for reproducible testing

#### Lessons Learned

1. **Device Compatibility**: MPS provides significant acceleration over CPU on Apple Silicon
2. **Model Loading**: Initial model download and loading dominates first-run latency
3. **Testing Strategy**: Comprehensive smoke tests catch integration issues early
4. **CI/CD**: Proper test marking prevents CI failures on CPU-only environments

#### Code Quality Metrics

- **Lines of Code**: 149 lines (local_baseline.py)
- **Test Coverage**: 8 comprehensive smoke tests
- **Documentation**: Comprehensive docstrings and CLI help
- **Error Handling**: Graceful device fallback and error reporting

---

## Phase 1B: Minimal Benchmark Client (Planned)

**Objective**: Create HTTP client and micro-benchmark tools for future vLLM server integration.

#### Planned Implementation

**Ping Client** (`src/server/ping_vllm.py`):
- OpenAI-compatible API client
- Configurable host/port/model parameters
- Simple text response extraction

**Benchmark Harness** (`src/benchmarks/run_bench.py`):
- Sequential request testing
- Latency measurement and statistics
- Lightweight dependencies (requests, time, statistics)

#### Expected Outcomes
- HTTP client ready for vLLM integration
- Baseline benchmarking methodology established
- Performance measurement framework in place

---

## Phase 1C: GitHub Actions Sanity (Planned)

**Objective**: Ensure CI runs CPU tests only and properly excludes GPU tests.

#### Planned Implementation
- Verify CI pipeline runs `pytest -k "not gpu" -q`
- Add GPU-marked tests with proper exclusions
- Validate CI/CD pipeline reliability

---

## Phase 1D: Documentation + Planning (Planned)

**Objective**: Professional repository setup with comprehensive documentation and roadmap.

#### Planned Implementation
- Update README with Phase 1 completion status
- Create GitHub issues for future development phases
- Establish project roadmap and milestones

---

## Research Contributions and Future Work

### Potential Research Directions

1. **Speculative Decoding Optimization**
   - Implementation of Medusa and EAGLE techniques
   - Performance comparison with baseline methods
   - Custom draft model architectures

2. **Custom CUDA Kernel Development**
   - FlashAttention variants for specific hardware
   - Memory-efficient attention implementations
   - Kernel fusion optimizations

3. **Dynamic Batching Algorithms**
   - Request scheduling optimization
   - Load balancing across multiple GPUs
   - Adaptive batch size selection

4. **Quantization Techniques**
   - BitsAndBytes 4-bit and 8-bit quantization
   - Performance vs. accuracy trade-offs
   - Quantization-aware training integration

### Methodology Documentation

This project follows a systematic approach to LLM inference optimization:

1. **Baseline Establishment**: Start with standard implementations
2. **Incremental Optimization**: Add optimizations one at a time
3. **Comprehensive Benchmarking**: Measure each improvement
4. **Reproducible Results**: Document all experimental conditions
5. **Open Source Development**: Share methodology and results

### Data Collection Strategy

- **Performance Metrics**: Latency, throughput, memory usage
- **Hardware Utilization**: GPU/CPU usage, memory consumption
- **Quality Metrics**: Perplexity, BLEU scores (where applicable)
- **Scalability Analysis**: Multi-GPU performance characteristics

---

## Development Notes

### Environment Setup
```bash
# Virtual environment creation
python3 -m venv env
source env/bin/activate

# Core dependencies
pip install pytest torch transformers

# Testing
python -m pytest -k "not gpu" -q
```

### Key Commands
```bash
# Run local baseline
python -m src.server.local_baseline --prompt "Hello, world!" --max-tokens 20

# Run tests
python -m pytest tests/test_cpu_smoke.py -v

# CI simulation
python -m pytest -k "not gpu" -q
```

### Git Workflow
- **Development Branch**: `dev` for feature work
- **Main Branch**: `main` for stable releases
- **Commit Convention**: `feat:`, `fix:`, `docs:`, `test:`

---

## References and Resources

### Technical References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [OPT Model Paper](https://arxiv.org/abs/2205.01068)

### Benchmarking Standards
- [MLPerf Inference](https://mlcommons.org/en/inference/)
- [vLLM Benchmarking](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [Transformers Performance](https://huggingface.co/docs/transformers/performance)

---

*Last Updated: Phase 1A Complete - Local Baseline Runner + CPU/MPS Smoke Tests*  
*Next Phase: 1B - Minimal Benchmark Client for vLLM Integration*
