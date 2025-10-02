# LLM Inference Lab

A comprehensive toolkit for optimizing and benchmarking Large Language Model inference performance with advanced techniques including speculative decoding, custom CUDA kernels, and intelligent batching.

## Overview

This project provides a modular framework for:
- High-performance LLM serving with custom CUDA kernels
- Speculative decoding for faster inference
- Intelligent request batching and scheduling
- Comprehensive benchmarking and profiling tools
- API serving with FastAPI and Streamlit interfaces

## Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with compute capability 7.0+

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

```bash
# Basic inference with default settings
python -m src.server.local_baseline --prompt "Hello, world!"

# Inference with custom parameters
python -m src.server.local_baseline --prompt "The future of AI is" --max-tokens 100 --verbose

# Run performance benchmark
python -m src.benchmarks.run_bench --prompt "Hello, world!" --iterations 5

# Use custom configuration
python -m src.server.local_baseline --prompt "Test" --config configs/baseline.yaml
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

### Programmatic Usage

```python
from src.server.local_baseline import LocalBaselineRunner
from src.benchmarks.run_bench import BenchmarkRunner

# Basic inference
runner = LocalBaselineRunner(config_path="configs/baseline.yaml")
result = runner.run("Hello, world!", max_new_tokens=50)
print(f"Generated: {result['text']}")
print(f"Latency: {result['latency_ms']:.2f} ms")

# Performance benchmarking
benchmark = BenchmarkRunner(config_path="configs/baseline.yaml")
stats = benchmark.run_benchmark("Hello, world!", iterations=10)
benchmark.print_summary(stats)
```

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
| 1B | Benchmark client | Next | Performance measurement and statistical analysis |
| 1C | CI/CD sanity | Planned | Automated testing and code quality checks |
| 2 | Speculative decoding | Future | Medusa/EAGLE implementation for faster inference |
| 3 | Quantization | Future | BitsAndBytes 4-bit/8-bit quantization experiments |
| 4 | Multi-GPU scaling | Future | Distributed inference and load balancing |
| 5 | Cloud deployment | Future | A100/H100 benchmarking and results collection |

### Current Focus
- **Phase 1A**: Complete - Local baseline runner with logging and configuration
- **Phase 1B**: In Progress - Benchmark harness for performance measurement
- **Phase 1C**: Planned - CI/CD pipeline optimization

### Future Phases
- **Phase 2**: Speculative decoding research and implementation
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
