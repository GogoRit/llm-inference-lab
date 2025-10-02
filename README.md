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
# Start the inference server
python -m src.server.main

# Run benchmarks
python -m src.benchmarks.run_benchmark

# Launch Streamlit interface
streamlit run src/server/streamlit_app.py
```

### API Usage

```python
from src.server.client import InferenceClient

client = InferenceClient("http://localhost:8000")
response = client.generate("Hello, world!", max_tokens=100)
print(response.text)
```

### Custom Kernel Development

```python
from src.kernels import CustomKernel

# Load and compile custom CUDA kernel
kernel = CustomKernel("attention_kernel.cu")
result = kernel.execute(input_tensor)
```

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

## Roadmap

### Phase 1: Core Infrastructure (Q1 2025)
- [ ] Basic inference server with FastAPI
- [ ] Request batching and scheduling
- [ ] Basic benchmarking framework
- [ ] Streamlit web interface

### Phase 2: Advanced Optimizations (Q2 2025)
- [ ] Speculative decoding implementation
- [ ] Custom CUDA kernels for attention
- [ ] Memory optimization techniques
- [ ] Multi-GPU support

### Phase 3: Production Features (Q3 2025)
- [ ] Load balancing and scaling
- [ ] Monitoring and observability
- [ ] Docker containerization
- [ ] Kubernetes deployment

### Phase 4: Advanced Features (Q4 2025)
- [ ] Dynamic batching optimization
- [ ] Model quantization support
- [ ] Distributed inference
- [ ] Advanced profiling tools

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
