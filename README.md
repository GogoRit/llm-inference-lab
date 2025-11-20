# Zero Copy Speculative Decoding Engine

This repository contains a research oriented inference engine for speculative decoding of large language models that is optimized for memory constrained hardware such as NVIDIA T4 and Apple Silicon.

The central goal is to reduce latency and memory overhead while preserving correctness. The engine introduces a zero copy architecture that avoids ragged tensor realignment and repeated memory allocation, and it validates the design on both MPS and CUDA devices.

## Motivation

Speculative decoding accelerates autoregressive generation by allowing a smaller draft model to propose several tokens that a larger base model then verifies. In practice this process leads to ragged sequences inside a batch because different prompts accept different numbers of draft tokens. Standard engines handle this through padding, slicing, and realignment of tensors and key value caches.

These operations are expensive on memory constrained hardware. They introduce additional memory bandwidth pressure and can dominate runtime once batch sizes and context lengths grow. At the same time, speculative decoding must preserve strict correctness invariants such as alignment of position ids, attention masks, and key value states.

This project studies how far we can push speculative decoding on a single modest GPU or Apple Silicon device by changing the memory layout and cache management rather than only adding new draft strategies.

## Core Architecture

The engine implements three core ideas.

1. Zero copy ring buffer for key value cache

   Instead of repeatedly allocating and concatenating tensors, the system allocates a fixed size memory arena for the key value cache at startup. All key and value tensors live inside this ring buffer. New tokens are written in place, and the current sequence length is tracked through integer pointers. This avoids repeated tensor allocation and reduces fragmentation.

2. Pointer based rollback for speculative rejection

   When speculative decoding rejects some of the draft tokens, the engine does not slice tensors or rebuild caches. It updates integer pointers that represent the current valid prefix and treats the invalidated region as dead space in the ring buffer. The effective rollback cost becomes constant in the number of rejected tokens, which makes aggressive speculative strategies more attractive on small GPUs.

3. Parallel verification with constant time across K

   Verification of K draft tokens is performed in a single forward pass of the base model. The draft tokens are treated as an extension of the prompt, and the base model processes them in one prefill step before producing the bonus token. This decouples verification latency from K and leads to approximately constant verification time for different draft depths, subject to cache use and model size.

The implementation integrates with Hugging Face models while maintaining a separate kernel registry for CUDA, Triton, and pure PyTorch backends so that all features have safe fallbacks.

## Current Status

The project is in an active research prototype stage.

1. The zero copy ring buffer and pointer rollback are implemented and tested on MPS and CUDA.

2. The constant K verification path is implemented and validated for small and medium models.

3. A deterministic profiling and metrics system is in place and collects per step timing and memory data.

4. The engine has been exercised on GPT2 with DistilGPT2 and on Llama 3.2 model pairs on both Apple Silicon and Tesla T4.

The detailed engineering history and full logs have been moved to docs/engineering/appendix_engineering.md so that this README remains concise.

## Experimental Overview

A small selection of representative results is kept here. Full tables and plots live in docs/results/curated_results.md.

1. GPT2 plus DistilGPT2 on Apple Silicon MPS shows throughput in the range of about ten tokens per second for speculative decoding with moderate acceptance rates. CUDA T4 runs reach around seventeen tokens per second at short sequence lengths.

2. For Llama 3.2 on Tesla T4, speculative decoding with batch size one reaches around eight tokens per second with high acceptance probabilities. Increasing batch size on T4 reduces both throughput and acceptance for this model pair, which highlights the importance of hardware aware design.

3. The parallel verification path demonstrates approximately constant verification latency when moving from one to several draft tokens, which confirms that the prefill based design behaves as intended.

These experiments indicate that speculative decoding on modest hardware is strongly shaped by the interaction between model pair choice, acceptance behaviour, and memory layout. The zero copy design removes a known source of overhead and provides a controlled platform for further study.

## Repository Layout

The project is organized into the following top-level directories:

### Core Implementation (`src/`)

- `src/specdec/`: Main speculative decoding package
  - `core/`: Pipeline, batch handlers, generation loop, and utilities
  - `cache/`: Zero-copy KV cache manager and types
  - `models/`: HuggingFace wrappers, draft models, and fake LM for testing
  - `policies/`: Acceptance policies and K controllers
  - `modes/`: Draft generation modes (Eagle, Medusa)
  - `utils/`: Interfaces, deterministic mode, token validation, graph capture

- `src/kernels/`: CUDA and Triton kernel implementations with registry
- `src/metrics/`: Profiling and metrics collection
- `src/optimization/`: Mixed precision and tokenizer optimizations
- `src/scheduler/`: Speculative scheduler for multi-stream verification
- `src/server/`: Server code for baseline comparisons
- `src/benchmarks/`: Benchmark runner and quality evaluation

### Testing (`tests/`)

- `tests/specdec/`: Unit tests for core components
- `tests/integrations/`: Integration tests including zero-copy correctness
- `tests/`: Additional unit tests for kernels, metrics, and utilities

### Scripts (`scripts/`)

- `scripts/comprehensive_k_sweep.py`: Main K-sweep benchmarking script
- `scripts/k_sweep/`: K-sweep utilities (runner, plotting, results, utils)
- `scripts/benchmarks/`: Benchmark scripts
- `scripts/dev/`: Development and smoke test scripts

### Documentation (`docs/`)

- `docs/design/`: Design documents (zero-copy architecture, correctness invariants)
- `docs/engineering/`: Engineering logs and backups
- `docs/results/`: Experimental results organized by date and configuration
- `docs/figures/`: Figures and plots

### Configuration (`configs/`)

- YAML configuration files for different model pairs and experimental setups

### Progress Reports (`progress/`)

- Research-facing progress reports and summaries

See `docs/engineering/appendix_engineering.md` for a preserved copy of the prior long form progress log and README backup.

## Getting Started

To run a basic speculative decoding benchmark on your local device, first create and activate your Python environment and install the project dependencies as documented in the existing setup instructions or requirements file.

Then you can run a simple smoke test such as:

```bash
python scripts/dev/smoke_cuda.py
```

or for MPS:

```bash
python scripts/comprehensive_k_sweep.py
```

These scripts exercise the core speculative decoding pipeline and will write metrics into the docs/results folder.

## Citation

If you use this repository in academic work, please cite it as follows:

```
LLM Inference Lab. A transparent research grade inference runtime for reproducible speculative decoding experiments. Repository available at https://github.com/GogoRit/llm-inference-lab
```

## Reproducing reference experiments

This section records the exact environment settings and commands that were used for the main CUDA experiments described in the progress report. These are intended as canonical recipes for reproducing the reference numbers on Tesla T4.

### Phase 3D: GPT2 plus DistilGPT2 on Tesla T4

This configuration was used to generate the Phase 3D T4 results with sixty four token sequences and a full K sweep.

Set the environment variables:

```bash
export SPECDEC_AMP=1
export SPECDEC_DTYPE=float16
export SPECDEC_DETAILED_METRICS=1
export SPECDEC_DETERMINISTIC=1
export SPECDEC_PROFILE=1
export SPECDEC_CUDA_GRAPH=1
export SPECDEC_PARALLEL_STREAMS=1
export SPECDEC_SYNC_MODE=event
```

Then run the comprehensive K sweep:

```bash
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 \
  --draft-model distilgpt2 \
  --max-tokens 64 \
  --iterations 100 \
  --device cuda \
  --deterministic \
  --output-dir docs/results/2025-10-30-T4-Phase3D-Run2-64tok-100iter-fp16 \
  --no-plots
```

This produces the reference Phase 3D T4 results with approximately six tokens per second throughput at sixty four tokens and about forty percent acceptance, as summarized in docs/results/curated_results.md and the engineering appendix.

### Phase 4A: Llama 3.2 on Tesla T4

This configuration was used to generate the Llama 3.2 speculative decoding results with K equal to one, focusing on the behaviour at different batch sizes.

Set the environment variables:

```bash
export SPECDEC_FORCE_PYTORCH_BACKEND=1
export SPECDEC_BATCH_SIZE=1
export SPECDEC_PARALLEL_STREAMS=1
export SPECDEC_DTYPE=float16
export SPECDEC_ENABLE_KV_APPEND=0
export SPECDEC_CUDA_GRAPH=0
```

Then run the K equals one sweep for batch size one:

```bash
python scripts/comprehensive_k_sweep.py \
  --base-model "meta-llama/Llama-3.2-3B" \
  --draft-model "meta-llama/Llama-3.2-1B" \
  --max-tokens 64 \
  --iterations 1 \
  --device cuda \
  --max-k 1 \
  --output-dir docs/results/2025-11-18-llama32-t4-batch1 \
  --no-plots
```

For batch size two and four, repeat the same command with:

```bash
export SPECDEC_BATCH_SIZE=2
```

or

```bash
export SPECDEC_BATCH_SIZE=4
```

and adjust the output directory accordingly. For batch size four, max tokens was reduced to thirty two tokens in order to respect the memory limits of Tesla T4.

These commands reproduce the T4 results where speculative decoding reaches around eight and a half tokens per second at batch size one, and where throughput and acceptance both drop at larger batch sizes. The non speculative Llama 3.2 baseline used the same script with the draft model argument omitted and the baseline mode enabled, as documented in the engineering appendix.
