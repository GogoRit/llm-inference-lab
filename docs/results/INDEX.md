# Results Index

Benchmark results organized chronologically (newest first).

## Format

`Date | Hardware | Models | Configuration | Dtype | Performance | Path`

---

## Phase 3D: GPU Optimization and CUDA Validation

### 2025-10-31 | MPS | Phase 3D Full Validation
- **Date**: 2025-10-31
- **Hardware**: Apple Silicon MPS
- **Models**: gpt2 + distilgpt2
- **Configuration**: 32 tokens, Phase 3D features enabled
- **Performance**: ~11.5 tok/s average
- **Path**: `docs/results/2025-10-31-MPS-Phase3D-Full-Validation/`

### 2025-10-31 | MPS | Phase 3D Dry-Run
- **Date**: 2025-10-31
- **Hardware**: Apple Silicon MPS
- **Configuration**: Dry-run mode (latency profiling without model compute)
- **Path**: `docs/results/2025-10-31-MPS-Phase3D-DryRun/`

### 2025-10-30 | Tesla T4 | Phase 3D Run #2 (In Progress)
- **Date**: 2025-10-30
- **Hardware**: Tesla T4 (CUDA)
- **Models**: gpt2 + distilgpt2
- **Configuration**: 64 tokens × 100 iterations, deterministic mode
- **Dtype**: fp16
- **Performance**: TBD
- **Path**: `docs/results/2025-10-30-T4-Phase3D-Run2-64tok-100iter-fp16-det/` (pending)

### 2025-10-30 | Tesla T4 | Phase 3D Run #1
- **Date**: 2025-10-30
- **Hardware**: Tesla T4 (CUDA)
- **Models**: gpt2 + distilgpt2
- **Configuration**: 32 tokens × 100 iterations
- **Dtype**: fp16
- **Performance**: ~17.6 tok/s @ K=2
- **Path**: `docs/results/2025-10-30-T4-Phase3D-Run1-32tok-100iter-fp16/`

---

## Phase 3C.5: KV Cache Integration

### 2025-10-31 | MPS | KV Cache ON (128 tokens)
- **Date**: 2025-10-31
- **Hardware**: Apple Silicon MPS
- **Models**: gpt2 + distilgpt2
- **Configuration**: 128 tokens, KV cache enabled
- **Performance**: 6.99-7.69 tok/s
- **Path**: `docs/results/2025-10-31-MPS-Phase3C5-KV-Cache-ON-128tok/`

### 2025-10-31 | MPS | KV Cache ON (32 tokens)
- **Date**: 2025-10-31
- **Hardware**: Apple Silicon MPS
- **Models**: gpt2 + distilgpt2
- **Configuration**: 32 tokens, KV cache enabled
- **Performance**: 8.44-9.16 tok/s
- **Path**: `docs/results/2025-10-31-MPS-Phase3C5-KV-Cache-ON-32tok/`

### 2025-10-31 | MPS | KV Cache OFF (32 tokens)
- **Date**: 2025-10-31
- **Hardware**: Apple Silicon MPS
- **Models**: gpt2 + distilgpt2
- **Configuration**: 32 tokens, KV cache disabled
- **Performance**: 8.33-9.77 tok/s
- **Path**: `docs/results/2025-10-31-MPS-Phase3C5-KV-Cache-OFF-32tok/`

### 2025-10-31 | MPS | Phase 3C.5 Baseline
- **Date**: 2025-10-31
- **Hardware**: Apple Silicon MPS
- **Models**: gpt2 + distilgpt2
- **Configuration**: Baseline before KV cache integration
- **Path**: `docs/results/2025-10-31-MPS-Phase3C5-Baseline/`

---

## Phase 3C: CUDA Kernels and GPU Optimizations

### 2025-10-06 | MPS | Phase 3C Test Runs
- **Date**: 2025-10-06
- **Hardware**: Apple Silicon MPS
- **Models**: gpt2 + distilgpt2
- **Configuration**: Multiple test runs during Phase 3C development
- **Path**: `docs/results/2025-10-06-MPS-Phase3C-Test-Runs/`

### 2025-10-04 | CPU | Phase 3C Baseline
- **Date**: 2025-10-04
- **Hardware**: CPU
- **Models**: gpt2 + distilgpt2
- **Configuration**: CPU baseline for Phase 3C
- **Path**: `docs/results/2025-10-04-CPU-Phase3C-Baseline/`

---

**Last Updated**: 2025-10-31
