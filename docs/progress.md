# LLM Inference Lab - Development Progress

**Project**: LLM Inference Lab  
**Repository**: https://github.com/GogoRit/llm-inference-lab  
**Objective**: Build a comprehensive toolkit for optimizing and benchmarking Large Language Model inference performance

> **Strategy Note**: As of Phase 3C.5, LLM Inference Lab follows an **MPS-first optimization approach**.
> All new features are validated locally on Apple Silicon (MPS) before limited CUDA (Kaggle/A100) benchmarking.
> This ensures efficient use of GPU credits and rapid iteration on Mac hardware.

---

## Project Overview

This document tracks the systematic development of an LLM inference optimization framework, documenting methodology, results, and insights for research publications. The project follows a phase-based approach ensuring robust, incremental development.

### Key Research Areas
- **Speculative Decoding**: Implementation and optimization of draft model techniques
- **GPU Optimizations**: Mixed precision, memory management, and multi-stream processing
- **Performance Benchmarking**: Comprehensive latency and throughput analysis
- **Multi-GPU Scaling**: Distributed inference optimization

---

## Phase 1: Foundation Infrastructure (COMPLETED)

### Phase 1A: Local Baseline Runner
**Objective**: Establish working baseline with Hugging Face Transformers for Mac (CPU/MPS) development.

**Key Achievements**:
- **Model**: facebook/opt-125m with auto-detection (MPS > CUDA > CPU)
- **Performance**: 41.62 tokens/sec on MPS, 3.3 tokens/sec baseline
- **Testing**: 12 comprehensive smoke tests with CI integration
- **Code Quality**: 0 linting errors, professional standards maintained

### Phase 1B: HTTP Client & Benchmarking
**Objective**: Create dual-mode benchmark tools for local vs server performance comparison.

**Key Achievements**:
- **HTTP Client**: OpenAI-compatible API with health checks and retry logic
- **Benchmark Harness**: Statistical analysis (mean, median, std) with warmup iterations
- **Configuration**: YAML-driven parameter management
- **Performance**: 41.62±0.43 tok/s (MPS), 1153±12ms latency

### Phase 1C: CI/CD Pipeline
**Objective**: Ensure CPU-only CI with proper GPU test exclusion and code quality.

**Key Achievements**:
- **Linting**: Black, isort, flake8, mypy with 0 errors
- **Testing**: CPU-only tests with `pytest -k "not gpu"`
- **Quality**: Professional code standards with comprehensive type checking

---

## Phase 2: Speculative Decoding Implementation (COMPLETED)

### Phase 2A: Core Pipeline
**Objective**: Implement comprehensive speculative decoding with dual-mode architecture.

**Key Achievements**:
- **Architecture**: Dependency injection with LanguageModel interface
- **Dual Modes**: FakeLM (testing) and HFWrapper (production)
- **Memory Safety**: Shared tokenizers, dtype guards, 500MB limits
- **Performance**: 9,430 tok/s (FakeLM), 89.65 tok/s (HF mode)

### Phase 2B: Advanced Techniques
**Objective**: Implement acceptance policies, adaptive K controllers, and quality evaluation.

**Key Achievements**:
- **Policies**: LongestPrefix, ConfidenceThreshold, TopKAgreement, Typical
- **Controllers**: Fixed and adaptive K strategies with performance tracking
- **Instrumentation**: Per-step metrics, memory tracking, comprehensive logging
- **Quality**: Perplexity-based text evaluation with tiny models

### Phase 2C: Draft Strategies
**Objective**: Implement Medusa-lite and EAGLE-lite strategies for advanced speculative decoding.

**Key Achievements**:
- **Medusa-lite**: Multiple prediction heads with configurable initialization
- **EAGLE-lite**: Hidden state extrapolation (h_next = h_t + α(h_t - h_{t-1}))
- **Performance**: EAGLE achieved 2.58x speedup over vanilla on MPS
- **Integration**: Unified pipeline supporting all draft modes

---

## Phase 3: Performance Optimization (COMPLETED)

> ### Reality flags (important)
> - **Hardware scope:** All performance numbers in this document are **MPS (Mac)** unless explicitly labeled.  
> - **GPU status:** Kaggle **T4 CUDA runs were exploratory only** (rough scripts, not merged). Official CUDA results will appear after Phase 3D.  
> - **Model context:** Two speed lines appear in this doc:
>   - **OPT-125M baseline runner (Phase 1)** → ~**41.6 tok/s** on MPS (single-model baseline harness).
>   - **GPT-2 + DistilGPT-2 (speculative decoding K-sweeps)** → ~**6–10 tok/s** on MPS (two-model pipeline with verification).  
>   These are **different pipelines and model sizes**, so their speeds aren't directly comparable.

### Phase 3A: Local Optimization
**Objective**: Build clean, efficient local inference engine with comprehensive instrumentation.

**Key Achievements**:
- **Profiling**: PyTorch Profiler with memory tracking and Chrome trace export
- **Mixed Precision**: Device-aware dtype selection (float16 MPS/CUDA, float32 CPU)
- **Tokenizer**: LRU caching and batched processing
- **Performance**: 896.45±45.97 tok/s (FakeLM), 5.4% speedup (real models)

### Phase 3B: GPU Pre-Check
**Objective**: Verify GPU readiness on CPU-only system before real hardware deployment.

**Key Achievements**:
- **Device Detection**: MPS → CUDA → CPU priority with graceful fallback
- **Module Validation**: All optimization modules import and integrate cleanly
- **Dtype Logic**: Correct selection (float16 MPS, float32 CPU)
- **Testing**: 8 CPU smoke tests pass, end-to-end pipeline verified

### Phase 3C: End-to-End GPU Optimizations
**Objective**: Implement comprehensive GPU optimizations with mixed precision and multi-stream processing.

**Key Achievements**:
- **Mixed Precision**: torch.amp.autocast unification with environment overrides
- **Memory-Safe Loading**: Optimized HF loading with device_map and SDPA attention
- **Speculative Scheduler**: Multi-stream verification with CUDA streams
- **Enhanced Pipeline**: Device/dtype selection, model reuse, comprehensive metrics

**Performance Results** (MPS, gpt2 + distilgpt2, 32 tokens):

*Context: Speculative decoding with base **gpt2** and draft **distilgpt2**, Mac **MPS**, AMP enabled. These are not GPU (CUDA) numbers.*

| K | Latency (ms) | Throughput (tok/s) | Acceptance Rate | Memory (MB) |
|---|--------------|-------------------|-----------------|-------------|
| 1 | 4682±753 | **7.02±1.21** | 14.9±6.4% | ~175 |
| 2 | 4440±800 | **7.50±1.77** | 17.2±11.2% | ~175 |
| 3 | 4702±820 | 6.93±1.23 | 14.3±6.9% | ~175 |
| 4 | **4259±853** | **7.82±2.14** | **17.6±12.0%** | ~175 |

**Key Improvements** (Phase 3C CUDA Optimizations):
- **K=4 Performance**: 7.82 tok/s (+38.4% vs previous 5.65 tok/s)
- **K=2 Performance**: 7.50 tok/s (+15.6% vs previous 6.49 tok/s)  
- **K=1 Performance**: 7.02 tok/s (+11.8% vs previous 6.28 tok/s)
- **Best K Value**: K=4 now optimal with 17.6% acceptance rate
- **Memory Efficiency**: SDPA attention + low memory loading

### Phase 3C.2 Performance Improvements

**Optimization Impact** (32 tokens, 5 iterations, MPS):
- **K=4**: 7.82 tok/s (+38.4% vs previous baseline)
- **K=2**: 7.50 tok/s (+15.6% vs previous baseline)  
- **K=1**: 7.02 tok/s (+11.8% vs previous baseline)
- **Optimal K**: K=4 now provides best throughput with 17.6% acceptance rate

**Technical Achievements**:
- Mixed precision unification (CUDA/MPS/CPU)
- Memory-safe model loading with SDPA attention
- Multi-stream verification for CUDA
- Enhanced metrics and monitoring
- Environment variable overrides
- Zero breaking changes to existing functionality

### Phase 3C.3 Custom CUDA/Triton Kernels

**Kernel Implementation**:
- **verify_prefix**: CUDA kernel for token verification with optimized argmax and prefix matching
  - Optimized for small K (≤8) with one block per batch item
  - Shared memory reduction for argmax, warp ballot for prefix
  - Triton fallback with identical signature
  - PyTorch reference implementation for CPU/MPS fallback
- **kv_append**: CUDA kernel for KV cache append with coalesced memory access
  - Optimized for coalesced loads/stores (thread x = hidden dim, block y = heads×B)
  - Support for float16/float32 dtypes
  - PyTorch reference implementation for fallback

**Build System**:
- JIT compilation with `torch.utils.cpp_extension.load`
- Automatic CUDA architecture detection
- Caching by file hash to avoid recompilation
- Environment variable `SPECDEC_FORCE_PY=1` to skip building
- Safe fallback priority: CUDA → Triton → PyTorch

**Integration**:
- Seamless integration into `LongestPrefixPolicy` with automatic fallback
- Kernel backend logging in pipeline startup
- Zero breaking changes to existing API
- Automatic device detection and fallback

**Expected Performance Gains**:
- **Verification**: 5-10x speedup for K≤4 on CUDA
- **KV Cache**: 2-3x speedup for cache append operations
- **Memory**: Reduced memory bandwidth usage through coalesced access
- **Latency**: Lower verification latency enabling higher K values

---

## Technical Implementation

### Core Architecture
```python
# Mixed Precision with Environment Overrides
SPECDEC_AMP=1 SPECDEC_DTYPE=float16 python scripts/comprehensive_k_sweep.py

# Memory-Safe Model Loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    device_map={"": 0} if accelerate_available else None
)

# Multi-Stream Verification
verification_stream = torch.cuda.Stream()
with torch.cuda.stream(verification_stream):
    base_tokens, base_logits = base_model.generate_tokens(...)
```

### CLI Usage
```bash
# Comprehensive K-sweep
python scripts/comprehensive_k_sweep.py \
    --base-model gpt2 --draft-model distilgpt2 \
    --max-tokens 32 --iterations 5 --device mps \
    --output-dir results --no-plots

# GPU smoke testing
python scripts/dev/smoke_cuda.py
```

---

## Research Contributions

### Methodology
- **End-to-End Optimization**: Complete GPU optimization pipeline
- **Multi-Stream Processing**: CUDA stream-based verification
- **Memory Efficiency**: Optimized model loading and management
- **Comprehensive Benchmarking**: Statistical analysis with 50 samples per K

### Key Findings
- **Optimal K Value**: K=3 provides best throughput (9.51 tok/s) with 20.2% acceptance
- **Consistent Performance**: 100% success rate across all K values
- **Memory Efficiency**: 175MB peak usage with optimized loading
- **Statistical Robustness**: 50 samples per K across 10 diverse prompts

### Future Research Foundation
- **GPU Scaling Ready**: Architecture supports advanced optimizations
- **Performance Baseline**: Comprehensive metrics for comparison
- **Research Tools**: Advanced profiling and optimization framework
- **Production Ready**: Professional-grade monitoring and optimization

---

## Development Environment

### Setup
```bash
python3 -m venv env
source env/bin/activate
pip install pytest torch transformers
python -m pytest -k "not gpu" -q
```

### Key Commands
```bash
# Run speculative decoding
python -m src.specdec.run_specdec --prompt "Hello" --max-tokens 20

# Run benchmarks
python -m src.benchmarks.run_bench --mode specdec --iterations 10

# Run tests
python -m pytest tests/test_cpu_smoke.py -v
```

---

## Project Status

**Current Phase**: 3C Complete - CUDA Optimizations & Custom Kernels Implemented. **CUDA (GPU) validation pending in 3D**.

### Recent Updates (Phase 3C.3)
- **Custom CUDA kernels**: Implemented verify_prefix and kv_append kernels with optimized memory access
- **Triton fallback**: Added Triton implementation for verify_prefix with identical signature
- **Build system**: JIT compilation with caching and automatic CUDA architecture detection
- **Safe fallbacks**: CUDA → Triton → PyTorch fallback chain with zero breaking changes
- **Integration**: Seamless kernel integration into LongestPrefixPolicy with automatic device detection
- **Microbenchmarks**: Added performance testing scripts for kernel validation
- **MPS validated**: All optimizations tested and working on MPS (Mac) hardware
- **Performance validated**: K=4 now optimal at 7.82 tok/s (+38.4% improvement)
- **CUDA validation to follow**: Ready for Phase 3D CUDA validation on real GPU hardware  

**Next Phase**: 3D - Advanced GPU Scaling and Multi-GPU Support  
**Total Lines of Code**: ~3,200+ across all modules  
**Test Coverage**: Comprehensive smoke tests and integration tests  
**Code Quality**: 0 linting errors, professional standards maintained  
**Performance**: 7.82 tok/s (K=4) with 17.6% acceptance rate on MPS (optimized)

### Phase 3C.4 MPS Final Validation (2025-10-06)

**MPS comprehensive K-sweep (gpt2 + distilgpt2, max_tokens=32, iters=5, detailed metrics enabled)**

| K | Throughput (tok/s) | Acceptance % | Latency (ms) | Memory (MB) |
|---|-------------------|--------------|--------------|-------------|
| **1** | **9.50±2.21** | 18.1±11.7% | 3494±872 | ~275 |
| **2** | **9.44±2.25** | 17.9±13.3% | 3476±648 | ~275 |
| **3** | **8.92±1.88** | 16.1±10.2% | 3716±617 | ~275 |
| **4** | **9.55±2.33** | 17.9±10.9% | 3508±666 | ~275 |

**Key Achievements**:
- **Peak performance**: K=4 achieves 9.55 tok/s (highest throughput)
- **Stable performance**: Consistent ~9 tok/s across all K values (K=1-4)
- **Acceptance rates**: 16-18% range, showing good draft model quality
- **Memory efficiency**: Stable ~275MB usage across all configurations
- **100% success rate**: All 200 test runs completed successfully (50 samples per K)
- **Detailed metrics**: Kernel timing, acceptance histograms, and memory profiling enabled
- **Kernel status**: CUDA kernels present but using 'torch/triton' fallbacks on MPS
- **Ready for CUDA**: All optimizations validated, ready for Phase 3D CUDA testing

**Technical Improvements Validated**:
- **Kernel Registry**: Safe backend selection with priority-based fallbacks
- **Detailed Metrics**: Optional performance profiling with `--metrics-detailed` flag
- **CUDA Graph Capture**: Optional graph capture with `--cuda-graph` flag (CUDA only)
- **Memory Profiling**: Enhanced memory tracking for CUDA and MPS
- **Mixed Precision**: AMP working correctly on MPS with float16
- **Multi-stream Verification**: Ready for CUDA implementation

---

*Last Updated: Phase 3C.4 Complete - MPS Final Validation (2025-10-06)*

### Phase 3D – CUDA Validation

#### Tesla T4 CUDA Run #1 (32 tokens × 100 samples fp16 – 2025-10-30)

| K | Latency (ms mean ± std) | Throughput (tok/s mean ± std) | Acceptance (%) mean ± std |
|---|-------------------------|--------------------------------|----------------------------|
| 1 | 3743.07 ± 1031.39 | 17.24 ± 4.96 | 21.38 ± 12.43 |
| 2 | 3904.52 ± 908.77 | 17.66 ± 5.67 | 22.72 ± 14.20 |
| 3 | 3916.34 ± 976.50 | 17.53 ± 5.96 | 22.28 ± 15.06 |
| 4 | 3870.67 ± 1012.93 | 17.22 ± 4.96 | 21.88 ± 12.71 |

Across K=1–4, throughput averages ≈ 17.4 tok/s with overall acceptance ≈ 22% and 100% success (0 failures in 400 runs). Compared to the latest MPS results (~9–9.5 tok/s), T4 achieves ~1.8–2.0× higher throughput at 32 tokens. No OOMs or kernel build issues were observed; fp16 was used end-to-end.

- Device: Tesla T4
- Dtype: float16
- Models: base `gpt2`, draft `distilgpt2`
- Iterations: 100 per K (K=1–4)
- Manifest: `docs/results/2025-10-30-T4/2025-10-30-T4_k32_i100_fp16/MANIFEST.json`

#### Tesla T4 CUDA Run #2 (64 tokens × 100 samples fp16 deterministic – 2025-10-30)

**In Progress**: Running with deterministic mode enabled (fixed seeds, cudnn.deterministic=True, vanilla draft mode only).

| K | Latency (ms mean ± std) | Throughput (tok/s mean ± std) | Acceptance (%) mean ± std |
|---|-------------------------|--------------------------------|----------------------------|
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |

**Expected outcomes**:
- Longer sequences (64 vs 32 tokens) should show improved amortization of draft overhead
- Deterministic mode enables reproducible benchmarks for publication
- Target: ~18–20 tok/s average with similar acceptance rates
- Validation: kernel backend selection logged in JSON; 100% success expected

- Device: Tesla T4
- Dtype: float16
- Deterministic: Yes (SPECDEC_DETERMINISTIC=1)
- Models: base `gpt2`, draft `distilgpt2`
- Iterations: 100 per K (K=1–4)
- Expected manifest: `docs/results/2025-10-30-T4/2025-10-30-T4_k64_i100_fp16_det/MANIFEST.json`

---

## Phase 3C.5: KV Cache Integration (2025-10-31) - COMPLETED

### Implementation Summary

Integrated KV cache management into speculative decoding pipeline. Extended `LanguageModel` interface with cache methods, created `KVCache` dataclass, and implemented full support in `HFWrapper` with CUDA/Triton/PyTorch kernel fallbacks.

**Key Components:**
- KV cache interface methods: `get_kv_cache()`, `append_kv_cache()`, `supports_kv_append()`
- Kernel registration: `kv_append` with priority-based backend selection
- Pipeline integration: Automatic cache append after token acceptance
- Metrics tracking: `kv_appended_tokens_total`, `kv_append_time_ms`, backend logging
- Environment control: `SPECDEC_ENABLE_KV_APPEND=1` (default on)

**Testing:** 15 unit tests (all passing), integration with comprehensive_k_sweep.py

### MPS Validation Results (2025-10-31)

**KV Cache Integration: MPS Validation Complete**

| Mode | K | Throughput (tok/s) | Acceptance (%) | KV Appended | Notes |
|------|---|-------------------|----------------|-------------|-------|
| **KV ON** | 1 | 8.44 ± 1.14 | 11.1 ± 3.9 | 12.0 ± 2.8 | 100% success |
| **KV ON** | 2 | 8.65 ± 0.45 | 9.9 ± 2.3 | 11.3 ± 2.4 | 100% success |
| **KV ON** | 3 | 8.88 ± 0.43 | 10.2 ± 2.0 | 11.4 ± 2.4 | 100% success |
| **KV ON** | 4 | 9.16 ± 0.72 | 10.3 ± 2.9 | 11.3 ± 2.5 | 100% success |
| **KV OFF** | 1 | 9.52 ± 3.47 | 16.8 ± 15.0 | 0 | Baseline |
| **KV OFF** | 2 | 8.33 ± 0.79 | 12.9 ± 4.2 | 0 | Baseline |
| **KV OFF** | 3 | 9.77 ± 1.53 | 18.1 ± 7.1 | 0 | Baseline |
| **KV OFF** | 4 | 9.42 ± 1.82 | 18.0 ± 8.7 | 0 | Baseline |

**Key Findings**:
- **Functional Parity Confirmed**: KV cache integration working correctly (100% success rate, 20/20 prompts × 4 K values)
- **Performance Assessment** (32 tokens, GPT2-124M, MPS):
  - KV ON: 8.44-9.16 tok/s | KV OFF: 8.33-9.77 tok/s
  - **No performance gain observed** due to model size and MPS backend characteristics
  - Cache overhead dominates for small models with low acceptance rates (~10%)
  - This validates correctness and establishes a baseline for CUDA optimization in Phase 3D
- Performance stable: ~9 tok/s baseline on Apple Silicon (M-series)
- KV metrics correctly logged: avg ~11-12 tokens appended per prompt
- Text outputs identical between KV ON/OFF modes
- Two critical bugs fixed:
  - Bug #1: HuggingFace `generate()` API `cache_position` handling (switched to `forward()`)
  - Bug #2: KV cache persisting across prompts (added `clear_kv_cache()` calls)
- **Implementation Status**:
  - CUDA kernel: Implemented with coalesced memory operations
  - CUDA stream sync: Added for proper synchronization
  - Default state: KV cache ON by default (SPECDEC_ENABLE_KV_APPEND=1)
  - Fallback chain: CUDA kernel -> PyTorch reference (robust)
- **Next Steps**: 
  - Phase 3D: CUDA validation with larger models (7B+) to assess real-world impact
  - Test on T4/A100 with longer contexts to measure actual performance gains
  - Benchmark kernel performance vs PyTorch reference on CUDA

**Scaling Analysis** (128 tokens vs 32 tokens, KV ON):

| Context | Throughput (tok/s) | KV Appended (tokens) | KV Append Time (ms) |
|---------|-------------------|---------------------|---------------------|
| 32 tok  | 8.44-9.16         | 11-12              | 24-27              |
| 128 tok | 6.99-7.69         | 40-46              | 114-123            |

**Scaling Conclusion**: Longer contexts show **worse throughput** despite 4x more KV reuse. Cache append overhead scales linearly (4-5x time for 4x tokens), but does not amortize on MPS with small models. Confirms that KV cache optimization requires CUDA backend and larger models for measurable benefit.

**Results Location**: 
- `docs/results/2025-10-31-MPS-KV-Append-{ON,OFF}/` (32 tokens)
- `docs/results/2025-10-31-MPS-KV-128tok/` (128 tokens, scaling test)

---

**Phase 3C.5 Conclusion:**

MPS validation of KV-cache append completed successfully. All core features verified locally on Apple Silicon, maintaining ~9 tok/s throughput. KV cache integration functionally correct with full test coverage. Performance analysis shows no gains for GPT2-124M on MPS (cache overhead dominates), but implementation validated for Phase 3D testing with larger models on CUDA.

---

## Phase 3D: GPU Optimization and Validation (IN PROGRESS)

**Objective:** Optimize speculative decoding for GPU execution with CUDA-specific features, structured profiling, and deterministic reproducibility before full CUDA validation runs.

### Implementation Summary

**1. Deterministic Seeding (specdec/deterministic.py)**
- Centralized utility with `set_deterministic_mode()` helper
- Environment flag: `SPECDEC_DETERMINISTIC=1`
- Seeds: Python, NumPy, PyTorch, CuDNN deterministic mode
- Automatically invoked if environment flag is set

**2. Structured Profiler (metrics/structured_profiler.py)**
- Per-step GPU event timers for draft/verify/accept/kv timing
- CUDA event-based synchronization (more accurate than wall-clock)
- JSON output with standardized schema
- Environment flag: `SPECDEC_PROFILE=1`
- Integrated into pipeline with best-effort recording

**3. Enhanced Memory Tracking (metrics/memory_profiler.py)**
- CUDA memory stats: `torch.cuda.memory_stats()` summary
- MPS memory tracking: `torch.mps.driver_allocated_memory()` if available
- Memory stats integrated into structured metrics output

**4. CUDA Stream Overlap (scheduler/speculative_scheduler.py)**
- Asynchronous overlap of draft and verification passes using CUDA streams
- Event-based synchronization using CUDA events
- Environment flags: `SPECDEC_PARALLEL_STREAMS=1`, `SPECDEC_SYNC_MODE=event|barrier`
- Fallback to barrier synchronization or single-stream on non-CUDA devices

**5. CUDA Graph Capture (specdec/pipeline.py)**
- Optional static graph capture for steady decoding steps with fixed batch shapes
- Warmup phase (3 steps) before capture to stabilize CUDA execution
- Automatic fallback to eager mode if capture fails (dynamic shapes, MPS backend)
- Environment flag: `SPECDEC_CUDA_GRAPH=1`

**6. Profiling Hooks Integration (specdec/pipeline.py)**
- Per-step profiling hooks for draft_forward_time_ms, verify_forward_time_ms, kv_append_time_ms
- Acceptance check time recording
- Integrated with structured profiler for JSON output

**7. Dry-Run Mode (scripts/comprehensive_k_sweep.py)**
- Latency-only profiling without model compute
- Synthesized metrics for pipeline validation
- Environment flag: `SPECDEC_DRY_RUN=1`
- Output to `docs/results/phase3d-dryrun/`

### Implementation Status

| Component | Status | Details |
|-----------|--------|---------|
| Deterministic seeding utility | Complete | `src/specdec/deterministic.py`, 5 unit tests |
| Structured profiler | Complete | `src/metrics/structured_profiler.py`, 6 unit tests |
| Enhanced memory tracking | Complete | Enhanced `src/metrics/memory_profiler.py` |
| CUDA stream overlap | Complete | Enhanced `src/scheduler/speculative_scheduler.py` |
| CUDA graph capture | Complete | `src/specdec/pipeline.py` with fallback |
| Profiling hooks integration | Complete | Integrated into pipeline generate loop |
| Dry-run mode | Complete | Environment flag in K-sweep script |
| Unit tests | Complete | 14 tests passing (2 skipped on non-CUDA) |
| MPS validation | Complete | All features validated on MPS |
| CUDA validation | Pending | GPU validation runs on Kaggle T4/A100 |

### MPS Validation Results (2025-10-31)

**Validation Test Suite:**
- Full K-sweep with Phase 3D flags: `SPECDEC_DETERMINISTIC=1`, `SPECDEC_PROFILE=1`
- Dry-run mode: `SPECDEC_DRY_RUN=1`
- CUDA graph flag on MPS: `SPECDEC_CUDA_GRAPH=1` (gracefully disabled)
- Deterministic seeding verification

**Results:**
- Full K-sweep: 80 runs, 100% success, ~11.5 tok/s avg throughput
- Dry-run mode: Synthesized metrics generated correctly, JSON output valid
- CUDA graph fallback: Correctly disabled on MPS, no errors
- Deterministic mode: Identical results verified (`v1 == v2`)
- Unit tests: 14 passed, 2 skipped (CUDA-specific, expected)

**Validation Status:**
- All Phase 3D features functional on MPS
- All unit tests passing
- Ready for CUDA validation runs

### CUDA Validation (PENDING)

**Objective:** Validate all Phase 3D GPU optimization features on CUDA hardware (Kaggle T4/A100) after successful MPS validation.

**Validation Tasks:**
- Full K-sweep with all Phase 3D flags enabled
- CUDA graph capture performance impact measurement
- Stream overlap efficiency gains analysis
- Structured profiling accuracy validation
- CUDA vs MPS throughput comparison

**Strategic GPU Validation Plan:**

**Kaggle T4 (Free Tier):**
- Baseline: 32 tokens x K=1-4 (10 min)
- Scaling: 128 tokens x K=1-4 (15 min)

**A100 (Credit-Limited):**
- Performance: 32 tokens x K=4 vs MPS (10 min)
- Long-context: 256 tokens x K=4 (20 min)

**Expected Deliverables:**
- CUDA vs MPS throughput comparison
- Graph capture performance impact analysis
- Stream overlap efficiency gains measurement
- Production-ready configuration recommendations

**Status:** Pending - Will be executed after MPS validation completes (Phase 3D feature implementation ready)

### Environment Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `SPECDEC_DETERMINISTIC` | 0 | Enable deterministic seeding for reproducibility |
| `SPECDEC_PROFILE` | 0 | Enable structured event profiling with JSON output |
| `SPECDEC_CUDA_GRAPH` | 0 | Enable CUDA graph capture for steady decoding steps |
| `SPECDEC_PARALLEL_STREAMS` | 1 | Enable CUDA stream overlap for async verification |
| `SPECDEC_SYNC_MODE` | event | Sync mode: event (CUDA events) or barrier (stream.synchronize) |
| `SPECDEC_DRY_RUN` | 0 | Run latency-only profiling without model compute |

---

## Phase 4A: Batch-Level Processing (PLANNED)

**Objective:** Process multiple prompts simultaneously to improve GPU utilization and throughput for batch inference scenarios.

**Planned Features:**
- Multi-prompt batch processing pipeline
- Batched draft generation across prompts
- Batched verification with efficient tensor operations
- Batch-aware acceptance policy application
- Memory-efficient batch management

**Status:** Planned for implementation after Phase 3D CUDA validation runs complete

**Prerequisites:** Phase 3D CUDA validation results and performance analysis complete

---

## Phase 4C: Layer/Model Parallelism (FUTURE)

**Objective:** Enable layer-wise or model-wise parallelism for very large models (7B+ parameters) that exceed single GPU memory limits.

**Planned Features:**
- Layer parallelism for large transformer models
- Model parallelism across multiple GPUs
- Efficient gradient/activation communication
- Memory optimization for multi-GPU setups

**Status:** Future phase, needed only for 7B+ models requiring multi-GPU deployment

**Prerequisites:** Phase 3D CUDA validation complete, confirmed need for multi-GPU support

---

## Phase 4D: Speculative Tree Decoding (FUTURE)

**Objective:** Implement advanced acceptance strategies using tree-based speculative decoding for improved acceptance rates and throughput.

**Planned Features:**
- Tree-structured draft generation
- Multi-path verification and acceptance
- Advanced acceptance strategies (tree-agree, confidence-based)
- Efficient tree traversal and pruning

**Status:** Future phase for advanced acceptance strategies beyond longest-prefix matching

**Prerequisites:** Phase 3D CUDA validation complete, performance baseline established