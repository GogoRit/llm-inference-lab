# State of the Codebase Report
**Generated:** 2025-01-XX  
**Repository:** llm-inference-lab (Zero-Copy Speculative Decoding Engine)

---

## A. Current Stage Summary (What Works Today)

The codebase implements a **zero-copy speculative decoding engine** with ring-buffer KV cache and pointer-based rollback. The system runs end-to-end on **CUDA (T4)**, **MPS (Apple Silicon)**, and **CPU** with correctness validation. The core speculative decoding loop (draft → verify → accept/reject → commit/rollback) is functional with both single-prompt and batch processing paths. Correctness is validated through deterministic mode tests (`tests/test_deterministic_mode.py`, `tests/test_specdec_cpu_correctness.py`) that ensure token-by-token parity with vanilla greedy decoding when `draft_model == base_model` and `do_sample=False`. The engine supports GPT2/DistilGPT2 and Llama 3.2 model pairs, with benchmark results documented in `docs/results/curated_results.md` showing throughput in the 5-9 tokens/sec range on T4 and MPS.

**Key Entry Points:**
- **CLI:** `src/specdec_cli/main.py` (`specdec bench`, `specdec run`)
- **Direct API:** `src/specdec/run_specdec.py` (module entrypoint)
- **Benchmarking:** `scripts/comprehensive_k_sweep.py` (main K-sweep script)

**Core Generation Path:**
- Single-prompt: `SpeculativePipeline.generate()` → `src/specdec/core/pipeline.py:1090`
- Batch: `SpeculativePipeline.generate_batch()` → `BatchGenerationLoop.run()` → `src/specdec/core/batch_loop.py:79`

---

## B. Architecture Map

### Main Modules and Responsibilities

**Core Decoding (`src/specdec/core/`):**
- `pipeline.py` - `SpeculativePipeline`: Main orchestrator, model initialization, single-prompt generation loop
- `batch_loop.py` - `BatchGenerationLoop`: Batch generation orchestrator
- `batch_handlers.py` - Handler classes:
  - `DraftGenerationHandler`: Draft token generation
  - `VerificationHandler`: Base model verification
  - `AcceptanceHandler`: Token acceptance policy application
  - `RollbackHandler`: Zero-copy pointer rollback (O(1) rejection)
  - `KVCacheHandler`: KV cache management operations
- `kv_cache_verification.py` - KV cache correctness checksums and alignment verification
- `sequence_pool.py` - Sequence pooling for batch management
- `sequence_utils.py` - Padding, unpadding, position ID creation

**KV Cache Management (`src/specdec/cache/`):**
- `kv_cache_manager.py` - `SafeKVCacheManager`: Pre-allocated ring buffer, pointer tracking, in-place updates
  - Pre-allocated buffers: `[batch_size, num_heads, max_seq_len, head_dim]` per layer
  - Pointer tracking: `base_current_seq_lens`, `draft_current_seq_lens` (integer lists)
  - Ring buffer reset: `reset()` preserves buffers, only resets pointers (lines 82-135)

**Models (`src/specdec/models/`):**
- `hf_wrappers.py` - HuggingFace model wrappers with KV cache support
- `draft_model.py` - Draft model interface
- `fake_lm.py` - Fake language model for testing

**Policies (`src/specdec/policies/`):**
- `policies.py` - Acceptance policies: `longest_prefix`, `conf_threshold`, `topk_agree`, `typical`
- `controllers.py` - K controllers: `fixed`, `adaptive`

**Kernels (`src/kernels/`):**
- `registry.py` - `KernelRegistry`: Priority-based backend selection (CUDA > Triton > PyTorch)
- `cuda/verify.cu` - CUDA verification kernel
- `cuda/kv_cache.cu` - CUDA KV cache append kernel
- `triton/verify.py` - Triton verification kernel
- `reference.py` - PyTorch fallback implementations

**Benchmarking (`scripts/`):**
- `comprehensive_k_sweep.py` - Main K-sweep benchmark runner
- `k_sweep/runner.py` - K-sweep execution logic
- `k_sweep/results.py` - CSV/JSON serialization (`save_results()`)
- `k_sweep/plotting.py` - Plot generation

**Metrics (`src/metrics/`):**
- `structured_profiler.py` - Per-step timing (draft, verify, accept, KV append)
- `detailed_profiler.py` - Detailed profiling hooks
- `memory_profiler.py` - Memory tracking

### Call Graph: Generation Loop

**Single-Prompt Path (`pipeline.py:1090`):**
```
SpeculativePipeline.generate()
  ├─> Tokenize prompt
  ├─> while len(generated_tokens) < max_tokens:
  │     ├─> controller.get_k() → K value
  │     ├─> draft_lm.generate_tokens(k) → draft_tokens, draft_logits
  │     ├─> base_lm.generate_tokens(k+1) → base_tokens, base_logits
  │     ├─> policy.accept_tokens() → accepted_len
  │     ├─> if accepted_len > 0:
  │     │     ├─> Append accepted tokens to generated_tokens
  │     │     └─> base_lm.append_kv_cache(accepted_kv) [if KV append enabled]
  │     └─> else:
  │           └─> base_lm.generate_tokens(1) → fallback token
  └─> Decode and return results
```

**Batch Path (`batch_loop.py:79`):**
```
BatchGenerationLoop.run()
  ├─> Initialize sequence_manager, metrics_collector
  ├─> while any(sequence_manager.batch_active):
  │     ├─> controller.get_k() → K value
  │     ├─> DraftGenerationHandler.generate_draft_tokens()
  │     │     └─> draft_lm.generate_tokens() [with CUDA streams if enabled]
  │     ├─> VerificationHandler.verify_draft_tokens()
  │     │     └─> base_lm.generate_tokens(k+1) [parallel verification]
  │     ├─> For each active sequence:
  │     │     ├─> AcceptanceHandler.apply_acceptance_policy() → accepted_len
  │     │     ├─> if accepted_len > 0:
  │     │     │     ├─> Extract accepted tokens
  │     │     │     └─> RollbackHandler.update_sequence_pointers()
  │     │     │           └─> kv_cache_manager.base_current_seq_lens[idx] = original_len + accepted_len
  │     │     └─> else:
  │     │           └─> AcceptanceHandler.sample_fallback_token()
  │     └─> KVCacheHandler.update_sequence_lengths_dict()
  └─> Return batch_generated_tokens, batch_metrics
```

### Memory Allocation and Reuse

**KV Ring Buffer (`kv_cache_manager.py`):**
- **Allocation:** `_ensure_buffers_initialized()` (lines 202-270) - Pre-allocates `[batch_size, num_heads, max_seq_len, head_dim]` tensors per layer at startup
- **Reuse:** `reset()` (lines 82-135) - Preserves GPU tensors, only resets `base_current_seq_lens` and `draft_current_seq_lens` pointers to 0
- **In-place updates:** `update_base_cache()` (lines 335-467) - Writes new KV cache directly into ring buffer at `current_pos:new_pos` indices
- **Pointer rollback:** `RollbackHandler.update_sequence_pointers()` (lines 798-836) - Updates `base_current_seq_lens[idx]` to `original_len + accepted_len` (O(1) operation)

**Logits Buffers:**
- Allocated per-forward-pass by model wrappers (no pre-allocation)
- Shape: `[batch_size, seq_len, vocab_size]`

**Token Buffers:**
- `current_input_ids`: List of 1D tensors per sequence (grows dynamically)
- `batch_generated_tokens`: List of lists (Python lists, not pre-allocated)

---

## C. What Has Been Achieved

### Supported Models/Pairs

**Tested and Working:**
- **GPT2 + DistilGPT2:** Primary test pair, validated on T4 and MPS
  - Results: `docs/results/curated_results.md` (T4: ~6 tok/s, MPS: ~9.5 tok/s)
- **Llama 3.2-3B + Llama 3.2-1B:** Validated on T4
  - Results: `docs/results/llama32_pair_results/` (T4: ~8.5 tok/s at batch_size=1)
- **Baseline mode:** Non-speculative decoding (`draft_model=None`) - `pipeline.py:1626`

**Model Loading:**
- `src/specdec/models/hf_wrappers.py` - `create_tiny_hf_wrapper()` supports any HuggingFace model
- Tokenizer compatibility checks: `pipeline.py:760` (`_check_compatibility()`)

### Device Support

**CUDA (T4):**
- ✅ Working with CUDA kernels (`src/kernels/cuda/`)
- ✅ CUDA streams for parallel draft/verify (`batch_handlers.py:161-193`)
- ✅ Results: `docs/results/2025-10-30-T4-Phase3D-Run1-32tok-100iter-fp16/`
- Environment: `SPECDEC_AMP=1`, `SPECDEC_DTYPE=float16`, `SPECDEC_CUDA_GRAPH=1` (disabled in code, see line 349)

**MPS (Apple Silicon):**
- ✅ Working with PyTorch fallback kernels
- ✅ Results: `docs/results/2025-10-31-MPS-Phase3D-Full-Validation/`
- Environment: `SPECDEC_DTYPE=float16` (auto-selected)

**CPU:**
- ✅ Working with PyTorch fallback
- ✅ Results: `docs/results/2025-10-04-CPU-Phase3C-Baseline/`
- Tests: `tests/test_cpu_smoke.py`, `tests/test_specdec_cpu_correctness.py`

### Existing Benchmark Results

**Location:** `docs/results/`

**Key Results Files:**
1. **T4 GPT2/DistilGPT2 (Phase 3D):**
   - `2025-10-30-T4-Phase3D-Run1-32tok-100iter-fp16/specdec_cuda_20251030_212942.csv`
   - K=1-4, ~6 tok/s, ~40% acceptance

2. **MPS GPT2/DistilGPT2 (Phase 3C5):**
   - `2025-10-31-MPS-Phase3C5-Baseline/specdec_mps_20251031_113744.csv`
   - `2025-10-31-MPS-Phase3C5-KV-Cache-ON-32tok/specdec_mps_20251031_115236.csv`
   - K=1-4, ~9.5 tok/s, ~18% acceptance

3. **T4 Llama 3.2 (Phase 4A):**
   - `llama32_pair_results/specdec_cuda_20251118_224025.csv` (batch_size=1)
   - `llama32_pair_bs2_results/specdec_cuda_20251118_230854.csv` (batch_size=2)
   - `llama32_pair_bs4_results/specdec_cuda_20251118_230957.csv` (batch_size=4)
   - K=1, ~8.5 tok/s (bs=1), ~5 tok/s (bs=2), ~4.7 tok/s (bs=4)

4. **Baseline (non-speculative):**
   - `baseline_nonspec_results/specdec_cuda_20251118_230536.csv`
   - Llama 3.2-3B only, ~17 tok/s (baseline is 2x faster than speculative for this pair)

**Summary Document:** `docs/results/curated_results.md` - Aggregated results with interpretation

---

## D. Known Issues / Tech Debt

### Correctness Edge Cases

1. **Ring Buffer Overflow:** `kv_cache_manager.py:415` - Hard assertion raises `RuntimeError` if `new_pos > max_seq_len`. No wrap-around logic implemented (intentional for correctness testing). **Fix needed:** Add wrap-around or better max_seq_len estimation.

2. **Partial Acceptance with KV Cache:** `batch_loop.py:320-334` - When `accepted_len > base_tokens.shape[1]`, uses draft tokens instead of base tokens. This is a workaround for correctness testing with `draft_model == base_model`. **Status:** Documented workaround, may need refinement.

3. **Duplication Detection:** `batch_loop.py:349-372` - Skipped for verified tokens in non-deterministic mode, but applied in deterministic mode. Inconsistent behavior. **Fix needed:** Clarify when duplication detection should run.

4. **EOS Token Handling:** `batch_loop.py:374-386` - EOS detection clears all accepted tokens, which may be too aggressive. **Status:** May need refinement for production use.

### Performance Bottlenecks

1. **CUDA Graph Capture Disabled:** `pipeline.py:347-353` - CUDA graph capture removed (incompatible with dynamic speculative decoding). **Impact:** Missing potential speedup from graph capture. **Status:** Documented, no immediate fix planned.

2. **KV Cache Append Overhead:** `pipeline.py:1351-1398` - KV cache append is optional (`SPECDEC_ENABLE_KV_APPEND=1`). When disabled, verification must recompute KV cache. **Status:** Feature flag, but may be a bottleneck if disabled.

3. **Batch Size Scaling:** Results show throughput drops with batch size for Llama 3.2 on T4 (8.5 → 5.0 → 4.7 tok/s). **Investigation needed:** Memory bandwidth, kernel launch overhead, or attention mask computation.

4. **Verification Cost:** `pipeline.py:1280-1295` - Verification generates `k+1` tokens in one forward pass (parallel verification). This is correct but may be slower than expected if prefill is expensive. **Status:** Working as designed, but may need profiling.

### Missing Tests / Flaky Behavior

1. **Integration Tests:** `tests/integrations/test_zero_copy_correctness.py` - Basic structure exists, but may need expansion for edge cases (wrap-around, partial acceptance, etc.).

2. **Stress Tests:** `tests/stress_test_long_run.py` - Exists but may not cover all failure modes (OOM, overflow, etc.).

3. **Determinism Tests:** `tests/test_deterministic_mode.py` - Exists but may not cover all seed combinations or model pairs.

4. **No MPS-Specific Tests:** MPS support is validated through benchmarks but lacks dedicated unit tests.

### Prototype-y Code Patterns

1. **Global State:** `pipeline.py:457-465` - Metrics dictionary stored as instance variable, but reset in `generate()`. **Status:** Acceptable for prototype, but could be refactored to return metrics.

2. **Mixed Concerns:** `pipeline.py:1090-1625` - `generate()` method is 535 lines, handles tokenization, generation loop, metrics, profiling, and cleanup. **Status:** Functional but could be split into smaller methods.

3. **Duplicated Logic:** `batch_loop.py:319-406` vs `pipeline.py:1348-1450` - Similar acceptance/rollback logic in single-prompt and batch paths. **Status:** Acceptable for now, but could be unified.

4. **Environment Variable Spaghetti:** Multiple `os.getenv()` calls throughout codebase (e.g., `SPECDEC_DETERMINISTIC`, `SPECDEC_ENABLE_KV_APPEND`, `SPECDEC_DEBUG`, etc.). **Status:** Functional but hard to track. Consider centralizing in config class.

5. **Error Handling:** Many `try/except` blocks with `pass` or generic logging (e.g., `pipeline.py:1452-1467`). **Status:** Prototype-y, may hide real issues.

---

## E. "Left to Test" Checklist (Aligned to Research Goals)

### Must-Run Baselines

- [ ] **Vanilla greedy decoding baseline** (no speculative decoding)
  - **Command:** `python scripts/comprehensive_k_sweep.py --draft-model none --base-model gpt2 --device cuda --max-tokens 64`
  - **Expected output:** `docs/results/baseline_nonspec_results/` (already exists for Llama, need GPT2)
  - **Metric:** Throughput (tok/s) to compare against speculative decoding

- [ ] **Baseline with same model pair** (draft == base, deterministic)
  - **Command:** `python scripts/comprehensive_k_sweep.py --base-model gpt2 --draft-model gpt2 --deterministic --device cuda`
  - **Expected output:** Should match vanilla baseline exactly (correctness test)
  - **Metric:** Token-by-token parity, throughput should be similar or slightly slower

### Must-Run Ablations

- [ ] **Zero-copy on/off comparison**
  - **Zero-copy ON:** `SPECDEC_ENABLE_KV_APPEND=1 python scripts/comprehensive_k_sweep.py ...`
  - **Zero-copy OFF:** `SPECDEC_ENABLE_KV_APPEND=0 python scripts/comprehensive_k_sweep.py ...`
  - **Expected output:** Compare throughput, memory usage, acceptance rates
  - **Files to check:** `kv_cache_manager.py` (ring buffer) vs fallback path in `pipeline.py:1351-1398`

- [ ] **K sweep (1, 2, 4, 8, 16)**
  - **Command:** `python scripts/comprehensive_k_sweep.py --max-k 16 --base-model gpt2 --draft-model distilgpt2 --device cuda`
  - **Expected output:** CSV with throughput vs K, acceptance rate vs K
  - **Analysis:** Identify optimal K, verify constant-time verification claim

- [ ] **Batch size sweep (1, 2, 4, 8, 16)**
  - **Command:** `SPECDEC_BATCH_SIZE=1 python scripts/comprehensive_k_sweep.py ...` (repeat for 2, 4, 8, 16)
  - **Expected output:** Throughput vs batch_size, acceptance rate vs batch_size
  - **Analysis:** Identify batch size sweet spot, investigate throughput drop at larger batch sizes

- [ ] **Sequence length sweep (32, 64, 128, 256, 512)**
  - **Command:** `python scripts/comprehensive_k_sweep.py --max-tokens 32 ...` (repeat for 64, 128, 256, 512)
  - **Expected output:** Throughput vs seq_len, memory usage vs seq_len
  - **Analysis:** Identify memory limits, verify ring buffer doesn't overflow

### Acceptance Rate Sensitivity Tests

- [ ] **Perfect draft (draft == base) vs realistic draft (draft != base)**
  - **Perfect:** `--base-model gpt2 --draft-model gpt2 --deterministic`
  - **Realistic:** `--base-model gpt2 --draft-model distilgpt2`
  - **Expected output:** Compare acceptance rates, throughput impact
  - **Analysis:** Quantify acceptance rate impact on throughput

- [ ] **Different acceptance policies**
  - **Command:** Modify `pipeline.py:249` or use CLI `--policy longest_prefix|conf_threshold|topk_agree|typical`
  - **Expected output:** Acceptance rate vs policy, throughput vs policy
  - **Analysis:** Identify best policy for model pair

### Memory/VRAM Behavior Tests

- [ ] **VRAM usage vs batch_size**
  - **Command:** Run batch size sweep with `SPECDEC_DETAILED_METRICS=1`, check `cuda_mem_peak_mb` in JSON results
  - **Expected output:** Memory usage curve, identify OOM threshold
  - **Files to check:** `src/metrics/memory_profiler.py`, JSON results include `cuda_mem_peak_mb`

- [ ] **Ring buffer memory footprint**
  - **Command:** Log `kv_cache_manager.get_kv_shapes()` at initialization
  - **Expected output:** Buffer sizes, verify pre-allocation is correct
  - **Files to check:** `kv_cache_manager.py:724` (`get_kv_shapes()`)

- [ ] **Memory fragmentation test (consecutive runs)**
  - **Command:** Run K-sweep multiple times in same process, check memory growth
  - **Expected output:** Memory should stabilize (ring buffer reuse)
  - **Files to check:** `kv_cache_manager.py:82-135` (`reset()` preserves buffers)

### Correctness Tests

- [ ] **Token-by-token parity (deterministic mode)**
  - **Command:** `SPECDEC_DETERMINISTIC=1 python scripts/comprehensive_k_sweep.py --base-model gpt2 --draft-model gpt2 --device cpu`
  - **Expected output:** Should match vanilla baseline exactly
  - **Files to check:** `tests/test_deterministic_mode.py`, `tests/test_specdec_cpu_correctness.py`

- [ ] **Determinism across seeds**
  - **Command:** Run same prompt with different seeds, verify output is deterministic
  - **Expected output:** Same output for same seed, different output for different seeds
  - **Files to check:** `pipeline.py:306-334` (`_set_seed()`, deterministic mode setup)

- [ ] **Seed control**
  - **Command:** `--seed 42 python scripts/comprehensive_k_sweep.py ...` (verify results are reproducible)
  - **Expected output:** Identical results across runs with same seed
  - **Files to check:** `pipeline.py:306-334`, `comprehensive_k_sweep.py` (seed handling)

- [ ] **KV cache alignment verification**
  - **Command:** Enable `SPECDEC_DEBUG=1`, check KV cache checksums in logs
  - **Expected output:** KV cache should match between speculative and vanilla paths
  - **Files to check:** `kv_cache_verification.py:63` (`verify_kv_cache_alignment()`)

---

## F. "Start Here Tomorrow" Plan

### Exact Commands to Run

**1. Smoke Test (Verify System Works):**
```bash
# Activate environment (if using venv)
source env/bin/activate  # or your venv path

# Run smoke test
python scripts/dev/smoke_cuda.py  # or for MPS: python scripts/comprehensive_k_sweep.py --device mps --max-k 1 --iterations 1
```

**2. Minimal Reproduction (Latest Benchmark Suite):**
```bash
# Set environment variables (from README.md:152-167)
export SPECDEC_AMP=1
export SPECDEC_DTYPE=float16
export SPECDEC_DETAILED_METRICS=1
export SPECDEC_DETERMINISTIC=1
export SPECDEC_PROFILE=1
export SPECDEC_CUDA_GRAPH=1  # Note: disabled in code, but env var exists
export SPECDEC_PARALLEL_STREAMS=1
export SPECDEC_SYNC_MODE=event

# Run comprehensive K-sweep (GPT2/DistilGPT2 on T4, Phase 3D config)
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 \
  --draft-model distilgpt2 \
  --max-tokens 64 \
  --iterations 100 \
  --device cuda \
  --deterministic \
  --output-dir docs/results/2025-01-XX-T4-Reproduction \
  --no-plots
```

**3. Baseline Comparison (Non-Speculative):**
```bash
# Run baseline (no draft model)
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 \
  --draft-model none \
  --max-tokens 64 \
  --iterations 100 \
  --device cuda \
  --output-dir docs/results/2025-01-XX-T4-Baseline \
  --no-plots
```

**4. Quick K-Sweep (Faster, for Testing):**
```bash
# Reduced iterations for quick testing
python scripts/comprehensive_k_sweep.py \
  --base-model gpt2 \
  --draft-model distilgpt2 \
  --max-tokens 32 \
  --iterations 10 \
  --max-k 4 \
  --device cuda \
  --output-dir docs/results/2025-01-XX-T4-QuickTest \
  --no-plots
```

### Output Files to Expect

**CSV File:** `docs/results/YYYY-MM-DD-*/specdec_cuda_YYYYMMDD_HHMMSS.csv`
- Columns: `k`, `n_samples`, `n_failures`, `success_rate`, `latency_ms_mean`, `latency_ms_std`, `tokens_per_sec_mean`, `tokens_per_sec_std`, `acceptance_rate_mean`, `acceptance_rate_std`, `proposed_mean`, `proposed_std`, `accepted_mean`, `accepted_std`
- **Interpretation:** 
  - `tokens_per_sec_mean`: Throughput (higher is better)
  - `acceptance_rate_mean`: Fraction of draft tokens accepted (higher is better, typically 0.3-0.5 for realistic pairs)
  - `success_rate`: Fraction of runs that completed without errors (should be 1.0)

**JSON File:** `docs/results/YYYY-MM-DD-*/specdec_cuda_YYYYMMDD_HHMMSS.json`
- Structure: `{"system_info": {...}, "summary_results": [...], "detailed_results": [...], "detailed_metrics": {...}}`
- **Key fields:**
  - `system_info.dtype`: Data type used (float16/float32)
  - `system_info.kernel_backends`: Which kernels are active (`verify_backend`, `kv_append_backend`)
  - `summary_results`: Aggregated per-K results (same as CSV)
  - `detailed_results`: Per-iteration results (for debugging)
  - `detailed_metrics`: Profiling data (if `SPECDEC_DETAILED_METRICS=1`)

**Console Output:**
- Summary table printed at end (K, Samples, Failures, Success%, Latency, Throughput, Accept Rate, Proposed, Accepted)
- Example: `K=1: 100 samples, 0 failures, 100.0% success, 178.5±12.3ms latency, 5.6±0.4 tok/s, 0.40±0.05 acceptance`

### What to Fix First if Throughput < Baseline on T4

**1. Check Kernel Backends:**
```bash
# Verify CUDA kernels are loaded
python -c "from kernels import get_kernel_info; print(get_kernel_info())"
# Expected: {"verify_backend": "cuda", "kv_append_backend": "cuda"} or similar
# If "pytorch" or "unavailable", kernels may not be compiled
```

**2. Check CUDA Streams:**
```bash
# Verify streams are enabled (should NOT see this warning)
# If you see: "CUDA_LAUNCH_BLOCKING=1 is set - async streams will be disabled!"
# Then: unset CUDA_LAUNCH_BLOCKING
```

**3. Check KV Cache Append:**
```bash
# Verify KV cache append is enabled
export SPECDEC_ENABLE_KV_APPEND=1
# Re-run benchmark, check JSON for "kv_append_enabled": true
```

**4. Profile Hot Paths:**
```bash
# Enable detailed profiling
export SPECDEC_DETAILED_METRICS=1
export SPECDEC_PROFILE=1
# Re-run, check JSON "detailed_metrics" section for timing breakdown
# Look for: draft_time_ms, verify_time_ms, kv_append_time_ms
```

**5. Check Memory Bandwidth:**
```bash
# Check if memory is bottleneck (T4 has limited bandwidth)
# Look at JSON "cuda_mem_peak_mb" - if > 14GB, may be hitting limits
# Consider: reduce batch_size, reduce max_seq_len, use float16
```

**6. Verify Parallel Verification:**
```bash
# Check that verify_time_ms is constant across K (should not grow linearly)
# In JSON "detailed_results", compare verify_time_ms for K=1 vs K=4
# If K=4 verify_time is ~4x K=1, parallel verification may not be working
```

**7. Check Acceptance Rate:**
```bash
# Low acceptance rate (< 0.3) means most draft tokens are rejected
# This defeats the purpose of speculative decoding
# Fix: Use better model pair (e.g., GPT2 + DistilGPT2 should be ~0.4)
# Or: Adjust temperature/sampling to improve alignment
```

**8. Common Issues:**
- **Throughput < baseline:** Normal for some model pairs (see Llama 3.2 results: baseline 17 tok/s, speculative 8.5 tok/s)
- **OOM errors:** Reduce `max_seq_len` (currently clamped to 4096 in `pipeline.py:509-562`)
- **Kernel errors:** Check CUDA version compatibility, may need to rebuild kernels (`src/kernels/build.py`)

### Where Commands Are Defined

- **Main benchmark script:** `scripts/comprehensive_k_sweep.py` (line 107: `main()`)
- **K-sweep runner:** `scripts/k_sweep/runner.py` (line ~100: `run_comprehensive_k_sweep()`)
- **Result serialization:** `scripts/k_sweep/results.py` (line 17: `save_results()`)
- **CLI commands:** `src/specdec_cli/main.py` (line 98: `main()`, line 10: `cmd_bench()`)
- **Direct API:** `src/specdec/run_specdec.py` (line 188: `main()`)

---

## G. File Reference Quick Index

**Entry Points:**
- `scripts/comprehensive_k_sweep.py` - Main benchmark script
- `src/specdec_cli/main.py` - CLI (`specdec bench`, `specdec run`)
- `src/specdec/run_specdec.py` - Direct API entrypoint

**Core Generation:**
- `src/specdec/core/pipeline.py` - `SpeculativePipeline` (single-prompt)
- `src/specdec/core/batch_loop.py` - `BatchGenerationLoop` (batch)
- `src/specdec/core/batch_handlers.py` - Handler classes (draft, verify, accept, rollback, KV)

**KV Cache:**
- `src/specdec/cache/kv_cache_manager.py` - `SafeKVCacheManager` (ring buffer)
- `src/specdec/core/kv_cache_verification.py` - Correctness checks

**Kernels:**
- `src/kernels/registry.py` - Kernel registry
- `src/kernels/cuda/verify.cu` - CUDA verification kernel
- `src/kernels/cuda/kv_cache.cu` - CUDA KV append kernel

**Results:**
- `scripts/k_sweep/results.py` - CSV/JSON serialization
- `docs/results/curated_results.md` - Aggregated results summary
- `docs/results/` - All benchmark result files

**Tests:**
- `tests/test_deterministic_mode.py` - Determinism tests
- `tests/test_specdec_cpu_correctness.py` - Correctness tests
- `tests/integrations/test_zero_copy_correctness.py` - Zero-copy tests

**Documentation:**
- `README.md` - High-level overview
- `docs/design/zero_copy_design.md` - Architecture design (placeholder)
- `docs/results/curated_results.md` - Results summary
- `progress/progress.md` - Research progress

---

**END OF REPORT**

