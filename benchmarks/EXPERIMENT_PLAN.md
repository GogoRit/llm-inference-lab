# Experiment Plan

## Overview

This document outlines the systematic benchmarking plan for evaluating speculative decoding performance across different configurations.

## Experiment Matrix

### GPT-2 Experiments (Primary)

**Target Devices:** T4 (CUDA), Mac (MPS)

**Models:**
1. `gpt2` → `gpt2` (perfect draft, deterministic correctness baseline)
2. `distilgpt2` → `gpt2` (realistic draft → target pair)

**Grid Parameters:**
- `batch_size`: [1, 2, 4, 8]
- `k` (draft tokens): [1, 2, 4, 8]
- `max_tokens`: 64
- `num_prompts`: 32
- `repeats`: 3 (for statistical significance)

**Total Configs:** 2 models × 4 batch sizes × 4 k values × 3 repeats = 96 runs per device

### Tiny Model Experiments (CPU Sanity)

**Target Device:** CPU

**Models:**
1. `sshleifer/tiny-gpt2` → `sshleifer/tiny-gpt2` (perfect draft)

**Grid Parameters:**
- `batch_size`: [1, 2, 4]
- `k`: [1, 2, 4]
- `max_tokens`: 64
- `num_prompts`: 32
- `repeats`: 2

**Total Configs:** 1 model × 3 batch sizes × 3 k values × 2 repeats = 18 runs

## Success Criteria

### Performance Targets

1. **Speedup > 1.0**: SpecDec should be faster than vanilla in at least some configurations
2. **Acceptance Rate**: Higher acceptance rates should correlate with better speedups
3. **Batch Size Scaling**: Performance should improve with larger batch sizes
4. **K Scaling**: Optimal k value should be identified (likely k=4 or k=8)

### Analysis Questions

1. At what (batch_size, k) combinations does SpecDec win?
2. How does speedup scale with k at fixed batch_size?
3. How does speedup scale with batch_size at fixed k?
4. What is the relationship between acceptance_rate and speedup?
5. Are there device-specific differences (T4 vs MPS)?

## Next Steps After Benchmarks

1. **If speedup < 1.0 everywhere:**
   - Add profiling harness to identify bottlenecks
   - Profile KV-cache operations, draft vs verify passes
   - Optimize hot paths (ring buffer indexing, verification, syncs)

2. **If speedup > 1.0 in some configs:**
   - Document winning configurations
   - Analyze why those configs win
   - Consider expanding grid to find optimal sweet spot

3. **Analysis:**
   - Use `scripts/analyze_benchmarks.py` to interpret results
   - Generate visualizations if needed (future work)
   - Document findings in progress.md

## Running Experiments

See `benchmarks/README.md` for detailed commands to run each experiment set.





