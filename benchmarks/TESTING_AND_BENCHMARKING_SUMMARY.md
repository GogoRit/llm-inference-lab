# Testing and Benchmarking Summary

## Overview
This document summarizes the testing and benchmarking work completed after fixing the duplication detection issue that was filtering verified tokens.

## Tests Added

### New Test File: `tests/test_duplication_detection.py`

Comprehensive test suite covering duplication detection behavior:

1. **`test_verified_tokens_not_filtered_in_non_deterministic_mode`**
   - Documents that verified tokens should not be filtered
   - Tests handler behavior when called directly

2. **`test_perfect_draft_repetitive_tokens_high_acceptance_rate`**
   - Integration test with real models
   - Verifies ~100% acceptance rate for perfect draft in non-deterministic mode
   - Ensures total tokens match expected counts

3. **`test_unverified_repetitive_tokens_filtered`**
   - Verifies that unverified repetitive tokens are still filtered
   - Tests phrase repetition and single token repetition detection

4. **`test_deterministic_mode_bypasses_duplication_detection`**
   - Confirms deterministic mode completely bypasses duplication detection
   - Ensures all tokens are returned unchanged

5. **`test_perfect_draft_matches_vanilla_with_repetition`**
   - End-to-end integration test
   - Compares deterministic vs non-deterministic modes
   - Verifies both modes produce correct results

### Test Results
✅ **All 5 tests pass** in `test_duplication_detection.py`
✅ **All 5 tests pass** in `test_deterministic_mode.py`
✅ **Total: 10 tests passing**

## Code Cleanup

### Debug Logging Cleanup
Downgraded temporary investigation logging from INFO to DEBUG level:

- `batch_loop.py`: Shape logging, policy results, duplication check logging
- `batch_handlers.py`: Verification handler output shapes, duplication detection details

All debug logging is now properly gated behind `SPECDEC_DEBUG_PRINTS` environment variable and uses DEBUG level.

## Benchmark Results

### CPU Benchmarks (Tiny GPT-2)

**Configuration:**
- Model: `sshleifer/tiny-gpt2` (perfect draft: draft==base)
- Device: CPU
- Implementation: torch
- Grid: batch_size=[1, 2, 4], k=[1, 2, 4], max_tokens=64, num_prompts=32
- Repeats: 2

**Key Results:**

| batch_size | k | vanilla_tok/s | specdec_tok/s | speedup | acceptance_rate |
|------------|---|---------------|---------------|---------|-----------------|
| 1 | 1 | 1074.77 | 197.50 | 0.18 | **1.000** ✅ |
| 1 | 2 | 1074.40 | 281.66 | 0.26 | **1.000** ✅ |
| 1 | 4 | 1065.75 | 385.39 | 0.36 | **1.000** ✅ |
| 2 | 1 | 1707.68 | 334.36 | 0.20 | **1.000** ✅ |
| 2 | 2 | 1717.59 | 458.10 | 0.27 | **1.000** ✅ |
| 2 | 4 | 1716.85 | 622.87 | 0.36 | **1.000** ✅ |
| 4 | 1 | 3149.29 | 595.33 | 0.19 | **1.000** ✅ |
| 4 | 2 | 3143.58 | 866.61 | 0.28 | **1.000** ✅ |
| 4 | 4 | 3158.46 | 1134.52 | 0.36 | **1.000** ✅ |

**Observations:**
- ✅ **Acceptance rate: 100%** across all configurations (perfect!)
- ✅ **Total tokens match**: `total_tokens_vanilla == total_tokens_specdec == 2048` for all configs
- ⚠️ **Speedup < 1.0 on CPU**: Expected due to CPU overhead (0.18-0.36x)
- 📈 **Speedup increases with k**: Higher k values show better relative performance (0.18 → 0.36)

**Pattern Analysis:**
- Speedup vs k: Consistent improvement with larger k (0.18-0.19 at k=1 → 0.36 at k=4)
- Speedup vs batch_size: Minimal variation (0.18-0.20 at batch_size=1 → 0.19-0.36 at batch_size=4)
- Best speedup: 0.36x at batch_size=2, k=4 (and batch_size=4, k=4)

## Key Achievements

1. ✅ **Fixed duplication detection bug**: Verified tokens are no longer incorrectly filtered
2. ✅ **100% acceptance rate**: Perfect draft now achieves 100% acceptance (was 12%)
3. ✅ **Comprehensive test coverage**: 5 new tests covering all duplication detection scenarios
4. ✅ **Code cleanup**: Debug logging properly downgraded to DEBUG level
5. ✅ **Benchmark infrastructure**: Grid runner and analysis tools working correctly

## Next Steps

### Ready for GPU Benchmarks
The infrastructure is now ready for GPU benchmarks where speedup should be more significant:

1. **MPS (Mac)**: Run `python -m scripts.run_benchmark_grid --device mps --impl torch --output_csv benchmarks/raw/mps_gpt2_grid.csv --repeats 3 --experiment_set gpt2`
2. **CUDA (T4)**: Run `python -m scripts.run_benchmark_grid --device cuda --impl torch --output_csv benchmarks/raw/t4_gpt2_grid.csv --repeats 3 --experiment_set gpt2`

### Expected GPU Results
- Acceptance rate should remain ~100% for perfect draft
- Speedup should be > 1.0 on GPU (where parallelization benefits outweigh overhead)
- Higher k values should show better speedup on GPU

## Contract Documentation

### Verified Tokens Contract
**"Verified tokens should not be filtered by duplication detection in non-deterministic mode."**

- Tokens accepted by the acceptance policy (draft matches base) are "verified"
- These tokens are legitimate model output, even if repetitive
- Duplication detection is skipped for verified tokens in non-deterministic mode
- This ensures high acceptance rates for perfect draft scenarios

### Deterministic Mode Contract
**"Deterministic mode bypasses duplication detection completely."**

- When `deterministic_mode=True`, all duplication checks are disabled
- This ensures specdec output matches vanilla decoding exactly
- Used for correctness testing and debugging

### Unverified Tokens Contract
**"Unverified repetitive tokens should still be filtered."**

- If draft proposes repetitive tokens that the base model doesn't confirm, they should be filtered
- This prevents generation artifacts and maintains quality
- Duplication detection still works for unverified tokens

## Files Modified

1. **`tests/test_duplication_detection.py`** (new): Comprehensive duplication detection tests
2. **`src/specdec/core/batch_loop.py`**: Skip duplication detection for verified tokens
3. **`src/specdec/core/batch_handlers.py`**: Cleanup debug logging
4. **`benchmarks/raw/cpu_tiny_grid.csv`**: CPU benchmark results
5. **`benchmarks/raw/cpu_tiny_grid_aggregated.csv`**: Aggregated CPU results

## Conclusion

The duplication detection fix is complete and verified:
- ✅ All tests passing
- ✅ 100% acceptance rate for perfect draft
- ✅ Code cleaned up
- ✅ Benchmarks running correctly
- 📝 Ready for GPU benchmarks to see actual speedup

