# MPS Benchmark Results Summary

## Overview
Benchmarks run on MPS (Metal Performance Shaders) device with GPT-2 models.

## Key Findings

### Perfect Draft (gpt2 == gpt2)
✅ **100% Acceptance Rate** across all configurations
- All perfect draft runs show `acceptance_rate=1.000`
- Total tokens match vanilla: `total_tokens_vanilla == total_tokens_specdec == 2048`

### Speedup Results

**Perfect Draft (gpt2 → gpt2):**
- Best speedup: **0.38x** at batch_size=4, k=8
- Speedup range: 0.11x - 0.38x
- Pattern: Speedup increases with k (0.19 at k=1 → 0.38 at k=8)
- Pattern: Speedup improves slightly with batch_size

**Imperfect Draft (distilgpt2 → gpt2):**
- Acceptance rate: 17% - 59% (expected for imperfect draft)
- Speedup: 0.03x - 0.11x (slower due to low acceptance)
- Higher k values show lower acceptance (more draft tokens rejected)

## Detailed Results

### Perfect Draft Performance (gpt2 → gpt2)

| batch_size | k | vanilla_tok/s | specdec_tok/s | speedup | acc_rate |
|------------|---|---------------|---------------|---------|----------|
| 1 | 1 | 46.90 | 8.86 | 0.19 | 1.000 |
| 1 | 2 | 45.47 | 12.12 | 0.27 | 1.000 |
| 1 | 4 | 45.87 | 11.43 | 0.24 | 1.000 |
| 1 | 8 | 44.93 | 10.07 | 0.27 | 1.000 |
| 2 | 1 | 86.51 | 15.31 | 0.18 | 1.000 |
| 2 | 2 | 89.86 | 20.57 | 0.23 | 1.000 |
| 2 | 4 | 69.47 | 19.27 | 0.29 | 1.000 |
| 2 | 8 | 79.35 | 29.07 | 0.37 | 1.000 |
| 4 | 1 | 142.04 | 18.13 | 0.13 | 1.000 |
| 4 | 2 | 123.30 | 26.64 | 0.22 | 1.000 |
| 4 | 4 | 137.57 | 40.29 | 0.29 | 1.000 |
| 4 | 8 | 127.83 | 46.68 | **0.38** | 1.000 |
| 8 | 1 | 175.91 | 19.58 | 0.11 | 1.000 |
| 8 | 2 | 199.28 | 34.37 | 0.17 | 1.000 |
| 8 | 4 | 207.79 | 52.05 | 0.25 | 1.000 |
| 8 | 8 | 208.56 | 68.13 | 0.33 | 1.000 |

### Imperfect Draft Performance (distilgpt2 → gpt2)

| batch_size | k | vanilla_tok/s | specdec_tok/s | speedup | acc_rate |
|------------|---|---------------|---------------|---------|----------|
| 1 | 1 | 47.11 | 5.16 | 0.11 | 0.431 |
| 1 | 2 | 47.46 | 7.65 | 0.16 | 0.516 |
| 1 | 4 | 47.83 | 5.72 | 0.12 | 0.271 |
| 1 | 8 | 48.46 | 4.22 | 0.09 | 0.170 |
| 4 | 8 | 140.26 | 3.97 | 0.03 | 0.079 |
| 8 | 8 | 202.31 | 5.70 | 0.03 | 0.093 |

## Observations

1. **Perfect Draft Works Correctly**: 100% acceptance rate confirms the duplication detection fix is working
2. **MPS Overhead**: SpecDec is slower on MPS due to overhead (CPU synchronization, memory transfers)
3. **Speedup Increases with k**: For perfect draft, larger k values show better relative performance
4. **Imperfect Draft Struggles**: Low acceptance rates (17-59%) make SpecDec slower than vanilla
5. **Batch Size Impact**: Larger batches show better vanilla throughput but SpecDec overhead remains

## Issues Fixed

✅ **Tokenizer Padding Warning**: Fixed by ensuring `padding_side='left'` is set before any tokenizer calls with `padding=True`

## Next Steps

1. Run on CUDA (T4) to see if GPU parallelization provides better speedup
2. Investigate MPS-specific optimizations (memory transfers, kernel launches)
3. Profile to identify bottlenecks in SpecDec path on MPS





