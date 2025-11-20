# Curated Experimental Results

This document collects a minimal set of results that illustrate the behaviour of the zero copy speculative decoding engine. Detailed logs and additional runs are preserved in the raw results folders and in the engineering appendix.

## 1. GPT2 plus DistilGPT2

### 1.1 Apple Silicon MPS

Configuration:

1. Base model: GPT2
2. Draft model: DistilGPT2
3. Device: Apple Silicon MPS
4. Sequence length: thirty two tokens
5. Multiple K values from one to four

Representative results:

| K value | Throughput (tok/s) | Acceptance rate (percent) |
|--------:|--------------------:|---------------------------:|
| 1       | about 9.5           | about 18                   |
| 2       | about 9.4           | about 18                   |
| 3       | about 8.9           | about 16                   |
| 4       | about 9.5           | about 18                   |

Throughput remains close to ten tokens per second across K and the acceptance rates are moderate. The runs are stable with no out of memory failures.

### 1.2 Tesla T4

Configuration:

1. Base model: GPT2
2. Draft model: DistilGPT2
3. Device: Tesla T4
4. Sequence length: sixty four tokens
5. K values from one to four

Representative results:

| K value | Throughput (tok/s) | Acceptance rate (percent) |
|--------:|--------------------:|---------------------------:|
| 1       | about 5.6           | about 40                   |
| 2       | about 5.9           | about 40                   |
| 3       | about 6.0           | about 40                   |
| 4       | about 6.0           | about 40                   |

All runs complete successfully with one hundred percent success rate. Acceptance remains stable near forty percent and the structured profiler confirms that the verify path behaves as expected.

## 2. Llama 3.2 on Tesla T4

Configuration:

1. Base model: Llama 3.2 three billion parameters
2. Draft model: Llama 3.2 one billion parameters
3. Device: Tesla T4
4. Sequence length: sixty four tokens for batch size one and two, thirty two tokens for batch size four
5. K equals one
6. Key value cache append disabled in the reported runs

Representative results:

| Batch size | Max tokens | Throughput (tok/s) | Acceptance rate (fraction) |
|-----------:|-----------:|--------------------:|----------------------------:|
| 1          | 64         | about 8.5           | about 0.86                  |
| 2          | 64         | about 5.0           | about 0.65                  |
| 4          | 32         | about 4.7           | about 0.62                  |

A non speculative baseline with the Llama 3.2 three billion model alone reaches roughly seventeen tokens per second at batch size one on T4. This makes the baseline about twice as fast as the speculative configuration for this model pair and hardware.

The results show that speculative decoding is not always a win on small GPUs, and that batch size and model choice interact strongly with hardware limits.

## 3. Constant Time Verification Observation

Using the parallel verification design, verification time for K between one and six remains approximately constant for the tested GPT2 and Llama pairs, while earlier naive implementations showed verification cost that grew roughly linearly with K. This behaviour matches the intended prefill based design and validates the zero copy verification path.

