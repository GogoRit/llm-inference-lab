# LLM Inference Lab Progress Report

This document summarizes the research progress of the zero copy speculative decoding engine. It is intended for collaborators, mentors, and reviewers who want a high level view of what has been implemented, what has been measured, and what remains open.

The older detailed chronological log has been preserved in docs/engineering/appendix_engineering.md.

## 1. Executive Summary

The project studies speculative decoding under tight memory budgets, with a strong focus on Tesla T4 and Apple Silicon devices. Instead of attempting to match massive multi GPU systems, the goal is to understand how much speedup and stability can be obtained by changing the memory layout and cache management of a speculative decoder.

The main achievements so far are:

1. Design and implementation of a zero copy key value cache based on a pre allocated ring buffer.

2. Implementation of pointer based rollback for rejected draft tokens with constant time cost.

3. Implementation of a parallel verification path that keeps verification latency approximately constant in the number of draft tokens K.

4. Integration with Hugging Face models and validation on both MPS and CUDA backends.

5. A structured profiling and metrics system that records per step latencies, acceptance rates, and memory usage.

Experiments on GPT2 plus DistilGPT2 and on Llama 3.2 model pairs show that speculative decoding behaviour is highly sensitive to the model pair, to batch size, and to hardware constraints. On Tesla T4, the zero copy design avoids out of memory failures and reduces overhead caused by padding and tensor concatenation.

## 2. Technical Contributions

This section summarizes the main technical ideas without going into implementation details.

### 2.1 Zero Copy Key Value Cache

The decoder allocates a fixed size memory arena for key and value tensors at initialization time. Each sequence is represented by integer indices into this arena that track the current valid prefix. New tokens extend the prefix by writing into the next positions in the arena, and rejection does not require moving any memory.

This design removes repeated tensor allocation and concatenation and avoids the need for unpad, append, and repad cycles that appear in many ragged tensor handling schemes.

### 2.2 Pointer Based Rollback

Speculative decoding often rejects some portion of the draft block. In the proposed design, rollback updates only the integer that represents the current sequence length. The invalidated positions remain in the arena but are masked out.

The cost of rollback no longer grows with the number of rejected tokens. This constant time rollback makes it feasible to explore more aggressive draft strategies without paying a large memory manipulation cost.

### 2.3 Parallel Verification with Constant K Latency

Verification of K draft tokens is implemented as a single prefill forward pass of the base model. The draft tokens are appended to the prompt, and the model processes the entire prefix in one call before generating the next bonus token.

In practice this produces verification latencies that remain stable across K in the tested regimes, subject to model size and cache use. This behaviour differs from naive implementations where verification cost grows approximately linearly with K.

### 2.4 Metrics, Determinism, and Kernel Registry

The engine includes:

1. A deterministic seeding utility that sets seeds for Python, NumPy, and PyTorch and can be enabled through an environment flag.

2. A structured profiler that collects timing information for draft, verify, accept, and cache operations and can export JSON logs.

3. A kernel registry that chooses between CUDA, Triton, and PyTorch implementations so that every operation has a safe fallback.

These components make it possible to run controlled sweeps across K, batch size, and model configurations and to compare results across hardware.

## 3. Experimental Summary

This section collects the main experimental findings in a compact form. The full detailed tables and raw logs live in docs/results and in the engineering appendix.

### 3.1 GPT2 plus DistilGPT2

On Apple Silicon MPS, the GPT2 base model with a DistilGPT2 draft model reaches throughput on the order of around ten tokens per second for short sequences of approximately thirty two tokens. Acceptance rates are in the range of fifteen to twenty percent for the tested K values.

On Tesla T4, the same pair reaches around seventeen tokens per second for short sequences under a speculative decoding configuration with K between one and four. At longer sequences of sixty four tokens the throughput stabilizes around six tokens per second with acceptance near forty percent and a one hundred percent success rate across hundreds of runs.

These results confirm that the implementation is stable across devices and that the structured profiling and seeding logic work as intended.

### 3.2 Llama 3.2 on Tesla T4

For the Llama 3.2 three billion parameter base model with a one billion parameter draft model, speculative decoding on Tesla T4 exhibits a distinct pattern.

1. At batch size one, speculative decoding achieves roughly eight and a half tokens per second with high acceptance rates near eighty five percent.

2. At batch size two and four, throughput drops to around five tokens per second and acceptance falls into the range of sixty to sixty five percent.

3. A non speculative baseline with the Llama 3.2 three billion model alone reaches approximately seventeen tokens per second at batch size one.

These findings indicate that on T4, speculative decoding for this model pair does not outperform a carefully tuned non speculative baseline at the same batch size. They also show that increasing batch size in this setting can harm both acceptance and throughput, which is consistent with theoretical concerns about verify overhead and memory pressure on small GPUs.

### 3.3 Constant Time Verification Behaviour

Using the parallel verification path with prefill, verification latency remains approximately flat when K is increased for moderate K values. This confirms that the implementation matches the intended design where additional draft tokens are processed as part of a single prompt extension rather than in separate autoregressive steps.

This property is especially important for memory constrained hardware because it prevents verify cost from exploding when K grows within the tested range.

## 4. Current Limitations

Several limitations and open issues remain.

1. Hugging Face cache integration

   The zero copy arena must interoperate correctly with the cache structures expected by Hugging Face models. For some models, the interaction between static buffers, position ids, and internal cache assumptions still requires careful validation.

2. Throughput on large models

   On Tesla T4, speculative decoding with Llama 3.2 does not yet outperform the non speculative baseline at batch size one. The overhead of running both draft and base models and of managing the speculative accept loop dominates. Further work is needed to understand whether larger GPUs or different model pairs benefit more from the proposed memory layout.

3. Batch size scaling

   For the tested Llama 3.2 configuration, larger batch sizes reduce throughput and acceptance. This suggests that straightforward batching is not always beneficial in speculative decoding on small GPUs and that more advanced scheduling and dynamic batching strategies are required.

4. Limited hardware diversity

   The current experiments focus on Apple Silicon MPS and Tesla T4. Planned work includes A100 and H100 validation to understand how the zero copy design behaves when memory and compute are less constrained.

## 5. Next Milestones

Planned next steps include:

1. Finish a formal specification of the correctness invariants for the zero copy ring buffer and its interaction with position ids and attention masks, and incorporate this into docs/design/zero_copy_design.md.

2. Refine Hugging Face cache integration so that more models can use the static arena without manual adjustments.

3. Implement and evaluate dynamic batching strategies that group sequences by length or acceptance behaviour in order to improve utilization.

4. Extend experimental coverage to larger GPUs such as A100 and H100 and compare the observed overhead patterns to the behaviour on T4 and MPS.

5. Prepare a concise research write up that positions this work relative to existing speculative decoding systems, with a particular focus on memory constrained hardware.

