# Zero Copy Speculative Decoding Design

This document describes the zero copy ring buffer architecture, pointer based rollback, and the interaction with key value caches, position ids, and attention masks.

It is intended as a formal design note that complements the higher level descriptions in README.md and progress/progress.md.

> **Note**: This is a placeholder document. The full design is currently documented in the codebase and README. This document will be expanded with formal specifications in future work.

## Planned Sections

1. Problem statement and relation to ragged tensor handling.

2. Ring buffer memory layout and indexing scheme.

3. Pointer based rollback and invalidation semantics.

4. Integration with Hugging Face cache structures.

5. Correctness invariants and informal proofs.

6. Interaction with parallel verification and acceptance logic.

## Current Implementation

The zero-copy ring buffer is implemented in:
- `src/specdec/cache/kv_cache_manager.py` - SafeKVCacheManager class
- `src/specdec/core/batch_handlers.py` - RollbackHandler for pointer-based rollback
- `src/specdec/core/batch_loop.py` - BatchGenerationLoop orchestrates the pipeline

Key features:
- Pre-allocated ring buffer for KV cache (no per-step allocation)
- Pointer-based rollback (O(1) cost vs O(K) tensor operations)
- Parallel verification (constant K latency via prefill)
- CUDA streams for overlapped draft/verify execution

