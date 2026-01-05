# Correctness Invariants for Speculative Decoding

This document spells out the invariants that must hold for speculative decoding to be correct, both in the standard setting and in the zero copy design.

> **Note**: This is a placeholder document. The correctness properties are currently validated through tests and verified in the implementation. This document will be expanded with formal invariants in future work.

## Planned Sections

1. Alignment of tokens, position ids, and attention masks between draft and base models.

2. Consistency of key value cache states across draft acceptance and rejection.

3. Behaviour of the system under different K values and batch sizes.

4. Conditions under which the zero copy ring buffer is observationally equivalent to a naive dynamic allocation scheme.

5. Known failure modes and how the implementation avoids them.

## Current Correctness Guarantees

The implementation maintains correctness through:

1. **Output Equivalence**: Same acceptance policy (longest prefix) and verification logic as standard speculative decoding.

2. **Position ID Alignment**: Position IDs tracked via sequence length pointers, attention masks match current_seq_lens.

3. **KV Cache Consistency**: Ring buffer tracks valid prefix via pointers; rejection = O(1) pointer rollback.

4. **Bonus Token Handling**: Bonus token included in accepted tokens, KV cache updated correctly.

## Testing

Correctness is validated through:
- `tests/integrations/test_zero_copy_correctness.py` - Core correctness tests
- `tests/specdec/test_pipeline_consolidated.py` - Integration tests
- Experimental validation on GPT2/DistilGPT2 and Llama 3.2 model pairs

