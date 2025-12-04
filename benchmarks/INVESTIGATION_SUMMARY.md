# Batch Path Low Acceptance Rate - Investigation Summary

## Problem
Batch benchmarks showed 12% acceptance rate for perfect draft (draft==base), while single-generation diagnostic showed 100% acceptance.

## Investigation Process

### Step 1: Verified Verification Fix
- ✅ Confirmed verification generates `k+1` tokens correctly
- ✅ Confirmed `base_logits.shape=[1, 5, 50257]` for k=4
- ✅ Policy correctly returns `accepted_len=4` per step

### Step 2: Added Debug Logging
Added comprehensive logging to track:
- Verification handler output shapes
- Batch loop token extraction
- Duplication detection filtering

### Step 3: Root Cause Identified
**Issue**: Duplication detection was filtering verified tokens

**Evidence**:
```
Step 1: accepted_tokens=[5087, 5087, 5087, 5087], generated_so_far=[]
        → After duplication: [5087, 5087, 5087, 5087] (no filtering, nothing generated yet)
        → Adds 4 tokens ✓

Step 2: accepted_tokens=[5087, 5087, 5087, 5087], generated_so_far=[5087, 5087, 5087, 5087]
        → After duplication: [] (ALL filtered out!)
```

**Why**: Duplication detection saw `accepted_tokens[0] == last_generated` (5087 == 5087) and filtered all tokens, even though these tokens were **verified by the base model** (draft matches base).

## Fix Applied

**Solution**: Skip duplication detection for verified tokens in non-deterministic mode.

**Rationale**: 
- Tokens accepted by the acceptance policy are verified (draft matches base)
- If the base model generates repetitive tokens, they're legitimate model output
- Duplication detection should only filter unverified or clearly erroneous tokens
- For performance benchmarks, we should trust verified tokens

**Code Change**: In `batch_loop.py`, skip `detect_duplication()` call when `deterministic_mode=False` for verified tokens.

## Results

### Before Fix
- Acceptance rate: 12%
- Proposed: 32, Accepted: 4
- Speedup: 0.22x (slower)

### After Fix
- Acceptance rate: 100% ✅
- Proposed: 8, Accepted: 8 (for max_tokens=8)
- Speedup: 1.66x on CPU (faster!) ✅
- Total tokens match vanilla: 32 tokens for 2 prompts ✅

## Key Learnings

1. **Verification fix was correct**: Generating `k+1` tokens gives proper logits
2. **Duplication detection was too aggressive**: Filtering verified tokens breaks speculative decoding
3. **Trust verified tokens**: If draft matches base (verified by policy), trust the model's output even if repetitive
4. **CPU overhead**: On CPU, speculative decoding can still be slower due to overhead, but acceptance rate is now correct

## Status
✅ **FIXED**: Batch path now shows 100% acceptance for perfect draft
✅ **VERIFIED**: Tokens match vanilla decoding exactly
📝 **READY**: Infrastructure ready for GPU benchmarks where speedup should be more significant

