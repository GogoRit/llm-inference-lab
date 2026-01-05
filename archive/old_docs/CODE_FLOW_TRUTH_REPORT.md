# Code Flow Truth Report
**Generated:** 2025-01-XX  
**Repository:** llm-inference-lab (Zero-Copy Speculative Decoding Engine)  
**Audit Type:** Strict code-grounded implementation verification

---

## A. Executive Summary

### What Happens Per Decoding Step (Plain English)

1. **Draft Generation:** Draft model generates K tokens autoregressively (one forward pass per token in async mode, or K sequential passes in sync mode). Draft tokens are stored in a tensor `[batch_size, K]`. Draft KV cache is written to pre-allocated ring buffer via `kv_cache_manager.update_draft_cache()` at `src/specdec/cache/kv_cache_manager.py:468-557`.

2. **Verification:** Base model generates K+1 tokens in a **single forward pass** (parallel verification) from the prompt (NOT from appended draft tokens). This happens at `src/specdec/core/pipeline.py:1288-1294` (single-prompt) or `src/specdec/core/batch_handlers.py:429-435` (batch). The base model uses `generate_tokens()` with `max_new_tokens=k+1`, which internally calls HuggingFace's `generate()` that processes all K+1 positions in one prefill step.

3. **Acceptance:** Policy compares `draft_tokens[:, :k]` with `base_tokens[:, :k]` (or `base_logits[:, :k].argmax()`) to find longest matching prefix. This happens at `src/specdec/policies/policies.py:101-148` (LongestPrefixPolicy). Accepted length is an integer (0 to K).

4. **Commit/Rollback:** If `accepted_len > 0`, accepted tokens are appended to `current_input` via `torch.cat()` at `src/specdec/core/pipeline.py:1415` (single) or `src/specdec/core/batch_loop.py:565` (batch). KV cache pointer is updated via `RollbackHandler.update_sequence_pointers()` at `src/specdec/core/batch_handlers.py:798-836`, which sets `base_current_seq_lens[idx] = original_len + accepted_len` (O(1) pointer update). **No tensor copy occurs for rollback** - rejected KV data remains in ring buffer but is masked out by sequence length pointer.

### What Is Genuinely "Zero-Copy" vs What Still Copies

**Genuinely Zero-Copy:**
- **KV cache rollback:** `RollbackHandler.update_sequence_pointers()` at `src/specdec/core/batch_handlers.py:828-830` only updates integer pointer `base_current_seq_lens[idx]`. No tensor operations.
- **KV cache writes:** `kv_cache_manager.update_base_cache()` at `src/specdec/cache/kv_cache_manager.py:443-447` writes directly into pre-allocated buffer via in-place assignment `key_buffer[batch_pos, :, current_pos:new_pos, :] = key[i].detach().contiguous()`. The `.detach().contiguous()` is necessary for gradient safety but is a view operation, not a copy of the entire buffer.

**Still Copying/Reallocating:**
- **Token concatenation:** `torch.cat([current_input, accepted_tokens_limited], dim=1)` at `src/specdec/core/pipeline.py:1415` creates a new tensor. This is unavoidable for token sequences.
- **KV cache append (single-prompt path):** `base_lm.append_kv_cache()` at `src/specdec/core/pipeline.py:1368` calls `hf_wrappers.py:1232-1269`, which uses `torch.cat([base_k, new_k], dim=2)` at `src/specdec/models/hf_wrappers.py:1288-1289` (PyTorch fallback) or kernel append (which may still copy internally). **This is NOT using the ring buffer** - it's a separate KV cache stored in `base_lm._kv_cache`.
- **Draft token generation:** `hf_wrappers.py:350-677` uses a Python loop `for step in range(max_new_tokens)` that calls model forward K times sequentially in async mode, or uses HuggingFace's `generate()` which internally concatenates tokens at `hf_wrappers.py:653` and `hf_wrappers.py:672-677`.
- **Batch padding:** `pad_sequences()` at `src/specdec/core/sequence_utils.py:36-63` creates new padded tensors via `torch.stack()`.

**Critical Finding:** The **single-prompt path** (`pipeline.generate()`) uses a **different KV cache mechanism** than the batch path. Single-prompt uses `base_lm.append_kv_cache()` which concatenates tensors, while batch path uses `kv_cache_manager.update_base_cache()` which writes to ring buffer. These are **two separate code paths** with different memory behavior.

---

## B. Canonical Call Graph (Single Prompt)

**Entry Point:**
```
scripts/comprehensive_k_sweep.py:main() [line 107]
  └─> scripts/k_sweep/runner.py:run_comprehensive_k_sweep() [line 50]
       └─> SpeculativePipeline.generate_batch() [if batch mode]
            OR
       └─> SpeculativePipeline.generate() [if single prompt, line 324]
```

**Single-Prompt Generation Path:**
```
src/specdec/core/pipeline.py:generate() [line 1090]
  ├─> Tokenize: base_lm.encode(prompt) [line 1162]
  ├─> Initialize: current_input = input_ids.clone() [line 1173]
  └─> while len(generated_tokens) < max_tokens [line 1181-1474]:
       ├─> controller.get_k() [line 1202]
       │    └─> src/specdec/policies/controllers.py:FixedKController.get_k() [or Adaptive]
       │
       ├─> DRAFT GENERATION [line 1205-1239]:
       │    └─> draft_lm.generate_tokens(current_input, max_new_tokens=k) [line 1216]
       │         └─> src/specdec/models/hf_wrappers.py:generate_tokens() [line 156]
       │              ├─> If async stream: _generate_tokens_async() [line 297]
       │              │    └─> for step in range(max_new_tokens): [line 350]
       │              │         └─> model.forward() [per token, sequential]
       │              └─> Else: model.generate() [line 263]
       │                   └─> HuggingFace generate() [internal, processes all K tokens]
       │
       ├─> VERIFICATION [line 1246-1299]:
       │    └─> base_lm.generate_tokens(current_input, max_new_tokens=k+1) [line 1288]
       │         └─> src/specdec/models/hf_wrappers.py:generate_tokens() [line 156]
       │              └─> model.generate(max_new_tokens=k+1) [line 263]
       │                   └─> HuggingFace generate() [single forward pass for k+1 tokens]
       │                        └─> Returns: base_tokens [batch, k+1], base_logits [batch, k+1, vocab]
       │
       ├─> ACCEPTANCE [line 1301-1313]:
       │    └─> policy.accept_tokens(draft_tokens, base_tokens, draft_logits, base_logits) [line 1311]
       │         └─> src/specdec/policies/policies.py:LongestPrefixPolicy.accept_tokens() [line 101]
       │              ├─> Try kernel: get_verify_prefix() [line 128-134] (if CUDA)
       │              └─> Else PyTorch: Compare draft_tokens[:, :k] == base_tokens[:, :k] [line 140-148]
       │                   └─> Returns: accepted_len (int, 0 to k)
       │
       └─> COMMIT/ROLLBACK [line 1347-1450]:
            ├─> if accepted_len > 0: [line 1348]
            │    ├─> KV APPEND (if enabled): [line 1351-1398]
            │    │    ├─> base_lm.get_last_generated_kv() [line 1359]
            │    │    ├─> base_kv.slice_prefix(accepted_len) [line 1366] [CREATES NEW TENSOR]
            │    │    └─> base_lm.append_kv_cache(accepted_kv) [line 1368]
            │    │         └─> src/specdec/models/hf_wrappers.py:append_kv_cache() [line 1232]
            │    │              └─> _append_kv_pytorch() [line 1271] OR _append_kv_with_kernel() [line 1303]
            │    │                   └─> torch.cat([base_k, new_k], dim=2) [line 1288] [COPY OPERATION]
            │    │
            │    ├─> Accept tokens: [line 1400-1417]
            │    │    ├─> accepted_tokens_limited = accepted_tokens[:, :tokens_to_accept] [line 1409]
            │    │    ├─> generated_tokens.extend(accepted_tokens_limited[0].cpu().tolist()) [line 1412]
            │    │    └─> current_input = torch.cat([current_input, accepted_tokens_limited], dim=1) [line 1415] [COPY]
            │    │
            │    └─> NO EXPLICIT ROLLBACK (single-prompt doesn't use ring buffer rollback)
            │
            └─> else (accepted_len == 0): [line 1424]
                 └─> base_lm.generate_tokens(current_input, max_new_tokens=1) [line 1429] [FALLBACK]
                      └─> current_input = torch.cat([current_input, fallback_tokens], dim=1) [line 1439] [COPY]
```

**Key Finding:** Single-prompt path does **NOT use ring buffer rollback**. It uses `base_lm.append_kv_cache()` which concatenates tensors. The ring buffer (`kv_cache_manager`) is only used in batch mode.

---

## C. Canonical Call Graph (Batch)

**Entry Point:**
```
scripts/comprehensive_k_sweep.py:main() [line 107]
  └─> scripts/k_sweep/runner.py:run_comprehensive_k_sweep() [line 50]
       └─> SpeculativePipeline.generate_batch() [line 514]
            └─> BatchGenerationLoop.run() [line 1985]
```

**Batch Generation Path:**
```
src/specdec/core/pipeline.py:generate_batch() [line 1820]
  ├─> Initialize: kv_cache_manager.reset() [line 1846]
  ├─> Create handlers: _create_batch_handlers() [line 1921]
  │    └─> Returns: {draft, verify, accept, rollback, kv} handlers
  └─> BatchGenerationLoop.run() [line 1985]
       └─> while any(sequence_manager.batch_active) [line 125]:
            ├─> controller.get_k() [line 145]
            │
            ├─> Extract active sequences: [line 149-153]
            │    └─> current_input_ids[i].detach().clone().contiguous() [line 151] [COPY per active seq]
            │
            ├─> Pad sequences: pad_sequences() [line 170]
            │    └─> src/specdec/core/sequence_utils.py:pad_sequences() [line 36]
            │         └─> torch.stack(padded_seqs, dim=0).contiguous() [line 63] [COPY]
            │
            ├─> DRAFT GENERATION [line 207-228]:
            │    └─> draft_handler.generate_draft_tokens() [line 214]
            │         └─> src/specdec/core/batch_handlers.py:DraftGenerationHandler.generate_draft_tokens() [line 53]
            │              ├─> If CUDA stream: with torch.cuda.stream(draft_stream) [line 162]
            │              └─> draft_lm.generate_tokens(active_input_ids, max_new_tokens=k) [line 164]
            │                   └─> src/specdec/models/hf_wrappers.py:generate_tokens() [line 156]
            │                        └─> model.generate(max_new_tokens=k) [line 263]
            │                             └─> Returns: draft_tokens [active_count, k], draft_logits [active_count, k, vocab]
            │
            ├─> VERIFICATION [line 230-251]:
            │    └─> verify_handler.verify_draft_tokens() [line 237]
            │         └─> src/specdec/core/batch_handlers.py:VerificationHandler.verify_draft_tokens() [line 286]
            │              ├─> Wait for draft: draft_end_event.synchronize() [line 360]
            │              └─> base_lm.generate_tokens(verify_input_ids, max_new_tokens=k+1) [line 429]
            │                   └─> src/specdec/models/hf_wrappers.py:generate_tokens() [line 156]
            │                        └─> model.generate(max_new_tokens=k+1) [line 263]
            │                             └─> Returns: base_tokens [active_count, k+1], base_logits [active_count, k+1, vocab]
            │
            ├─> ACCEPTANCE (per sequence) [line 253-500]:
            │    └─> for idx_in_active, global_idx in enumerate(active_indices): [line 261]
            │         ├─> Extract per-prompt: draft_tokens[idx_in_active:idx_in_active+1] [line 266] [VIEW, no copy]
            │         ├─> accept_handler.apply_acceptance_policy() [line 282]
            │         │    └─> src/specdec/core/batch_handlers.py:AcceptanceHandler.apply_acceptance_policy() [line 518]
            │         │         └─> policy.accept_tokens() [line 540]
            │         │              └─> Returns: accepted_len (int)
            │         │
            │         ├─> if accepted_len > 0: [line 319]
            │         │    ├─> Extract accepted tokens: prompt_base_tokens[0, :accepted_len] [line 330-334]
            │         │    │    └─> OR prompt_draft_tokens[0, :accepted_len].clone() [line 327] [COPY if using draft]
            │         │    ├─> sequence_manager.add_generated_tokens(global_idx, accepted_tokens) [line 393]
            │         │    └─> ROLLBACK: rollback_handler.update_sequence_pointers() [line 516]
            │         │         └─> src/specdec/core/batch_handlers.py:RollbackHandler.update_sequence_pointers() [line 798]
            │         │              └─> kv_cache_manager.base_current_seq_lens[idx_in_active] = original_len + accepted_len [line 829] [O(1) POINTER UPDATE]
            │         │
            │         └─> else (accepted_len == 0): [line 407]
            │              └─> accept_handler.sample_fallback_token() [line 419]
            │                   └─> Returns: fallback_token [1] tensor
            │
            └─> Update current_input_ids: [line 549-580]
                 └─> current_input_ids[global_idx] = torch.cat([current_seq, accepted_tokens_tensor], dim=0) [line 565] [COPY]
                      └─> updated_seq.detach().clone().contiguous() [line 579] [COPY]
```

**Key Finding:** Batch path uses ring buffer rollback (`RollbackHandler.update_sequence_pointers()`), but still copies tokens when updating `current_input_ids`. KV cache writes use in-place assignment to ring buffer.

---

## D. Data Flow of Tokens + Logits

### Draft Tokens Creation

**Single-Prompt:**
- `src/specdec/core/pipeline.py:1216` → `draft_lm.generate_tokens(current_input, max_new_tokens=k)`
- `src/specdec/models/hf_wrappers.py:156` → `generate_tokens()` method
- If async: `_generate_tokens_async()` at `hf_wrappers.py:297` → Python loop `for step in range(k)` at `hf_wrappers.py:350` → calls `model.forward()` K times sequentially
- If sync: `model.generate()` at `hf_wrappers.py:263` → HuggingFace processes all K tokens in one call
- Returns: `draft_tokens` `[1, k]`, `draft_logits` `[1, k, vocab_size]`

**Batch:**
- `src/specdec/core/batch_handlers.py:164` → `draft_lm.generate_tokens(active_input_ids, max_new_tokens=k)`
- Same path as single-prompt, but with `active_input_ids` `[active_count, seq_len]`
- Returns: `draft_tokens` `[active_count, k]`, `draft_logits` `[active_count, k, vocab_size]`

### Base Tokens/Logits Creation for Verification

**Single-Prompt:**
- `src/specdec/core/pipeline.py:1288` → `base_lm.generate_tokens(current_input, max_new_tokens=k+1)`
- `src/specdec/models/hf_wrappers.py:156` → `generate_tokens()` method
- `model.generate(max_new_tokens=k+1)` at `hf_wrappers.py:263`
- **CRITICAL:** HuggingFace's `generate()` processes all `k+1` tokens in **one forward pass** (prefill phase). This is parallel verification.
- Returns: `base_tokens` `[1, k+1]`, `base_logits` `[1, k+1, vocab_size]`

**Batch:**
- `src/specdec/core/batch_handlers.py:429` → `base_lm.generate_tokens(verify_input_ids, max_new_tokens=k+1)`
- Same path as single-prompt, but with `verify_input_ids` `[active_count, seq_len]`
- Returns: `base_tokens` `[active_count, k+1]`, `base_logits` `[active_count, k+1, vocab_size]`

**Verification Input:** `verify_input_ids = active_input_ids` (prompt only, NOT appended with draft tokens) at `batch_handlers.py:375`. This is correct - base model generates k+1 tokens from prompt, then first k are compared with draft.

### Acceptance Length Computation

**Single-Prompt:**
- `src/specdec/core/pipeline.py:1311` → `policy.accept_tokens(draft_tokens, base_tokens, draft_logits, base_logits)`
- `src/specdec/policies/policies.py:101` → `LongestPrefixPolicy.accept_tokens()`
- If CUDA kernel available: `get_verify_prefix(base_logits, proposed_tokens)` at `policies.py:128-134`
- Else PyTorch: `(draft_tokens[:, :k] == base_tokens[:, :k])` comparison at `policies.py:140-148`
- Returns: `accepted_len` (int, 0 to k)

**Batch:**
- `src/specdec/core/batch_handlers.py:282` → `accept_handler.apply_acceptance_policy()`
- `src/specdec/core/batch_handlers.py:540` → `policy.accept_tokens()` (same as single-prompt)
- Returns: `accepted_len` (int, 0 to k)

### Fallback Token Generation (When Rejection Happens)

**Single-Prompt:**
- `src/specdec/core/pipeline.py:1429` → `base_lm.generate_tokens(current_input, max_new_tokens=1)`
- Same path as verification, but `max_new_tokens=1`
- Returns: `fallback_tokens` `[1, 1]`

**Batch:**
- `src/specdec/core/batch_handlers.py:419` → `accept_handler.sample_fallback_token()`
- `src/specdec/core/batch_handlers.py:731` → `sample_fallback_token()` method
- Uses `sample_bonus_token_from_logits()` at `pipeline.py:57` with `prompt_base_logits[:, -1, :]` (last position logits from verification)
- Returns: `fallback_token` `[1]` tensor

---

## E. KV Cache Flow (Most Important)

### Base Model KV Cache

#### Allocation (Prealloc)

**Batch Path (Ring Buffer):**
- `src/specdec/cache/kv_cache_manager.py:202` → `_ensure_buffers_initialized()`
- Called from `update_base_cache()` at `kv_cache_manager.py:380` (lazy initialization)
- Allocates: `torch.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype, device=device)` at `kv_cache_manager.py:245-250`
- Shape: `[batch_size, num_heads, max_seq_len, head_dim]` per layer
- Stored in: `self.base_cache[layer_idx] = [key_buffer, value_buffer]` at `kv_cache_manager.py:255`

**Single-Prompt Path (NOT Ring Buffer):**
- `src/specdec/models/hf_wrappers.py:1232` → `append_kv_cache()` method
- First call: `self._kv_cache = kv_chunk.to(torch.device(self._device))` at `hf_wrappers.py:1253` [COPY via .to()]
- Subsequent: `_append_kv_pytorch()` or `_append_kv_with_kernel()` at `hf_wrappers.py:1257-1264`
- **No pre-allocation** - grows via concatenation

#### Write/Append for Accepted Tokens

**Batch Path (Ring Buffer - In-Place Write):**
- `src/specdec/cache/kv_cache_manager.py:335` → `update_base_cache(new_kv, active_indices)`
- Called from model wrapper after base model forward (indirectly via `past_key_values` return)
- In-place assignment: `key_buffer[batch_pos, :, current_pos:new_pos, :] = key[i].detach().contiguous()` at `kv_cache_manager.py:443-447`
- **The `.detach().contiguous()` is for gradient safety but is a view operation on the source tensor, not a copy of the entire buffer.**
- Updates pointer: `self.base_current_seq_lens[batch_pos] = new_pos` at `kv_cache_manager.py:452`

**Single-Prompt Path (Concatenation - COPY):**
- `src/specdec/core/pipeline.py:1368` → `base_lm.append_kv_cache(accepted_kv)`
- `src/specdec/models/hf_wrappers.py:1232` → `append_kv_cache(kv_chunk)`
- If kernel: `_append_kv_with_kernel()` at `hf_wrappers.py:1303` → kernel may still copy internally
- Else PyTorch: `_append_kv_pytorch()` at `hf_wrappers.py:1271` → `torch.cat([base_k, new_k], dim=2)` at `hf_wrappers.py:1288-1289` [COPY OPERATION]
- **This is NOT zero-copy** - concatenation creates a new tensor.

#### Rollback (What It Does Exactly)

**Batch Path (Ring Buffer - O(1) Pointer Update):**
- `src/specdec/core/batch_handlers.py:798` → `RollbackHandler.update_sequence_pointers()`
- Updates pointer: `self.kv_cache_manager.base_current_seq_lens[idx_in_active] = original_len + accepted_len` at `batch_handlers.py:829`
- **No tensor operations** - only integer assignment
- Rejected KV data remains in ring buffer but is masked out by sequence length pointer
- **This is genuinely O(1) and zero-copy**

**Single-Prompt Path (NO ROLLBACK):**
- Single-prompt path does not use ring buffer, so no rollback mechanism exists
- If tokens are rejected, base model simply regenerates from `current_input` (which doesn't include rejected draft tokens)

#### Rejection Behavior (Does It Cause Copy/Recompute?)

**Batch Path:**
- **No copy/recompute on rejection** - pointer is updated to `original_len + accepted_len`, and rejected KV data is ignored
- Next draft generation uses `kv_cache_manager.get_draft_past_kv()` which returns sliced view `[:, :, :current_seq_len, :]` at `kv_cache_manager.py:578-580`
- The view automatically excludes rejected positions

**Single-Prompt Path:**
- **No explicit rejection handling** - if `accepted_len == 0`, fallback token is generated, but no KV cache rollback occurs
- Base model's internal KV cache (if any) is not rolled back

#### Every Place KV Tensors Are Created/Copied/Concatenated/Cloned

**Ring Buffer (Batch Path):**
1. **Allocation:** `torch.zeros()` at `kv_cache_manager.py:245-250` [ONE-TIME ALLOCATION]
2. **Write:** `key_buffer[batch_pos, :, current_pos:new_pos, :] = key[i].detach().contiguous()` at `kv_cache_manager.py:443-447` [IN-PLACE ASSIGNMENT, .detach().contiguous() is view on source]
3. **Read:** `filter_kv_cache()` returns sliced views `[:, :, :max_seq_len, :]` at `kv_cache_manager.py:328-329` [VIEW, no copy]

**Single-Prompt Path (Concatenation):**
1. **First append:** `self._kv_cache = kv_chunk.to(torch.device(self._device))` at `hf_wrappers.py:1253` [COPY via .to()]
2. **Subsequent append (PyTorch):** `torch.cat([base_k, new_k], dim=2)` at `hf_wrappers.py:1288-1289` [COPY]
3. **Subsequent append (kernel):** Kernel may copy internally (implementation not visible)

**Draft Model KV Cache:**
- Same as base model - ring buffer in batch path (`update_draft_cache()` at `kv_cache_manager.py:468`), concatenation in single-prompt (if used)

### Draft Model KV Cache

**Allocation:** Same as base model - `_ensure_buffers_initialized(cache_type="draft")` at `kv_cache_manager.py:507`

**Write:** `update_draft_cache()` at `kv_cache_manager.py:468` → in-place assignment at `kv_cache_manager.py:548-552`

**Rollback:** `sync_draft_cache_pointer()` at `batch_handlers.py:838` → updates `draft_current_seq_lens[idx_in_active] = original_len + accepted_len + 1` at `batch_handlers.py:861` (includes bonus token)

---

## F. Verification Behavior vs K (Math Alignment)

### How Verification Is Implemented

**Answer: Base model is called ONCE for k+1 tokens (parallel verification)**

**Evidence:**
- `src/specdec/core/pipeline.py:1288` → `base_lm.generate_tokens(current_input, max_new_tokens=k+1)`
- `src/specdec/models/hf_wrappers.py:263` → `model.generate(max_new_tokens=k+1)`
- HuggingFace's `generate()` with `max_new_tokens=k+1` processes all k+1 positions in **one forward pass** (prefill phase for the entire sequence)

**This is NOT a loop per token** - it's a single call that generates k+1 tokens autoregressively but in one model invocation.

### Exact Code Lines Where K Affects Compute

1. **Draft generation:** `max_new_tokens=k` at `pipeline.py:1218` (single) or `batch_handlers.py:166` (batch)
   - If async: Python loop `for step in range(k)` at `hf_wrappers.py:350` → **K sequential forward passes**
   - If sync: HuggingFace `generate(max_new_tokens=k)` → **single call, but internally processes K tokens**

2. **Verification:** `max_new_tokens=k+1` at `pipeline.py:1290` (single) or `batch_handlers.py:431` (batch)
   - HuggingFace `generate(max_new_tokens=k+1)` → **single forward pass for k+1 tokens**

3. **Acceptance:** Compares `draft_tokens[:, :k]` with `base_tokens[:, :k]` at `policies.py:140-148`
   - This is element-wise comparison, O(K) but very fast (vectorized)

### Hidden Per-Token Python Loops That Scale with K

**FOUND:**
1. **Draft generation (async mode):** `for step in range(max_new_tokens)` at `hf_wrappers.py:350` → **K sequential forward passes**
2. **Draft generation (sync mode):** HuggingFace `generate()` internally processes K tokens, but this is in C++/CUDA, not Python loop

**NOT FOUND:**
- Verification does NOT have a Python loop - it's a single `generate(k+1)` call

**Critical Finding:** In async mode, draft generation has a Python loop that scales with K. This contradicts the "constant K verification" claim - verification is constant, but draft generation is NOT constant in async mode.

---

## G. Benchmarking Truth

### Source of Truth Benchmark Script(s)

**Primary Script:**
- `scripts/comprehensive_k_sweep.py` [line 107: `main()`]
- This is the canonical benchmark script used in all documented results

**Alternative Scripts (NOT primary):**
- `scripts/benchmark_specdec_vs_vanilla.py` - Comparison script (not used in main results)
- `scripts/run_benchmark_grid.py` - Grid search (not used in main results)

### Script Arguments and Defaults

**`scripts/comprehensive_k_sweep.py`:**
- `--base-model`: default `"gpt2"` [line 111]
- `--draft-model`: default `"distilgpt2"` [line 112]
- `--max-tokens`: default `32` [line 120]
- `--iterations`: default `10` [line 123]
- `--output-dir`: default `"results"` [line 126]
- `--device`: default `"auto"` [line 131]
- `--max-k`: default `4` [line 160]
- `--deterministic`: flag, default `False` [line 150]
- `--no-plots`: flag, default `False` [line 136]
- `--batch-size`: default `None` (uses `SPECDEC_BATCH_SIZE` env var) [line 180]

### Environment Variables That Matter

**From `scripts/k_sweep/runner.py` and codebase:**
- `SPECDEC_BATCH_SIZE`: default `"8"` [runner.py:108]
- `SPECDEC_DETERMINISTIC`: default `"0"` [runner.py:163]
- `SPECDEC_ENABLE_KV_APPEND`: default `"0"` for batch path [pipeline.py:1914], `"1"` for single-prompt [hf_wrappers.py:52]
- `SPECDEC_FORCE_PYTORCH_BACKEND`: default `"0"` [runner.py:186]
- `SPECDEC_PARALLEL_STREAMS`: default `"1"` [runner.py:109]
- `SPECDEC_DTYPE`: default `"auto"` [runner.py:110]
- `SPECDEC_DRY_RUN`: default `"0"` [runner.py:131]

### Metrics Recorded and Where (CSV/JSON Schema)

**CSV File:** `scripts/k_sweep/results.py:57` → `specdec_{device}_{timestamp}.csv`

**CSV Schema (from `runner.py:812-836`):**
- `k`: K value
- `n_samples`: Number of successful samples
- `n_failures`: Number of failed samples
- `success_rate`: Fraction of successful samples
- `latency_ms_mean`, `latency_ms_std`: Mean and std of latency per token (ms)
- `tokens_per_sec_mean`, `tokens_per_sec_std`: Mean and std of throughput (tok/s)
- `acceptance_rate_mean`, `acceptance_rate_std`: Mean and std of acceptance rate
- `proposed_mean`, `proposed_std`: Mean and std of proposed tokens
- `accepted_mean`, `accepted_std`: Mean and std of accepted tokens
- `kv_appended_tokens_mean`, `kv_appended_tokens_std`: Mean and std of KV appended tokens
- `kv_append_time_ms_mean`, `kv_append_time_ms_std`: Mean and std of KV append time (ms)
- `device`: Device name
- `dtype`: Data type (float16/float32)

**JSON File:** `scripts/k_sweep/results.py:65` → `specdec_{device}_{timestamp}.json`

**JSON Schema (from `results.py:67-76`):**
```json
{
  "system_info": {...},  // System metadata
  "summary_results": [...],  // Same as CSV rows
  "detailed_results": [...],  // Per-iteration results
  "detailed_metrics": {...}  // Profiling data (if enabled)
}
```

**Detailed Results Schema (from `runner.py:623-656`):**
- `k`, `iteration`, `prompt_idx`, `prompt_name`, `prompt`, `prompt_text`, `completion_text`, `full_text`
- `completion_token_count`, `latency_ms`, `tokens_per_sec`, `acceptance_rate`
- `proposed`, `accepted`, `kv_appended_tokens`, `kv_append_time_ms`
- `kv_append_enabled`, `kv_append_backend`, `success`, `device`, `dtype`, `batch_size`

### Whether Tokenization, Logging, Plotting Are Included in Timing

**Tokenization:**
- **NOT included** - Tokenization happens before timing starts:
  - Single-prompt: `base_lm.encode(prompt)` at `pipeline.py:1162` (before `start_time`)
  - Batch: Tokenization in `generate_batch()` at `pipeline.py:1882` (before `batch_start_time` at `runner.py:513`)

**Logging:**
- **NOT included** - All logging uses `logger.debug()` or `logger.info()`, which are not timed
- Console prints in `runner.py` are outside timing blocks

**Plotting:**
- **NOT included** - Plotting happens after benchmark completes:
  - `scripts/comprehensive_k_sweep.py:262` → `create_plots()` called after `save_results()`

**What IS Included in Timing:**
- `pipeline.generate()`: Total time from `start_time = time.time()` at `pipeline.py:1111` to `total_time_ms = (time.time() - start_time) * 1000` at `pipeline.py:1490`
- `pipeline.generate_batch()`: Total time from `batch_start_time = time.time()` at `runner.py:513` to `batch_end_time = time.time()` at `runner.py:525`
- This includes: model forward passes, token generation, acceptance policy, KV cache operations, but NOT tokenization/logging/plotting

---

## H. "Red Flags" List

### Code Paths That Contradict Zero-Copy Claim

1. **Single-prompt KV cache append uses concatenation:**
   - `src/specdec/models/hf_wrappers.py:1288-1289` → `torch.cat([base_k, new_k], dim=2)` [COPY]
   - **Location:** Single-prompt path only
   - **Impact:** High - every accepted token causes a full KV cache copy

2. **Token concatenation in generation loop:**
   - `src/specdec/core/pipeline.py:1415` → `torch.cat([current_input, accepted_tokens_limited], dim=1)` [COPY]
   - `src/specdec/core/batch_loop.py:565` → `torch.cat([current_seq, accepted_tokens_tensor], dim=0)` [COPY]
   - **Location:** Both single-prompt and batch paths
   - **Impact:** Medium - unavoidable for token sequences, but contradicts "zero-copy" marketing

3. **Batch sequence cloning:**
   - `src/specdec/core/batch_loop.py:151` → `current_input_ids[i].detach().clone().contiguous()` [COPY per active sequence]
   - **Location:** Batch path, every step
   - **Impact:** Medium - copies entire input sequence for each active sequence

4. **Batch padding:**
   - `src/specdec/core/sequence_utils.py:63` → `torch.stack(padded_seqs, dim=0).contiguous()` [COPY]
   - **Location:** Batch path, every step
   - **Impact:** Medium - creates new padded tensor

### Places Where KV Append Is Disabled by Default or Not Actually Used

1. **Batch path KV append disabled by default:**
   - `src/specdec/core/pipeline.py:1914` → `os.getenv("SPECDEC_ENABLE_KV_APPEND", "0") == "1"`
   - **Default:** `"0"` (disabled)
   - **Impact:** High - batch path does NOT use KV append by default, so ring buffer may not be populated

2. **Single-prompt path KV append enabled by default but uses concatenation:**
   - `src/specdec/models/hf_wrappers.py:52` → `os.getenv("SPECDEC_ENABLE_KV_APPEND", "1")`
   - **Default:** `"1"` (enabled), but uses `torch.cat()` which is NOT zero-copy
   - **Impact:** High - single-prompt path claims to use KV append but actually copies

3. **KV append only works if `supports_kv_append()` returns True:**
   - `src/specdec/core/pipeline.py:1352` → `hasattr(self.base_lm, "supports_kv_append") and self.base_lm.supports_kv_append()`
   - **Impact:** Medium - may silently fail if model doesn't support it

### Places Where Results Comparisons Are Not Apples-to-Apples

1. **Single-prompt vs batch use different KV cache mechanisms:**
   - Single-prompt: Concatenation-based (`append_kv_cache()`)
   - Batch: Ring buffer (`kv_cache_manager.update_base_cache()`)
   - **Impact:** High - cannot directly compare single-prompt and batch results

2. **Async vs sync draft generation:**
   - Async: Python loop with K sequential forward passes (`hf_wrappers.py:350`)
   - Sync: HuggingFace `generate()` (single call, but internal processing may vary)
   - **Impact:** Medium - async mode has different performance characteristics

3. **KV append enabled vs disabled:**
   - If disabled, base model recomputes KV cache from scratch each step
   - If enabled, KV cache is reused (but may copy in single-prompt path)
   - **Impact:** High - results with `SPECDEC_ENABLE_KV_APPEND=0` vs `=1` are not comparable

4. **Deterministic vs non-deterministic mode:**
   - Deterministic: Disables duplication detection, uses fixed seeds
   - Non-deterministic: May filter tokens, uses random seeds
   - **Impact:** Medium - acceptance rates may differ

---

## Start Debugging Here (Top 3 Files)

1. **`src/specdec/cache/kv_cache_manager.py`** (763 lines)
   - **Why:** Contains ring buffer implementation (`SafeKVCacheManager`)
   - **Key functions:** `update_base_cache()` [335], `update_draft_cache()` [468], `_ensure_buffers_initialized()` [202], `reset()` [82]
   - **Read first:** Lines 335-467 (base cache update), 82-135 (reset), 202-270 (buffer initialization)

2. **`src/specdec/core/batch_loop.py`** (610 lines)
   - **Why:** Contains batch generation loop with rollback logic
   - **Key functions:** `run()` [79] (main loop), calls `RollbackHandler.update_sequence_pointers()` [516]
   - **Read first:** Lines 79-260 (main loop structure), 511-534 (rollback call site)

3. **`src/specdec/models/hf_wrappers.py`** (1398 lines)
   - **Why:** Contains KV cache append implementation (concatenation vs kernel)
   - **Key functions:** `append_kv_cache()` [1232], `_append_kv_pytorch()` [1271], `_append_kv_with_kernel()` [1303], `generate_tokens()` [156]
   - **Read first:** Lines 1232-1269 (append entry point), 1271-1301 (PyTorch fallback), 156-295 (token generation)

**Secondary Files:**
- `src/specdec/core/pipeline.py` (2120 lines) - Single-prompt path, different from batch
- `src/specdec/core/batch_handlers.py` (921 lines) - Handler classes (draft, verify, accept, rollback)
- `src/specdec/policies/policies.py` (488 lines) - Acceptance policy implementation

---

**END OF REPORT**

