#!/usr/bin/env python3
"""
EQSPEC Correctness Regression Test

Compares speculative decoding vs non-speculative target-only decoding.
With EQSPEC bonus token implementation, they must match token-for-token
for the full sequence (until floating-point nondeterminism).

This test verifies that:
1. Bonus tokens are sampled from target model distribution (not draft)
2. Accepted draft tokens + bonus token = correct sequence
3. Full token sequences match between speculative and non-speculative
4. Decoded text matches exactly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import random
import numpy as np
from specdec.core.pipeline import SpeculativePipeline
from specdec.models.hf_wrappers import create_tiny_hf_wrapper


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_eqspec_correctness():
    """Test that speculative decoding matches non-speculative decoding."""
    print("=" * 80)
    print("EQSPEC CORRECTNESS REGRESSION TEST")
    print("Testing bonus token implementation")
    print("=" * 80)
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print()
    
    # Test configurations: (prompt, max_tokens, temperature, do_sample, top_p, top_k, seed)
    test_configs = [
        ("The capital of France is", 10, 0.7, True, 0.9, 50, 42),
        ("Machine learning is", 10, 0.0, False, None, None, 123),  # Greedy
        ("Hello world", 8, 1.0, True, None, None, 456),  # Temperature=1.0, no filtering
    ]
    
    # Create pipeline for speculative decoding
    print("Creating speculative pipeline...")
    try:
        pipeline = SpeculativePipeline(
            base_model="gpt2",
            draft_model="distilgpt2",
            max_draft=4,
            device=device,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            implementation="hf",
        )
        print("✅ Pipeline created successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize pipeline: {e}")
        return False
    
    print()
    
    all_passed = True
    
    for test_idx, (prompt, max_tokens, temperature, do_sample, top_p, top_k, seed) in enumerate(test_configs, 1):
        print(f"Test {test_idx}: {prompt!r}")
        print(f"  Config: max_tokens={max_tokens}, temp={temperature}, do_sample={do_sample}, "
              f"top_p={top_p}, top_k={top_k}, seed={seed}")
        print("-" * 80)
        
        try:
            # Set seed for reproducibility
            set_seed(seed)
            
            # Prepare kwargs for generation
            gen_kwargs = {
                "temperature": temperature,
                "do_sample": do_sample,
            }
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            
            # Speculative decoding with bonus token
            set_seed(seed)  # Reset seed before speculative generation
            spec_result = pipeline.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            spec_tokens = spec_result.get("generated_tokens", [])
            spec_text = spec_result.get("text", "")
            
            # Non-speculative (target-only) decoding
            # Create a simple target-only pipeline by using base model only
            base_lm = create_tiny_hf_wrapper(
                model_name="gpt2",
                device=device,
                dtype=torch.float16 if device != "cpu" else torch.float32,
            )
            
            # Tokenize prompt
            tokenizer = base_lm._tokenizer
            prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            
            # Generate with base model only (non-speculative) - same settings
            set_seed(seed)  # Reset seed before non-speculative generation
            non_spec_tokens = []
            current_input = prompt_tokens
            for _ in range(max_tokens):
                tokens, _ = base_lm.generate_tokens(
                    current_input,
                    max_new_tokens=1,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                )
                if tokens.numel() > 0:
                    token_id = tokens[0, -1].item()
                    non_spec_tokens.append(token_id)
                    current_input = torch.cat([current_input, tokens[:, -1:]], dim=1)
                else:
                    break
            
            # Decode non-speculative tokens
            if non_spec_tokens:
                non_spec_text = tokenizer.decode(
                    non_spec_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            else:
                non_spec_text = ""
            
            # Compare full sequences
            min_len = min(len(spec_tokens), len(non_spec_tokens))
            max_len = max(len(spec_tokens), len(non_spec_tokens))
            
            if min_len == 0:
                print("⚠️  WARNING: No tokens generated")
                continue
            
            # Check token-by-token equality
            tokens_match = True
            first_mismatch = None
            for j in range(min_len):
                if spec_tokens[j] != non_spec_tokens[j]:
                    tokens_match = False
                    first_mismatch = j
                    break
            
            # Check if lengths match
            length_match = len(spec_tokens) == len(non_spec_tokens)
            
            # Check text equality (after normalization)
            spec_text_normalized = spec_text.strip()
            non_spec_text_normalized = non_spec_text.strip()
            text_match = spec_text_normalized == non_spec_text_normalized
            
            # Report results
            if tokens_match and length_match and text_match:
                print(f"✅ PASSED: Full sequence match ({min_len} tokens)")
                print(f"   Speculative tokens: {spec_tokens}")
                print(f"   Non-speculative tokens: {non_spec_tokens}")
                print(f"   Text match: {repr(spec_text_normalized[:80])}")
            else:
                print(f"❌ FAILED:")
                if not tokens_match:
                    print(f"   Token mismatch at position {first_mismatch}")
                    print(f"   Speculative:   {spec_tokens[first_mismatch]}")
                    print(f"   Non-speculative: {non_spec_tokens[first_mismatch]}")
                    print(f"   Speculative tokens: {spec_tokens}")
                    print(f"   Non-speculative tokens: {non_spec_tokens}")
                if not length_match:
                    print(f"   Length mismatch: spec={len(spec_tokens)}, non_spec={len(non_spec_tokens)}")
                if not text_match:
                    print(f"   Text mismatch:")
                    print(f"   Speculative:   {repr(spec_text_normalized[:80])}")
                    print(f"   Non-speculative: {repr(non_spec_text_normalized[:80])}")
                all_passed = False
            
        except Exception as e:
            print(f"❌ ERROR: Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        
        print()
    
    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("EQSPEC bonus token implementation is correct!")
    else:
        print("❌ SOME TESTS FAILED")
        print("EQSPEC bonus token implementation needs fixes.")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_eqspec_correctness()
    sys.exit(0 if success else 1)



