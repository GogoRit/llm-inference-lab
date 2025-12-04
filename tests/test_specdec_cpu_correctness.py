"""
CPU Correctness Test Harness for Speculative Decoding

This test verifies that speculative decoding produces bit-identical output
to vanilla greedy decoding when:
- Draft model == Target model (perfect draft)
- Greedy decoding (do_sample=False)
- Deterministic mode enabled

This is a critical correctness test that must pass before any GPU experiments.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from specdec.core.pipeline import SpeculativePipeline
from specdec.models.hf_wrappers import HFWrapper
from specdec.policies.controllers import create_controller
from specdec.policies.policies import create_policy
from specdec.utils.deterministic import set_deterministic_mode
from specdec.utils.interfaces import LanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def vanilla_generate(
    prompt: str,
    model: LanguageModel,
    max_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> List[int]:
    """
    Run vanilla greedy autoregressive generation.

    Args:
        prompt: Input prompt text
        model: Language model instance
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (ignored if do_sample=False)
        do_sample: Whether to use sampling (False for greedy)

    Returns:
        List of generated token IDs
    """
    # Set deterministic mode
    set_deterministic_mode(seed=42, device="cpu")

    # Encode prompt and ensure proper shape/device
    try:
        input_ids = model.encode(prompt)
    except Exception as e:
        logger.error(f"Failed to encode prompt: {e}")
        # Fallback: use tokenizer directly
        if hasattr(model, "_tokenizer"):
            input_ids = model._tokenizer.encode(prompt, return_tensors="pt")
        else:
            raise

    # Ensure input_ids is 2D [batch_size, seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # Ensure on CPU device and contiguous
    if input_ids.device != torch.device("cpu"):
        input_ids = input_ids.to("cpu")
    input_ids = input_ids.contiguous()

    generated_tokens: List[int] = []
    current_input = input_ids.clone()

    for step in range(max_tokens):
        try:
            # Generate one token at a time (vanilla autoregressive)
            tokens, _ = model.generate_tokens(
                current_input,
                max_new_tokens=1,
                temperature=temperature,
                do_sample=do_sample,
            )

            if tokens.numel() == 0:
                break

            # Ensure tokens are on CPU and have correct shape
            if tokens.device != torch.device("cpu"):
                tokens = tokens.to("cpu")

            # Extract token ID safely
            if tokens.dim() == 2 and tokens.shape[0] > 0 and tokens.shape[1] > 0:
                token_id = tokens[0, 0].item()
            elif tokens.dim() == 1 and tokens.shape[0] > 0:
                token_id = tokens[0].item()
            else:
                logger.warning(f"Unexpected token shape: {tokens.shape}, breaking")
                break

            generated_tokens.append(token_id)

            # Append to input for next step (ensure same device)
            new_token = (
                tokens[:, :1] if tokens.dim() == 2 else tokens.unsqueeze(0).unsqueeze(1)
            )
            if new_token.device != current_input.device:
                new_token = new_token.to(current_input.device)
            current_input = torch.cat([current_input, new_token], dim=1)

        except Exception as e:
            logger.error(f"Error in vanilla generation step {step}: {e}")
            import traceback

            traceback.print_exc()
            break

    return generated_tokens


def specdec_generate(
    prompt: str,
    base_model: LanguageModel,
    draft_model: LanguageModel,
    max_tokens: int,
    k: int = 4,
    temperature: float = 1.0,
    do_sample: bool = False,
    debug: bool = True,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[int], Dict[str, any]]:
    """
    Run speculative decoding generation.

    Args:
        prompt: Input prompt text
        base_model: Base/target language model
        draft_model: Draft language model (should equal base_model for correctness test)
        max_tokens: Maximum tokens to generate
        k: Number of draft tokens per step
        temperature: Sampling temperature (ignored if do_sample=False)
        do_sample: Whether to use sampling (False for greedy)
        debug: Enable debug logging
        max_seq_len: Maximum sequence length for KV cache

    Returns:
        Tuple of (generated_tokens, debug_info)
    """
    # Set deterministic mode
    set_deterministic_mode(seed=42, device="cpu")

    # Enable KV cache for correctness testing (tests ring buffer logic)
    os.environ["SPECDEC_ENABLE_KV_APPEND"] = "1"

    # Enable deterministic mode via environment variable
    os.environ["SPECDEC_DETERMINISTIC"] = "1"

    # Enable debug logging
    if debug:
        os.environ["SPECDEC_DEBUG_PRINTS"] = "1"
        logging.getLogger("specdec").setLevel(logging.DEBUG)

    # Create pipeline with same model for draft and base
    pipeline = SpeculativePipeline(
        base_lm=base_model,
        draft_lm=draft_model,
        max_draft=k,
        device="cpu",
        seed=42,
        policy="longest_prefix",
        controller="fixed",
        controller_params={"k": k},
        max_seq_len=max_seq_len or 2048,  # Large enough for correctness testing
    )

    # Generate using speculative decoding (use generate_batch to test batch loop fixes)
    # Pass single prompt as list to use the batch path
    results = pipeline.generate_batch(
        prompts=[prompt],
        max_tokens=max_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )

    if not results:
        raise RuntimeError("No results returned from generate_batch")

    result = results[0]
    generated_tokens = result.get("generated_tokens", [])

    # Extract debug info
    debug_info = {
        "total_proposed": result.get("proposed", 0),
        "total_accepted": result.get("accepted", 0),
        "acceptance_rate": result.get("acceptance_rate", 0.0),
        "total_steps": result.get("steps", 0),
    }

    return generated_tokens, debug_info


def compare_sequences(
    vanilla_tokens: List[int],
    specdec_tokens: List[int],
    prompt: str,
) -> Tuple[bool, Optional[int], str]:
    """
    Compare two token sequences and report first divergence.

    Args:
        vanilla_tokens: Tokens from vanilla generation
        specdec_tokens: Tokens from speculative decoding
        prompt: Original prompt (for error messages)

    Returns:
        Tuple of (match, divergence_pos, error_message)
    """
    min_len = min(len(vanilla_tokens), len(specdec_tokens))

    # Check for length mismatch
    if len(vanilla_tokens) != len(specdec_tokens):
        return (
            False,
            min_len,
            f"Length mismatch: vanilla={len(vanilla_tokens)}, specdec={len(specdec_tokens)}",
        )

    # Check for token-by-token match
    for i in range(min_len):
        if vanilla_tokens[i] != specdec_tokens[i]:
            return (
                False,
                i,
                f"Token mismatch at position {i}: vanilla={vanilla_tokens[i]}, specdec={specdec_tokens[i]}",
            )

    return (True, None, "Sequences match")


def run_correctness_test(
    model_name: str = "gpt2",
    num_prompts: int = 10,
    max_tokens: int = 20,
    k: int = 4,
    max_seq_len: Optional[int] = None,
) -> bool:
    """
    Run correctness test suite.

    Args:
        model_name: Model name to use (must be small for CPU)
        num_prompts: Number of test prompts
        max_tokens: Maximum tokens to generate per prompt
        k: Number of draft tokens per step
        max_seq_len: Maximum sequence length for KV cache

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("=" * 80)
    logger.info("CPU CORRECTNESS TEST FOR SPECULATIVE DECODING")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompts: {num_prompts}")
    logger.info(f"Max tokens: {max_tokens}")
    logger.info(f"K (draft tokens): {k}")
    logger.info(f"Max seq len: {max_seq_len or 'auto'}")
    logger.info("")

    # Set threading BEFORE any model operations (critical for macOS stability)
    torch.set_num_threads(1)  # Single thread for reproducibility and stability

    # Set deterministic mode globally
    set_deterministic_mode(seed=42, device="cpu")

    # Load model (same for draft and base for correctness test)
    logger.info(f"Loading model: {model_name}")
    try:
        base_model = HFWrapper(
            model_name=model_name,
            device="cpu",
            torch_dtype=torch.float32,  # Use float32 for CPU correctness
            max_memory_mb=2000,  # Generous for CPU
        )
        draft_model = base_model  # Same model for perfect draft
        logger.info("Model loaded successfully")

        # Verify model is on CPU
        if hasattr(base_model, "_model"):
            device_check = next(base_model._model.parameters()).device
            logger.info(f"Model device: {device_check}")
            if device_check != torch.device("cpu"):
                logger.warning(f"Model not on CPU! Moving to CPU...")
                base_model._model = base_model._model.to("cpu")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test prompts (diverse set)
    test_prompts = [
        "The quick brown fox",
        "In a world where",
        "Once upon a time",
        "The answer is",
        "To be or not to be",
        "Hello, how are you",
        "The weather today is",
        "I think that",
        "According to the",
        "The most important thing",
    ][:num_prompts]

    all_passed = True
    total_divergences = 0
    total_proposed = 0
    total_accepted = 0

    for prompt_idx, prompt in enumerate(test_prompts, 1):
        logger.info("-" * 80)
        logger.info(f"Test {prompt_idx}/{len(test_prompts)}: '{prompt}'")
        logger.info("-" * 80)

        try:
            # Run vanilla generation
            logger.info("Running vanilla greedy decode...")
            vanilla_tokens = vanilla_generate(
                prompt=prompt,
                model=base_model,
                max_tokens=max_tokens,
                temperature=1.0,
                do_sample=False,  # Greedy
            )
            logger.info(f"Vanilla generated {len(vanilla_tokens)} tokens")

            # Run speculative decoding
            logger.info("Running speculative decode...")
            specdec_tokens, debug_info = specdec_generate(
                prompt=prompt,
                base_model=base_model,
                draft_model=draft_model,
                max_tokens=max_tokens,
                k=k,
                temperature=1.0,
                do_sample=False,  # Greedy
                debug=True,
                max_seq_len=max_seq_len,
            )
            logger.info(f"SpecDec generated {len(specdec_tokens)} tokens")

            # Compare sequences
            match, divergence_pos, error_msg = compare_sequences(
                vanilla_tokens, specdec_tokens, prompt
            )

            # Track metrics
            total_proposed += debug_info.get("total_proposed", 0)
            total_accepted += debug_info.get("total_accepted", 0)

            if match:
                logger.info("✓ PASSED: Sequences match")
                logger.info(f"  Proposed: {debug_info['total_proposed']}")
                logger.info(f"  Accepted: {debug_info['total_accepted']}")
                logger.info(f"  Acceptance rate: {debug_info['acceptance_rate']:.3f}")
                logger.info(f"  Steps: {debug_info['total_steps']}")
            else:
                logger.error("✗ FAILED: Sequences diverge")
                logger.error(f"  {error_msg}")
                logger.error(f"  Vanilla tokens: {vanilla_tokens[:divergence_pos+5]}")
                logger.error(f"  SpecDec tokens:  {specdec_tokens[:divergence_pos+5]}")
                all_passed = False
                total_divergences += 1

        except Exception as e:
            logger.error(f"✗ ERROR: Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False
            total_divergences += 1

    # Final summary
    acceptance_rate = total_accepted / max(total_proposed, 1)
    logger.info("")
    logger.info("=" * 80)
    if all_passed:
        logger.info("ALL TESTS PASSED — SPECULATIVE DECODING MATCHES VANILLA DECODING")
    else:
        logger.error(
            f"TESTS FAILED — {total_divergences} out of {len(test_prompts)} prompts diverged"
        )
    logger.info("=" * 80)
    logger.info("")
    logger.info("SUMMARY:")
    logger.info(
        f"specdec_run: prompts={len(test_prompts)}, max_tokens={max_tokens}, "
        f"k={k}, acceptance_rate={acceptance_rate:.2f}, "
        f"proposed={total_proposed}, accepted={total_accepted}, "
        f"deterministic=True"
    )
    logger.info("")
    return all_passed


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CPU Correctness Test for Speculative Decoding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="Model name (default: sshleifer/tiny-gpt2 for faster/safer testing, use 'gpt2' for full model)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of test prompts (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate per prompt (default: 20)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of draft tokens per step (default: 4)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for KV cache (default: auto)",
    )

    args = parser.parse_args()

    success = run_correctness_test(
        model_name=args.model,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        k=args.k,
        max_seq_len=args.max_seq_len,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
