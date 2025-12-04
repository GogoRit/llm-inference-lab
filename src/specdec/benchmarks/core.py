"""
Core benchmarking functions for speculative decoding.

Provides reusable benchmark logic that can be used by both single-run
and grid-run scripts.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from ..core.pipeline import SpeculativePipeline
from ..models.hf_wrappers import HFWrapper

logger = logging.getLogger(__name__)


def validate_device(device: str) -> str:
    """
    Validate device availability and return actual device.

    Args:
        device: Requested device (cpu, cuda, mps, auto)

    Returns:
        Validated device string

    Raises:
        ValueError: If device is not available
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but not available. Use --device cpu or --device auto"
            )
        return "cuda"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS requested but not available. Use --device cpu or --device auto"
            )
        return "mps"
    elif device == "cpu":
        return "cpu"
    elif device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        raise ValueError(f"Invalid device: {device}. Must be cpu, cuda, mps, or auto")


def generate_synthetic_prompts(num_prompts: int) -> List[str]:
    """
    Generate synthetic prompts for benchmarking.

    Args:
        num_prompts: Number of prompts to generate

    Returns:
        List of prompt strings
    """
    return [f"Benchmark prompt {i}" for i in range(num_prompts)]


def run_vanilla_batch(
    pipeline: SpeculativePipeline,
    prompts: List[str],
    max_tokens: int,
    device: str,
) -> Tuple[List[List[int]], float]:
    """
    Run vanilla greedy decoding for a batch of prompts.

    Args:
        pipeline: Pipeline instance (with speculative disabled)
        prompts: List of prompt strings
        max_tokens: Maximum tokens to generate per prompt
        device: Device string for synchronization

    Returns:
        Tuple of (list of generated token lists, elapsed time in seconds)
    """
    # Disable speculative decoding for vanilla baseline
    original_speculative = pipeline.speculative_enabled
    pipeline.speculative_enabled = False

    try:
        # Synchronize if CUDA
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        # Run baseline generation
        results = pipeline.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=1.0,
            do_sample=False,  # Greedy
        )

        # Synchronize if CUDA
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_secs = time.time() - start_time

        # Extract generated tokens
        generated_tokens = [r.get("generated_tokens", []) for r in results]

        return generated_tokens, elapsed_secs

    finally:
        # Restore original speculative setting
        pipeline.speculative_enabled = original_speculative


def run_specdec_batch(
    pipeline: SpeculativePipeline,
    prompts: List[str],
    max_tokens: int,
    device: str,
) -> Tuple[List[List[int]], float, Dict[str, float]]:
    """
    Run speculative decoding for a batch of prompts.

    Args:
        pipeline: Pipeline instance (with speculative enabled)
        prompts: List of prompt strings
        max_tokens: Maximum tokens to generate per prompt
        device: Device string for synchronization

    Returns:
        Tuple of (list of generated token lists, elapsed time in seconds, metrics dict)
    """
    # Ensure speculative decoding is enabled
    pipeline.speculative_enabled = True

    # Synchronize if CUDA
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    # Run speculative generation
    results = pipeline.generate_batch(
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=1.0,
        do_sample=False,  # Greedy
    )

    # Synchronize if CUDA
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_secs = time.time() - start_time

    # Extract generated tokens and metrics
    generated_tokens = [r.get("generated_tokens", []) for r in results]

    # Aggregate metrics from all results
    total_proposed = sum(r.get("proposed", 0) for r in results)
    total_accepted = sum(r.get("accepted", 0) for r in results)
    acceptance_rate = total_accepted / max(total_proposed, 1)

    metrics = {
        "acceptance_rate": acceptance_rate,
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
    }

    return generated_tokens, elapsed_secs, metrics


def run_single_benchmark(
    model_name: str,
    draft_model_name: Optional[str],
    device: str,
    impl: str,
    batch_size: int,
    max_tokens: int,
    k: int,
    num_prompts: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one benchmark config and return metrics as a dict.

    Args:
        model_name: Base model name
        draft_model_name: Draft model name (None = same as base for perfect draft)
        device: Device to run on (validated)
        impl: Implementation type (fake, torch) - "torch" maps to "hf"
        batch_size: Batch size for processing
        max_tokens: Maximum tokens per prompt
        k: Draft tokens per step
        num_prompts: Total number of prompts to test
        verbose: Whether to print detailed progress logs

    Returns:
        Dictionary with benchmark results containing:
        - model_name, draft_model_name, device, impl
        - batch_size, max_tokens, k, num_prompts
        - total_tokens_vanilla, total_time_vanilla, vanilla_tok_s
        - total_tokens_specdec, total_time_specdec, specdec_tok_s
        - acceptance_rate, total_proposed, total_accepted
        - speedup
    """
    if verbose:
        logger.info("=" * 80)
        logger.info("BENCHMARK: Speculative Decoding vs Vanilla Decoding")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Implementation: {impl}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"K (draft tokens): {k}")
        logger.info(f"Total prompts: {num_prompts}")
        logger.info("")

    # Generate prompts
    all_prompts = generate_synthetic_prompts(num_prompts)

    # Load models
    if verbose:
        logger.info("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = HFWrapper(model_name=model_name, device=device, tokenizer=tokenizer)
    draft_model_name_actual = draft_model_name or model_name
    draft_model = HFWrapper(
        model_name=draft_model_name_actual, device=device, tokenizer=tokenizer
    )

    # Create pipeline
    # Map "torch" to "hf" (pipeline uses "hf" for HuggingFace models)
    impl_mapped = "hf" if impl == "torch" else impl
    if verbose:
        logger.info("Creating pipeline...")
    pipeline = SpeculativePipeline(
        base_lm=base_model,
        draft_lm=draft_model,
        max_draft=k,
        device=device,
        seed=42,
        implementation=impl_mapped,
        policy="longest_prefix",
        controller="fixed",
        controller_params={"k": k},
        max_seq_len=2048,
    )

    # Disable deterministic mode for performance benchmarks
    pipeline.deterministic_mode = False

    # Split prompts into batches
    batches = [
        all_prompts[i : i + batch_size] for i in range(0, len(all_prompts), batch_size)
    ]

    # ===== VANILLA BASELINE =====
    if verbose:
        logger.info("Running vanilla baseline...")
    vanilla_tokens_all = []
    vanilla_time_total = 0.0

    for batch_idx, batch_prompts in enumerate(batches):
        if verbose:
            logger.info(
                f"  Vanilla batch {batch_idx + 1}/{len(batches)} ({len(batch_prompts)} prompts)"
            )
        batch_tokens, batch_time = run_vanilla_batch(
            pipeline, batch_prompts, max_tokens, device
        )
        vanilla_tokens_all.extend(batch_tokens)
        vanilla_time_total += batch_time

    total_tokens_vanilla = sum(len(tokens) for tokens in vanilla_tokens_all)
    vanilla_tok_s: float = total_tokens_vanilla / max(vanilla_time_total, 0.001)

    if verbose:
        logger.info(
            f"  Vanilla: {total_tokens_vanilla} tokens in {vanilla_time_total:.2f}s = {vanilla_tok_s:.2f} tok/s"
        )

    # ===== SPECULATIVE DECODING =====
    if verbose:
        logger.info("Running speculative decoding...")
    specdec_tokens_all = []
    specdec_time_total = 0.0
    total_proposed = 0
    total_accepted = 0

    for batch_idx, batch_prompts in enumerate(batches):
        if verbose:
            logger.info(
                f"  SpecDec batch {batch_idx + 1}/{len(batches)} ({len(batch_prompts)} prompts)"
            )
        batch_tokens, batch_time, batch_metrics = run_specdec_batch(
            pipeline, batch_prompts, max_tokens, device
        )
        specdec_tokens_all.extend(batch_tokens)
        specdec_time_total += batch_time
        total_proposed += int(batch_metrics["total_proposed"])
        total_accepted += int(batch_metrics["total_accepted"])

    total_tokens_specdec = sum(len(tokens) for tokens in specdec_tokens_all)
    specdec_tok_s: float = total_tokens_specdec / max(specdec_time_total, 0.001)
    acceptance_rate = total_accepted / max(total_proposed, 1)

    if verbose:
        logger.info(
            f"  SpecDec: {total_tokens_specdec} tokens in {specdec_time_total:.2f}s = {specdec_tok_s:.2f} tok/s"
        )
        logger.info(
            f"  Acceptance rate: {acceptance_rate:.3f}, proposed: {total_proposed}, accepted: {total_accepted}"
        )

    # Calculate speedup (specdec_tok_s / vanilla_tok_s)
    # > 1.0 means SpecDec is faster, < 1.0 means vanilla is faster
    speedup = specdec_tok_s / max(vanilla_tok_s, 0.001)

    # Build results dictionary
    results = {
        "model_name": model_name,
        "draft_model_name": draft_model_name_actual,
        "device": device,
        "impl": impl,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "k": k,
        "num_prompts": num_prompts,
        "total_tokens_vanilla": total_tokens_vanilla,
        "total_time_vanilla": vanilla_time_total,
        "vanilla_tok_s": vanilla_tok_s,
        "total_tokens_specdec": total_tokens_specdec,
        "total_time_specdec": specdec_time_total,
        "specdec_tok_s": specdec_tok_s,
        "acceptance_rate": acceptance_rate,
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "speedup": speedup,
    }

    return results
