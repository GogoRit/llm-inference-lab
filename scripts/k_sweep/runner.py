"""
K-Sweep Runner - Core execution logic for comprehensive K-sweep benchmarking.

This module contains the main run_comprehensive_k_sweep function that executes
K-sweep tests across multiple K values with batched prompt processing.
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kernels import get_kernel_info  # noqa: E402
from specdec import SpeculativePipeline  # noqa: E402

from .utils import get_system_info, resolve_device, set_deterministic_mode

logger = logging.getLogger(__name__)

# 10-prompt test suite
PROMPT_SUITE = [
    "Explain KV cache simply.",
    "What is the capital of France?",
    "Write a short poem about coding.",
    "How does machine learning work?",
    "Describe the process of photosynthesis.",
    "What are the benefits of exercise?",
    "Explain quantum computing basics.",
    "How do neural networks learn?",
    "What is the meaning of life?",
    "Describe a typical day in the life of a programmer.",
]


def run_comprehensive_k_sweep(
    base_model="gpt2",
    draft_model="distilgpt2",
    max_tokens=32,
    iterations=10,
    device="auto",
    deterministic: bool = False,
    verbose: bool = False,
    max_k: int = 4,
    single_prompt: Optional[str] = None,
    use_prompt_suite: bool = True,
    t4_warmup: bool = False,
    reuse_pipeline: bool = False,
):
    """
    Run comprehensive K-sweep test.

    Supports multiple modes:
    - Comprehensive mode: 10-prompt suite with batched processing (default)
    - Simple mode: Single prompt with single generation
    - T4-optimized mode: T4-specific warmup and memory management

    Args:
        base_model: Base model name
        draft_model: Draft model name
        max_tokens: Maximum tokens to generate
        iterations: Number of iterations per K
        device: Device to use ("auto", "cuda", "mps", "cpu")
        deterministic: Enable deterministic mode
        verbose: Enable verbose output
        max_k: Maximum K value to sweep
        single_prompt: If provided, use this single prompt instead of prompt suite
        use_prompt_suite: If True and single_prompt is None, use PROMPT_SUITE
        t4_warmup: Enable T4-specific warmup and memory management
        reuse_pipeline: Reuse pipeline across K values (more efficient, T4-style)

    Returns:
        Tuple of (results, detailed_results, run_metadata)
    """

    # Track start time for final diagnostics
    benchmark_start_time = time.time()

    # Resolve device
    resolved_device = resolve_device(device)
    logger.info(f"Using device: {resolved_device}")

    # Startup diagnostics block
    print("=" * 80, flush=True)
    print("[STARTUP] CUDA GPU Optimization Diagnostics", flush=True)
    print("=" * 80, flush=True)

    if resolved_device == "cuda" and torch.cuda.is_available():
        print(f"[STARTUP] CUDA Device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[STARTUP] CUDA Version: {torch.version.cuda}", flush=True)
        print(f"[STARTUP] PyTorch Version: {torch.__version__}", flush=True)

        # Environment variables
        batch_size = os.getenv("SPECDEC_BATCH_SIZE", "8")
        parallel_streams = os.getenv("SPECDEC_PARALLEL_STREAMS", "1")
        dtype_env = os.getenv("SPECDEC_DTYPE", "auto")
        print(f"[STARTUP] SPECDEC_BATCH_SIZE: {batch_size}", flush=True)
        print(f"[STARTUP] SPECDEC_PARALLEL_STREAMS: {parallel_streams}", flush=True)
        print(f"[STARTUP] SPECDEC_DTYPE: {dtype_env}", flush=True)

        # GPU memory
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        print(f"[STARTUP] GPU Memory: {total_mem_gb:.2f} GB total", flush=True)
        print(
            f"[STARTUP] GPU Memory: {allocated_mb:.2f} MB allocated, "
            f"{reserved_mb:.2f} MB reserved",
            flush=True,
        )
    else:
        print(f"[STARTUP] Running on {resolved_device} (not CUDA)", flush=True)

    print("=" * 80, flush=True)

    # Phase 3D: Dry-run mode (env flag only, minimal surface change)
    if os.getenv("SPECDEC_DRY_RUN", "0").lower() in ("1", "true", "yes"):
        out_dir = Path("docs/results/phase3d-dryrun")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = []
        for k in [1, 2, 3, 4]:
            draft_ms = float(5.0 * k)
            verify_ms = float(6.0 * k)
            kv_ms = 1.0
            tok_s = 1000.0 / (draft_ms + verify_ms)
            summary.append(
                {
                    "device": resolved_device,
                    "dtype": (
                        "float16" if resolved_device in ["cuda", "mps"] else "float32"
                    ),
                    "k": k,
                    "tokens_per_sec": tok_s,
                    "acceptance_rate": 0.1,
                    "draft_forward_time_ms": draft_ms,
                    "verify_forward_time_ms": verify_ms,
                    "kv_append_time_ms": kv_ms,
                    "run_id": f"phase3d_dryrun_K{k}",
                }
            )
        out_path = out_dir / f"dryrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump({"summary_results": summary}, f, indent=2)
        logger.info("Dry-run mode complete. Results saved to %s", out_path)
        # Return empty results compatible with normal flow
        return [], [], get_system_info(resolved_device)

    # Deterministic mode (optional via flag/env)
    env_det = os.getenv("SPECDEC_DETERMINISTIC", "0").lower() in ("1", "true", "yes")
    deterministic = deterministic or env_det
    set_deterministic_mode(deterministic)
    if deterministic:
        logger.info(
            "Deterministic mode: ON (fixed seeds; cudnn.deterministic; no benchmark)"
        )
    else:
        logger.info("Deterministic mode: OFF")

    # Kernel backend audit - log which verify backend is active
    kinfo = get_kernel_info()
    dtype_str = "float16" if resolved_device in ["cuda", "mps"] else "float32"
    verify_backend = kinfo.get("verify_backend", "unknown")
    logger.info(f"Using verify backend: {verify_backend}")
    logger.info(
        "Kernel backends: "
        f"verify={verify_backend}, "
        f"kv_append={kinfo.get('kv_append_backend')}, "
        f"device={resolved_device}, dtype={dtype_str}"
    )

    # Check if forced to PyTorch
    force_pytorch = os.getenv("SPECDEC_FORCE_PYTORCH_BACKEND", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if force_pytorch:
        logger.info(
            "SPECDEC_FORCE_PYTORCH_BACKEND is set - Triton verify kernel disabled"
        )

    if resolved_device == "cuda" and verify_backend not in ("cuda", "triton", "torch"):
        logger.warning(
            f"CUDA requested but verify kernel backend is {verify_backend}; "
            "this may indicate a configuration issue"
        )

    # Determine prompts to use
    if single_prompt:
        prompts_to_use = [single_prompt]
        logger.info(f"Using single prompt mode: '{single_prompt}'")
    elif use_prompt_suite:
        prompts_to_use = PROMPT_SUITE
        logger.info(f"Using prompt suite: {len(PROMPT_SUITE)} prompts")
    else:
        # Default to single prompt if neither specified
        prompts_to_use = ["Explain KV cache simply."]
        logger.info("Using default single prompt")

    results = []
    detailed_results = []

    # Cache pipelines per K to avoid reloading models
    # If reuse_pipeline is True, create one pipeline for max_k and reuse it
    pipeline_cache = {}
    shared_pipeline = None  # For reuse_pipeline mode

    # T4 warmup: If enabled, create a warmup pipeline first
    if t4_warmup and resolved_device == "cuda" and torch.cuda.is_available():
        logger.info("T4 warmup: Creating warmup pipeline...")
        try:
            warmup_pipeline = SpeculativePipeline(
                base_model=base_model,
                draft_model=draft_model,
                max_draft=max_k,
                implementation="hf",
                enable_optimization=True,
                enable_profiling=False,
                device=resolved_device,
                draft_mode="vanilla",
            )
            # Perform warmup generation
            warmup_prompt = prompts_to_use[0] if prompts_to_use else "Hello"
            logger.info(f"T4 warmup: Running warmup with '{warmup_prompt[:30]}...'")
            with torch.no_grad():
                _ = warmup_pipeline.generate_batch(
                    prompts=[warmup_prompt],
                    max_tokens=5,
                    temperature=1e-5,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            # Clear warmup pipeline
            del warmup_pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            logger.info("T4 warmup: Complete")
        except Exception as e:
            logger.warning(f"T4 warmup failed (non-fatal): {e}")

    for k in range(1, max_k + 1):  # K = 1, 2, ..., max_k
        logger.info(f"Testing K={k} (max_k={max_k})...")

        # Pipeline creation logic
        if reuse_pipeline:
            # Reuse shared pipeline across K values (T4-style)
            if shared_pipeline is None:
                logger.info(f"  Creating shared pipeline for max_k={max_k}...")
                try:
                    shared_pipeline = SpeculativePipeline(
                        base_model=base_model,
                        draft_model=draft_model,
                        max_draft=max_k,
                        implementation="hf",
                        enable_optimization=True,
                        enable_profiling=True,
                        device=resolved_device,
                        draft_mode="vanilla",
                    )
                    logger.info("  Shared pipeline created successfully")
                    # Set K via controller if available
                    if (
                        hasattr(shared_pipeline, "controller")
                        and shared_pipeline.controller
                    ):
                        if hasattr(shared_pipeline.controller, "set_fixed_k"):
                            shared_pipeline.controller.set_fixed_k(k)
                        elif hasattr(shared_pipeline.controller, "k"):
                            shared_pipeline.controller.k = k
                except Exception as e:
                    logger.error(f"  Failed to create shared pipeline: {e}")
                    shared_pipeline = None
            pipeline = shared_pipeline
            # Update K for this iteration if controller supports it
            if pipeline and hasattr(pipeline, "controller") and pipeline.controller:
                if hasattr(pipeline.controller, "set_fixed_k"):
                    pipeline.controller.set_fixed_k(k)
                elif hasattr(pipeline.controller, "k"):
                    pipeline.controller.k = k
        elif k not in pipeline_cache:
            # Create pipeline once per K (cached)
            logger.info(f"  Creating pipeline for K={k}...")
            try:
                pipeline_cache[k] = SpeculativePipeline(
                    base_model=base_model,
                    draft_model=draft_model,
                    max_draft=k,
                    implementation="hf",
                    enable_optimization=True,
                    enable_profiling=True,
                    device=resolved_device,
                    draft_mode="vanilla",  # disable Medusa/EAGLE paths in sweeps
                )
                logger.info(f"  Pipeline for K={k} created successfully")

                # GPU warmup: run dummy generation to initialize CUDA kernels
                # Skip if T4 warmup was already done
                if (
                    not t4_warmup
                    and resolved_device == "cuda"
                    and torch.cuda.is_available()
                ):
                    logger.info("  Performing GPU warmup...")
                    try:
                        # Use pipeline's generate for warmup (initializes all CUDA contexts)
                        warmup_prompt = "Hello"
                        with torch.no_grad():
                            _ = pipeline_cache[k].generate(
                                prompt=warmup_prompt,
                                max_tokens=4,  # Short warmup
                                temperature=1.0,
                                do_sample=False,
                            )
                        torch.cuda.synchronize()
                        logger.info("  GPU warmup complete")
                    except Exception as e:
                        logger.warning(f"  GPU warmup failed (continuing): {e}")
            except Exception as e:
                logger.error(f"  Failed to create pipeline for K={k}: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                pipeline_cache[k] = None

                # Skip this K entirely - record failure summary and continue to next K
                total_attempts = iterations * len(prompts_to_use)
                error_msg = f"Pipeline creation failed for K={k}: {str(e)}"

                # Add summary row for this failed K
                results.append(
                    {
                        "k": k,
                        "n_samples": 0,
                        "n_failures": total_attempts,
                        "success_rate": 0.0,
                        "latency_ms_mean": float("nan"),
                        "latency_ms_std": float("nan"),
                        "tokens_per_sec_mean": float("nan"),
                        "tokens_per_sec_std": float("nan"),
                        "acceptance_rate_mean": float("nan"),
                        "acceptance_rate_std": float("nan"),
                        "proposed_mean": float("nan"),
                        "proposed_std": float("nan"),
                        "accepted_mean": float("nan"),
                        "accepted_std": float("nan"),
                        "device": resolved_device,
                        "dtype": (
                            "float16"
                            if resolved_device in ["cuda", "mps"]
                            else "float32"
                        ),
                    }
                )

                # Add one detailed_results entry per prompt (not per iteration)
                for prompt_idx, prompt in enumerate(prompts_to_use):
                    detailed_result = {
                        "k": k,
                        "iteration": 1,  # Use iteration=1 for pipeline init failures
                        "prompt_idx": prompt_idx + 1,
                        "prompt_name": prompt,
                        "prompt": prompt,
                        "prompt_text": prompt,
                        "completion_text": "",
                        "full_text": "",
                        "completion_token_count": 0,
                        "error": error_msg,
                        "error_type": "pipeline_init",
                        "success": False,
                        "device": resolved_device,
                        "dtype": (
                            "float16"
                            if resolved_device in ["cuda", "mps"]
                            else "float32"
                        ),
                    }
                    detailed_results.append(detailed_result)

                logger.warning(
                    f"  Skipping K={k} entirely due to pipeline creation failure. "
                    f"Recorded {len(prompts_to_use)} failure entries."
                )
                continue  # Skip to next K without running iterations

        k_results = []
        k_failures = 0

        # OPTIMIZATION: Use batching to process multiple prompts at once
        # For single prompt mode, use batch_size=1 (single generation)
        # For prompt suite, use configured batch size
        if len(prompts_to_use) == 1:
            BATCH_SIZE = 1  # Single prompt = single generation
            use_batch = False
        else:
            BATCH_SIZE = int(os.getenv("SPECDEC_BATCH_SIZE", "8"))
            use_batch = True
        logger.info(f"  Using batch size: {BATCH_SIZE} (batch mode: {use_batch})")

        # Heartbeat tracking
        last_heartbeat_time = time.time()

        for iteration in range(iterations):
            if verbose:
                print(
                    f"[ITER] ===== Starting Iteration {iteration+1}/{iterations} =====",
                    flush=True,
                )
            logger.info(f"  Iteration {iteration+1}/{iterations}")
            iter_start_time = time.time()
            iter_results = []
            iter_samples = 0

            # Process prompts in batches (or single generation for single prompt)
            if use_batch:
                num_batches = (len(prompts_to_use) + BATCH_SIZE - 1) // BATCH_SIZE
                if verbose:
                    print(
                        f"[ITER] Processing {num_batches} batches of up to {BATCH_SIZE} prompts each",
                        flush=True,
                    )
            else:
                num_batches = 1
                if verbose:
                    print(
                        f"[ITER] Processing single prompt (no batching)",
                        flush=True,
                    )

            for batch_idx, batch_start in enumerate(
                range(0, len(prompts_to_use), BATCH_SIZE)
            ):
                batch_end = min(batch_start + BATCH_SIZE, len(prompts_to_use))
                batch_prompts = prompts_to_use[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))

                # Heartbeat check (every 60 seconds) - only if verbose
                current_time = time.time()
                if verbose and current_time - last_heartbeat_time >= 60.0:
                    print(
                        f"[HEARTBEAT] {time.strftime('%H:%M:%S')} still running... "
                        f"K={k}, Iter={iteration+1}/{iterations}, "
                        f"Batch={batch_idx+1}/{num_batches}",
                        flush=True,
                    )
                    last_heartbeat_time = current_time

                # GPU utilization estimate (memory-based proxy - NOT actual compute)
                # NOTE: This is a flawed metric - actual GPU compute utilization
                # requires nvidia-smi. Memory allocation != compute utilization
                # (GPU can be 100% utilized with low memory)
                if resolved_device == "cuda" and torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.memory_allocated() / (1024**2)
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                    # Memory-based estimate (not accurate for compute utilization)
                    gpu_util_est = min(100.0, (gpu_mem_mb / (gpu_mem_gb * 1024)) * 100)
                elif resolved_device == "mps" and torch.backends.mps.is_available():
                    # MPS GPU memory tracking (if available in PyTorch)
                    try:
                        # PyTorch may not expose MPS memory stats directly
                        # Use a simple heuristic: assume GPU is active if we're on MPS
                        gpu_util_est = 50.0  # Conservative estimate for MPS
                    except Exception:
                        gpu_util_est = 0.0
                else:
                    gpu_util_est = 0.0

                if verbose:
                    print(
                        f"[INFO] Iteration {iteration+1}/{iterations} | "
                        f"Batch {batch_idx+1}/{num_batches} (prompts {batch_start+1}-{batch_end}) | "
                        f"GPU util: {gpu_util_est:.1f}% | "
                        f"Elapsed: {time.time() - iter_start_time:.2f}s",
                        flush=True,
                    )

                try:
                    # Get pipeline (either from cache or shared)
                    if reuse_pipeline:
                        pipeline = shared_pipeline
                        if pipeline is None:
                            logger.error(
                                f"  Shared pipeline is None for K={k}, skipping"
                            )
                            k_failures += len(batch_prompts)
                            continue
                    else:
                        # Use cached pipeline
                        # Note: pipeline_cache[k] should never be None here because
                        # we skip K entirely if pipeline creation fails (see above)
                        pipeline = pipeline_cache[k]

                    # GPU sync before batch to ensure clean state
                    if resolved_device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Generate for batch
                    batch_start_time = time.time()
                    batch_results = pipeline.generate_batch(
                        prompts=batch_prompts,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True,
                    )

                    # GPU sync after batch to measure actual GPU time
                    if resolved_device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    batch_end_time = time.time()

                    batch_time_ms = (batch_end_time - batch_start_time) * 1000
                    batch_time_s = batch_end_time - batch_start_time

                    # Post-batch GPU utilization
                    if resolved_device == "cuda" and torch.cuda.is_available():
                        gpu_mem_mb_after = torch.cuda.memory_allocated() / (1024**2)
                        gpu_util_est_after = min(
                            100.0, (gpu_mem_mb_after / (gpu_mem_gb * 1024)) * 100
                        )
                    elif resolved_device == "mps" and torch.backends.mps.is_available():
                        # MPS: assume active utilization after batch processing
                        gpu_util_est_after = 60.0  # Conservative estimate
                    else:
                        gpu_util_est_after = gpu_util_est

                    if verbose:
                        print(
                            f"[INFO] Batch {batch_idx+1} completed | "
                            f"Time: {batch_time_s:.2f}s ({batch_time_ms:.1f}ms) | "
                            f"GPU util: {gpu_util_est_after:.1f}% | "
                            f"Prompts: {len(batch_prompts)}",
                            flush=True,
                        )
                        print(
                            f"[BATCH] Processing {len(batch_results)} results from batch {batch_idx+1}",
                            flush=True,
                        )

                    for prompt_idx, result in zip(batch_indices, batch_results):
                        prompt = prompts_to_use[prompt_idx]

                        # Extract metrics with validation
                        latency_ms = result.get("latency_ms", 0)
                        tokens_per_sec = result.get("tokens_per_sec", 0)
                        acceptance_rate = result.get("acceptance_rate", 0.0)
                        proposed = result.get("proposed", 0)
                        accepted = result.get("accepted", 0)
                        generated_tokens_list = result.get("generated_tokens", [])
                        if not isinstance(generated_tokens_list, list):
                            generated_tokens_list = []
                        generated_tokens_count = len(generated_tokens_list)
                        kv_appended = result.get("kv_appended_tokens_total", 0)
                        kv_append_time = result.get("kv_append_time_ms", 0.0)
                        kv_append_enabled = result.get("kv_append_enabled", False)
                        kv_append_backend = result.get("kv_append_backend", "unknown")

                        # Validate metrics are not NaN or inf
                        if np.isnan(tokens_per_sec) or np.isinf(tokens_per_sec):
                            tokens_per_sec = 0.0
                        if np.isnan(latency_ms) or np.isinf(latency_ms):
                            latency_ms = 0.0
                        if np.isnan(acceptance_rate) or np.isinf(acceptance_rate):
                            acceptance_rate = 0.0

                        # Separate prompt and completion text
                        # Get tokenizer from pipeline
                        tokenizer = pipeline.base_lm.tokenizer

                        # Tokenize the prompt exactly as it was done before calling the model
                        # For a single string, input_ids is a list of token IDs
                        prompt_encoded = tokenizer(prompt, add_special_tokens=False)
                        prompt_ids = prompt_encoded.input_ids
                        # Ensure prompt_ids is a flat list (handle case where it might be nested)
                        if prompt_ids and isinstance(prompt_ids[0], list):
                            prompt_ids = prompt_ids[0]

                        # generated_tokens_list contains only the generated tokens (no prompt)
                        completion_ids = generated_tokens_list

                        # Decode separately
                        prompt_text = prompt  # The actual prompt text
                        completion_text = (
                            tokenizer.decode(completion_ids, skip_special_tokens=True)
                            if completion_ids
                            else ""
                        )

                        # Also decode full sequence for debugging (optional)
                        full_ids = prompt_ids + completion_ids
                        full_text = (
                            tokenizer.decode(full_ids, skip_special_tokens=True)
                            if full_ids
                            else ""
                        )

                        # Log progress for each prompt (only if verbose)
                        if verbose:
                            print(
                                f"[PROMPT] {prompt_idx+1}/{len(prompts_to_use)} | "
                                f"Tokens: {generated_tokens_count} | "
                                f"Accept: {accepted}/{proposed} ({acceptance_rate:.1%}) | "
                                f"Throughput: {tokens_per_sec:.2f} tok/s",
                                flush=True,
                            )

                        # Store detailed result with separate prompt and completion fields
                        detailed_result = {
                            "k": k,
                            "iteration": iteration + 1,
                            "prompt_idx": prompt_idx + 1,
                            "prompt_name": prompt,  # Short label (kept for backward compatibility)
                            "prompt": prompt,  # Keep for backward compatibility
                            "prompt_text": prompt_text,  # Full prompt text
                            "completion_text": completion_text,  # Only the generated continuation
                            "full_text": full_text,  # Full sequence for debugging
                            "completion_token_count": generated_tokens_count,  # Number of generated tokens
                            "latency_ms": latency_ms,
                            "tokens_per_sec": tokens_per_sec,
                            "acceptance_rate": acceptance_rate,
                            "proposed": proposed,
                            "accepted": accepted,
                            "kv_appended_tokens": kv_appended,
                            "kv_append_time_ms": kv_append_time,
                            "kv_append_enabled": kv_append_enabled,
                            "kv_append_backend": kv_append_backend,
                            # For backward compatibility, `text` mirrors the generated completion (truncated).
                            "text": (
                                completion_text[:100] + "..."
                                if len(completion_text) > 100
                                else completion_text
                            ),
                            "success": True,
                            "device": resolved_device,
                            "dtype": (
                                "float16"
                                if resolved_device in ["cuda", "mps"]
                                else "float32"
                            ),
                            "batch_size": result.get("batch_size", 1),
                        }
                        detailed_results.append(detailed_result)

                        # Store for K-level aggregation
                        result_data = {
                            "latency_ms": latency_ms,
                            "tokens_per_sec": tokens_per_sec,
                            "acceptance_rate": acceptance_rate,
                            "proposed": proposed,
                            "accepted": accepted,
                            "kv_appended_tokens": kv_appended,
                            "kv_append_time_ms": kv_append_time,
                        }
                        k_results.append(result_data)
                        iter_results.append(result_data)
                        iter_samples += 1

                except AssertionError as e:
                    # CUDA assertion errors (NaN/inf probability tensors)
                    error_msg = str(e)
                    if (
                        "probability tensor" in error_msg.lower()
                        or "inf/nan" in error_msg.lower()
                    ):
                        print(
                            f"[ERROR] CUDA Assertion (NaN/Inf) in batch {batch_idx+1}: {error_msg}",
                            flush=True,
                        )
                        print(
                            f"[ERROR] Skipping batch {batch_idx+1} gracefully (batch {batch_start+1}-{batch_end})",
                            flush=True,
                        )
                    else:
                        logger.error(f"Assertion error: {e}")
                        print(
                            f"[ERROR] Assertion error in batch {batch_idx+1}: {e}",
                            flush=True,
                        )

                    k_failures += len(batch_prompts)
                    # Record failure for each prompt in batch
                    for prompt_idx in batch_indices:
                        detailed_result = {
                            "k": k,
                            "iteration": iteration + 1,
                            "prompt_idx": prompt_idx + 1,
                            "prompt": prompts_to_use[prompt_idx],
                            "error": f"CUDA Assertion: {error_msg}",
                            "error_type": "cuda_assertion",
                            "success": False,
                            "device": resolved_device,
                            "dtype": (
                                "float16"
                                if resolved_device in ["cuda", "mps"]
                                else "float32"
                            ),
                        }
                        detailed_results.append(detailed_result)
                    continue  # Skip to next batch

                except Exception as e:
                    k_failures += len(batch_prompts)
                    error_traceback = traceback.format_exc()
                    error_type = type(e).__name__
                    print(
                        f"[ERROR] Exception in batch {batch_idx+1} ({error_type}): {e}",
                        flush=True,
                    )
                    logger.error(
                        f"      K={k}, Iter={iteration+1}, Batch {batch_start}-{batch_end} "
                        f"failed: {e}"
                    )
                    logger.error(f"      Traceback: {error_traceback}")

                    # Record failure for each prompt in batch
                    for prompt_idx in batch_indices:
                        prompt = PROMPT_SUITE[prompt_idx]
                        detailed_result = {
                            "k": k,
                            "iteration": iteration + 1,
                            "prompt_idx": prompt_idx + 1,
                            "prompt": prompt,
                            "error": str(e),
                            "traceback": error_traceback,
                            "success": False,
                            "device": resolved_device,
                            "dtype": (
                                "float16"
                                if resolved_device in ["cuda", "mps"]
                                else "float32"
                            ),
                        }
                        detailed_results.append(detailed_result)

            # Iteration summary (always print, even if no results)
            iter_elapsed = time.time() - iter_start_time

            if iter_results:
                iter_throughputs = [
                    r["tokens_per_sec"] for r in iter_results if "tokens_per_sec" in r
                ]
                iter_accept_rates = [
                    r["acceptance_rate"] for r in iter_results if "acceptance_rate" in r
                ]
                iter_latencies = [
                    r["latency_ms"] for r in iter_results if "latency_ms" in r
                ]
                avg_throughput = np.mean(iter_throughputs) if iter_throughputs else 0.0
                avg_accept_rate = (
                    np.mean(iter_accept_rates) if iter_accept_rates else 0.0
                )
                avg_latency = np.mean(iter_latencies) if iter_latencies else 0.0

                print(
                    f"[SUMMARY] Iteration {iteration+1}/{iterations} COMPLETE: "
                    f"{iter_samples}/{len(prompts_to_use)} samples | "
                    f"avg throughput={avg_throughput:.2f} tok/s | "
                    f"acceptance={avg_accept_rate:.2%} | "
                    f"avg latency={avg_latency:.1f}ms | "
                    f"elapsed={iter_elapsed:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[SUMMARY] Iteration {iteration+1}/{iterations} COMPLETE: "
                    f"0/{len(prompts_to_use)} samples (all failed) | "
                    f"elapsed={iter_elapsed:.2f}s",
                    flush=True,
                )

                if verbose:
                    print(
                        f"[ITER] ===== Iteration {iteration+1}/{iterations} finished =====",
                        flush=True,
                    )

        # Calculate statistics for this K
        valid_results = [r for r in k_results if "error" not in r]
        total_attempts = iterations * len(prompts_to_use)

        if valid_results:
            latencies = [r["latency_ms"] for r in valid_results]
            throughputs = [r["tokens_per_sec"] for r in valid_results]
            acceptance_rates = [r["acceptance_rate"] for r in valid_results]
            proposed_counts = [r["proposed"] for r in valid_results]
            accepted_counts = [r["accepted"] for r in valid_results]
            kv_appended_counts = [r["kv_appended_tokens"] for r in valid_results]
            kv_append_times = [r["kv_append_time_ms"] for r in valid_results]

            # Clamp acceptance rate mean to [0.0, 1.0] for human readability
            # (raw values in detailed_results remain unclamped for debugging)
            acceptance_rate_mean = (
                float(np.mean(acceptance_rates)) if acceptance_rates else 0.0
            )
            acceptance_rate_mean = max(0.0, min(1.0, acceptance_rate_mean))

            results.append(
                {
                    "k": k,
                    "n_samples": len(valid_results),
                    "n_failures": k_failures,
                    "success_rate": len(valid_results) / total_attempts,
                    "latency_ms_mean": np.mean(latencies),
                    "latency_ms_std": np.std(latencies),
                    "tokens_per_sec_mean": np.mean(throughputs),
                    "tokens_per_sec_std": np.std(throughputs),
                    "acceptance_rate_mean": acceptance_rate_mean,
                    "acceptance_rate_std": np.std(acceptance_rates),
                    "kv_appended_tokens_mean": np.mean(kv_appended_counts),
                    "kv_appended_tokens_std": np.std(kv_appended_counts),
                    "kv_append_time_ms_mean": np.mean(kv_append_times),
                    "kv_append_time_ms_std": np.std(kv_append_times),
                    "proposed_mean": np.mean(proposed_counts),
                    "proposed_std": np.std(proposed_counts),
                    "accepted_mean": np.mean(accepted_counts),
                    "accepted_std": np.std(accepted_counts),
                    "device": resolved_device,
                    "dtype": (
                        "float16" if resolved_device in ["cuda", "mps"] else "float32"
                    ),
                }
            )

            # K-level summary with real values
            print(
                f"\n[K-SUMMARY] K={k} Results:",
                flush=True,
            )
            print(
                f"  Throughput: {np.mean(throughputs):.2f}±{np.std(throughputs):.2f} tok/s",
                flush=True,
            )
            print(
                f"  Acceptance: {np.mean(acceptance_rates):.3%}±{np.std(acceptance_rates):.3%}",
                flush=True,
            )
            print(
                f"  Latency: {np.mean(latencies):.1f}±{np.std(latencies):.1f} ms",
                flush=True,
            )
            print(
                f"  Success: {len(valid_results)}/{total_attempts} ({len(valid_results)/total_attempts:.1%})",
                flush=True,
            )
            print(
                f"  Proposed: {np.mean(proposed_counts):.1f}±{np.std(proposed_counts):.1f}",
                flush=True,
            )
            print(
                f"  Accepted: {np.mean(accepted_counts):.1f}±{np.std(accepted_counts):.1f}",
                flush=True,
            )
            print("", flush=True)

            logger.info(
                f"  K={k} AVERAGE: "
                f"{np.mean(throughputs):.2f}±{np.std(throughputs):.2f} "
                f"tok/s, {np.mean(acceptance_rates):.3f}±"
                f"{np.std(acceptance_rates):.3f} accept rate, "
                f"success rate: {len(valid_results)}/{total_attempts} "
                f"({len(valid_results)/total_attempts:.1%})"
            )
        else:
            # Still write a row even if all failed
            results.append(
                {
                    "k": k,
                    "n_samples": 0,
                    "n_failures": k_failures,
                    "success_rate": 0.0,
                    "latency_ms_mean": float("nan"),
                    "latency_ms_std": float("nan"),
                    "tokens_per_sec_mean": float("nan"),
                    "tokens_per_sec_std": float("nan"),
                    "acceptance_rate_mean": float("nan"),
                    "acceptance_rate_std": float("nan"),
                    "proposed_mean": float("nan"),
                    "proposed_std": float("nan"),
                    "accepted_mean": float("nan"),
                    "accepted_std": float("nan"),
                    "device": resolved_device,
                    "dtype": (
                        "float16" if resolved_device in ["cuda", "mps"] else "float32"
                    ),
                }
            )
            logger.error(
                f"  K={k}: All {total_attempts} attempts failed ({k_failures} failures)"
            )
            print(
                f"\n[K-SUMMARY] K={k} Results: ALL FAILED ({k_failures} failures)",
                flush=True,
            )
            print("", flush=True)

    # Final diagnostics summary
    total_runtime = time.time() - benchmark_start_time
    total_runtime_min = total_runtime / 60.0

    print("\n" + "=" * 80, flush=True)
    print("[FINAL] Benchmark Complete - Final Diagnostics", flush=True)
    print("=" * 80, flush=True)

    if resolved_device == "cuda" and torch.cuda.is_available():
        gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        gpu_mem_current_mb = torch.cuda.memory_allocated() / (1024**2)
        gpu_mem_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(
            f"[FINAL] GPU Memory Peak: {gpu_mem_peak_mb:.2f} MB / "
            f"{gpu_mem_total_gb:.2f} GB",
            flush=True,
        )
        print(f"[FINAL] GPU Memory Current: {gpu_mem_current_mb:.2f} MB", flush=True)
        # NOTE: This is memory-based, NOT actual compute utilization
        # For real GPU utilization, use: nvidia-smi dmon -s u
        gpu_util_mem_based = (gpu_mem_peak_mb / (gpu_mem_total_gb * 1024)) * 100
        print(
            f"[FINAL] GPU Memory Utilization: {gpu_util_mem_based:.1f}% "
            "(NOTE: This is memory-based, not compute utilization. "
            "Use 'nvidia-smi' for actual GPU%)",
            flush=True,
        )

    print(
        f"[FINAL] Total Runtime: {total_runtime_min:.2f} minutes "
        f"({total_runtime:.2f}s)",
        flush=True,
    )
    k_values_tested = len([r for r in results if r.get("n_samples", 0) > 0])
    print(
        f"[FINAL] Total K Values Tested: {k_values_tested}",
        flush=True,
    )

    # Calculate overall stats
    all_valid_results = [r for r in results if r.get("n_samples", 0) > 0]
    if all_valid_results:
        overall_throughput = np.mean(
            [r["tokens_per_sec_mean"] for r in all_valid_results]
        )
        overall_acceptance = np.mean(
            [r["acceptance_rate_mean"] for r in all_valid_results]
        )
        overall_success = np.mean([r["success_rate"] for r in all_valid_results])

        print(f"[FINAL] Overall Throughput: {overall_throughput:.2f} tok/s", flush=True)
        print(f"[FINAL] Overall Acceptance: {overall_acceptance:.2%}", flush=True)
        print(f"[FINAL] Overall Success Rate: {overall_success:.1%}", flush=True)

    print("=" * 80 + "\n", flush=True)

    # Cleanup pipeline resources to prevent hanging in Kaggle
    print("[CLEANUP] Cleaning up pipeline resources...", flush=True)
    for k, pipeline in pipeline_cache.items():
        if pipeline is not None:
            try:
                # Clear any CUDA streams or resources
                if hasattr(pipeline, "scheduler") and pipeline.scheduler:
                    if hasattr(pipeline.scheduler, "reset"):
                        pipeline.scheduler.reset()
                # Clear KV caches
                if hasattr(pipeline, "kv_cache_manager"):
                    pipeline.kv_cache_manager.reset()
                if hasattr(pipeline.base_lm, "clear_kv_cache"):
                    pipeline.base_lm.clear_kv_cache()
                if hasattr(pipeline.draft_lm, "clear_kv_cache"):
                    pipeline.draft_lm.clear_kv_cache()
            except Exception as e:
                logger.warning(f"Error cleaning up pipeline K={k}: {e}")
    pipeline_cache.clear()
    # Force CUDA cleanup
    if resolved_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[CLEANUP] Pipeline resources cleaned up", flush=True)

    return (
        results,
        detailed_results,
        {"kernel_info": kinfo, "deterministic": deterministic},
    )
