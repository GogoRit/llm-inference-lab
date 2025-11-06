#!/usr/bin/env python3
"""
Comprehensive K-Sweep Testing Script for Speculative Decoding
Tests K=1-4 with 10 iterations on 10-prompt suite, generates detailed results and plots.
"""

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Force unbuffered output for Kaggle/notebook environments
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Silence tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path before importing project modules
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Configure PyTorch for CUDA inference
if torch.cuda.is_available():
    # Set precision flags for optimal performance
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Suppress CuBLAS deterministic warnings by setting workspace config
    # This is required for deterministic algorithms with CUDA >= 10.2
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Enable deterministic algorithms with warnings only
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Set CUDA arch list to avoid compilation warnings
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        # Auto-detect from visible CUDA devices
        device_capability = torch.cuda.get_device_capability(0)
        compute_capability = f"{device_capability[0]}{device_capability[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = compute_capability

    # Note: CUDA_LAUNCH_BLOCKING should be set by user if needed for debugging
    # We do NOT auto-enable it as it disables async stream optimization
    # For production performance, leave it unset or set to "0"
    if os.getenv("CUDA_LAUNCH_BLOCKING") == "1":
        print(
            "[STARTUP] WARNING: CUDA_LAUNCH_BLOCKING=1 is set - "
            "async streams will be disabled! "
            "This will reduce GPU utilization and throughput. "
            "Remove for production runs.",
            flush=True,
        )
    else:
        print(
            "[STARTUP] CUDA_LAUNCH_BLOCKING not set - "
            "async streams enabled for optimal performance",
            flush=True,
        )

    # Reset dynamo and disable dynamic shape capture (fixes graph capture issues)
    try:
        import torch._dynamo  # noqa: E402

        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
    except ImportError:
        pass  # torch._dynamo not available in older PyTorch versions

from kernels import get_kernel_info  # noqa: E402
from specdec import SpeculativePipeline  # noqa: E402

# Set up logging with explicit stream configuration for Kaggle
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Explicitly use stdout for Kaggle visibility
    force=True,  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# Ensure initial output is visible
print("=" * 80, flush=True)
print("[SCRIPT] Comprehensive K-Sweep Script Starting", flush=True)
print("=" * 80, flush=True)
sys.stdout.flush()

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


def get_system_info(device):
    """Get system and environment metadata."""
    # Get kernel backend info
    kinfo = get_kernel_info()

    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available()
            else (
                "MPS"
                if device == "mps" and torch.backends.mps.is_available()
                else "CPU"
            )
        ),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "dtype": "float16" if device in ["cuda", "mps"] else "float32",
        "kernel_backends": {
            "verify": kinfo.get("verify_backend", "unknown"),
            "kv_append": kinfo.get("kv_append_backend", "unknown"),
        },
        "kv_append_enabled": os.getenv("SPECDEC_ENABLE_KV_APPEND", "1").lower()
        in ("1", "true", "yes"),
        "batch_size": int(os.getenv("SPECDEC_BATCH_SIZE", "8")),
        "parallel_streams": os.getenv("SPECDEC_PARALLEL_STREAMS", "1").lower()
        in ("1", "true", "yes"),
        "cuda_graph": os.getenv("SPECDEC_CUDA_GRAPH", "0").lower()
        in ("1", "true", "yes"),
    }

    # Add GPU memory info if CUDA
    if device == "cuda" and torch.cuda.is_available():
        info["cuda_total_memory_gb"] = torch.cuda.get_device_properties(
            0
        ).total_memory / (1024**3)
        info["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        info["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

    return info


def resolve_device(device_arg):
    """Resolve device argument to actual device. MPS-first for testing."""
    if device_arg == "auto":
        # Test MPS first, then CUDA (MPS-first approach for development)
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_arg


def set_deterministic_mode(enable: bool):
    """Enable reproducible behavior across libraries."""
    if not enable:
        return
    try:
        import random

        import numpy as np
        import torch

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.warning(f"Failed to set deterministic mode: {e}")


def run_comprehensive_k_sweep(
    base_model="gpt2",
    draft_model="distilgpt2",
    max_tokens=32,
    iterations=10,
    device="auto",
    deterministic: bool = False,
):
    """Run comprehensive K-sweep test with 10-prompt suite."""

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

    # Kernel backend audit (single line)
    kinfo = get_kernel_info()
    dtype_str = "float16" if resolved_device in ["cuda", "mps"] else "float32"
    logger.info(
        "Kernel backends: "
        f"verify={kinfo.get('verify_backend')}, "
        f"kv_append={kinfo.get('kv_append_backend')}, "
        f"device={resolved_device}, dtype={dtype_str}"
    )
    if resolved_device == "cuda" and kinfo.get("verify_backend") != "cuda":
        logger.warning(
            "CUDA requested but verify kernel backend is not CUDA; falling back to "
            f"{kinfo.get('verify_backend')}"
        )

    results = []
    detailed_results = []

    # Cache pipelines per K to avoid reloading models
    pipeline_cache = {}

    for k in range(1, 5):  # K = 1, 2, 3, 4
        logger.info(f"Testing K={k}...")

        # Create pipeline once per K (cached)
        if k not in pipeline_cache:
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
                if resolved_device == "cuda" and torch.cuda.is_available():
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

        k_results = []
        k_failures = 0

        # OPTIMIZATION: Use batching to process multiple prompts at once
        BATCH_SIZE = int(os.getenv("SPECDEC_BATCH_SIZE", "8"))
        logger.info(f"  Using batch size: {BATCH_SIZE}")

        # Heartbeat tracking
        last_heartbeat_time = time.time()

        for iteration in range(iterations):
            print(
                f"[ITER] ===== Starting Iteration {iteration+1}/{iterations} =====",
                flush=True,
            )
            logger.info(f"  Iteration {iteration+1}/{iterations}")
            iter_start_time = time.time()
            iter_results = []
            iter_samples = 0

            # Process prompts in batches
            num_batches = (len(PROMPT_SUITE) + BATCH_SIZE - 1) // BATCH_SIZE
            print(
                f"[ITER] Processing {num_batches} batches of up to {BATCH_SIZE} prompts each",
                flush=True,
            )

            for batch_idx, batch_start in enumerate(
                range(0, len(PROMPT_SUITE), BATCH_SIZE)
            ):
                batch_end = min(batch_start + BATCH_SIZE, len(PROMPT_SUITE))
                batch_prompts = PROMPT_SUITE[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))

                # Heartbeat check (every 60 seconds)
                current_time = time.time()
                if current_time - last_heartbeat_time >= 60.0:
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

                print(
                    f"[INFO] Iteration {iteration+1}/{iterations} | "
                    f"Batch {batch_idx+1}/{num_batches} (prompts {batch_start+1}-{batch_end}) | "
                    f"GPU util: {gpu_util_est:.1f}% | "
                    f"Elapsed: {time.time() - iter_start_time:.2f}s",
                    flush=True,
                )

                try:
                    # Use cached pipeline
                    pipeline = pipeline_cache[k]
                    if pipeline is None:
                        raise Exception("Pipeline creation failed for this K")

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

                    print(
                        f"[INFO] Batch {batch_idx+1} completed | "
                        f"Time: {batch_time_s:.2f}s ({batch_time_ms:.1f}ms) | "
                        f"GPU util: {gpu_util_est_after:.1f}% | "
                        f"Prompts: {len(batch_prompts)}",
                        flush=True,
                    )

                    # Process each result in the batch
                    print(
                        f"[BATCH] Processing {len(batch_results)} results from batch {batch_idx+1}",
                        flush=True,
                    )

                    for prompt_idx, result in zip(batch_indices, batch_results):
                        prompt = PROMPT_SUITE[prompt_idx]

                        # Extract metrics with validation
                        latency_ms = result.get("latency_ms", 0)
                        tokens_per_sec = result.get("tokens_per_sec", 0)
                        acceptance_rate = result.get("acceptance_rate", 0.0)
                        proposed = result.get("proposed", 0)
                        accepted = result.get("accepted", 0)
                        generated_tokens_count = result.get("generated_tokens", 0)
                        if isinstance(generated_tokens_count, list):
                            generated_tokens_count = len(generated_tokens_count)
                        text = result.get("text", "")
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

                        # Log progress for each prompt
                        print(
                            f"[PROMPT] {prompt_idx+1}/{len(PROMPT_SUITE)} | "
                            f"Tokens: {generated_tokens_count} | "
                            f"Accept: {accepted}/{proposed} ({acceptance_rate:.1%}) | "
                            f"Throughput: {tokens_per_sec:.2f} tok/s",
                            flush=True,
                        )

                        # Store detailed result (minimal logging during generation)
                        detailed_result = {
                            "k": k,
                            "iteration": iteration + 1,
                            "prompt_idx": prompt_idx + 1,
                            "prompt": prompt,
                            "latency_ms": latency_ms,
                            "tokens_per_sec": tokens_per_sec,
                            "acceptance_rate": acceptance_rate,
                            "proposed": proposed,
                            "accepted": accepted,
                            "kv_appended_tokens": kv_appended,
                            "kv_append_time_ms": kv_append_time,
                            "kv_append_enabled": kv_append_enabled,
                            "kv_append_backend": kv_append_backend,
                            "text": text[:100] + "..." if len(text) > 100 else text,
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
                            "prompt": PROMPT_SUITE[prompt_idx],
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
                    f"{iter_samples}/{len(PROMPT_SUITE)} samples | "
                    f"avg throughput={avg_throughput:.2f} tok/s | "
                    f"acceptance={avg_accept_rate:.2%} | "
                    f"avg latency={avg_latency:.1f}ms | "
                    f"elapsed={iter_elapsed:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[SUMMARY] Iteration {iteration+1}/{iterations} COMPLETE: "
                    f"0/{len(PROMPT_SUITE)} samples (all failed) | "
                    f"elapsed={iter_elapsed:.2f}s",
                    flush=True,
                )

            print(
                f"[ITER] ===== Iteration {iteration+1}/{iterations} finished =====",
                flush=True,
            )

        # Calculate statistics for this K
        valid_results = [r for r in k_results if "error" not in r]
        total_attempts = iterations * len(PROMPT_SUITE)

        if valid_results:
            latencies = [r["latency_ms"] for r in valid_results]
            throughputs = [r["tokens_per_sec"] for r in valid_results]
            acceptance_rates = [r["acceptance_rate"] for r in valid_results]
            proposed_counts = [r["proposed"] for r in valid_results]
            accepted_counts = [r["accepted"] for r in valid_results]
            kv_appended_counts = [r["kv_appended_tokens"] for r in valid_results]
            kv_append_times = [r["kv_append_time_ms"] for r in valid_results]

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
                    "acceptance_rate_mean": np.mean(acceptance_rates),
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

    return (
        results,
        detailed_results,
        {"kernel_info": kinfo, "deterministic": deterministic},
    )


def save_results(results, detailed_results, system_info, output_dir, device):
    """Save results to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get detailed metrics if available
    detailed_metrics = {}
    try:
        from src.metrics.detailed_profiler import get_summary

        detailed_metrics = get_summary()
    except ImportError:
        pass

    # Add detailed metrics to system info
    if detailed_metrics and detailed_metrics.get("enabled", False):
        system_info["detailed_metrics"] = detailed_metrics

    # Save summary results to CSV
    csv_file = output_dir / f"specdec_{device}_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save detailed results to JSON
    json_file = output_dir / f"specdec_{device}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "system_info": system_info,
                "summary_results": results,
                "detailed_results": detailed_results,
                "detailed_metrics": detailed_metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {csv_file} and {json_file}")
    return csv_file, json_file


def create_plots(results, output_dir):
    """Create performance plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Create figures directory
        figures_dir = Path(output_dir).parent / "docs" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        k_values = [r["k"] for r in results]
        throughput_means = [r["tokens_per_sec_mean"] for r in results]
        throughput_stds = [r["tokens_per_sec_std"] for r in results]
        acceptance_means = [r["acceptance_rate_mean"] for r in results]
        acceptance_stds = [r["acceptance_rate_std"] for r in results]

        # Plot 1: Tokens/sec vs K
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            k_values,
            throughput_means,
            yerr=throughput_stds,
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("K (Draft Tokens)", fontsize=12)
        plt.ylabel("Throughput (tokens/sec)", fontsize=12)
        plt.title("Speculative Decoding Performance: Throughput vs K", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)

        # Add value labels
        for i, (k, mean, std) in enumerate(
            zip(k_values, throughput_means, throughput_stds)
        ):
            plt.annotate(
                f"{mean:.2f}±{std:.2f}",
                (k, mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "throughput_vs_k.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 2: Acceptance Rate vs K
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            k_values,
            acceptance_means,
            yerr=acceptance_stds,
            marker="s",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
            color="orange",
        )
        plt.xlabel("K (Draft Tokens)", fontsize=12)
        plt.ylabel("Acceptance Rate", fontsize=12)
        plt.title("Speculative Decoding Performance: Acceptance Rate vs K", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)

        # Add value labels
        for i, (k, mean, std) in enumerate(
            zip(k_values, acceptance_means, acceptance_stds)
        ):
            plt.annotate(
                f"{mean:.3f}±{std:.3f}",
                (k, mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "acceptance_vs_k.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Plots saved to {figures_dir}")

    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
    except Exception as e:
        logger.error(f"Error creating plots: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive K-Sweep Testing for Speculative Decoding"
    )
    parser.add_argument("--base-model", default="gpt2", help="Base model name")
    parser.add_argument("--draft-model", default="distilgpt2", help="Draft model name")
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations per K"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to run on (auto selects best available)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--metrics-detailed",
        action="store_true",
        help="Enable detailed metrics collection",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture (CUDA only)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic mode "
            "(seeds, cudnn.deterministic, disable experimental draftors)"
        ),
    )

    args = parser.parse_args()

    # Set environment variables for detailed metrics
    if args.metrics_detailed:
        os.environ["SPECDEC_DETAILED_METRICS"] = "1"

    if args.cuda_graph:
        os.environ["SPECDEC_CUDA_GRAPH"] = "1"

    # Resolve device
    resolved_device = resolve_device(args.device)

    logger.info(
        f"Starting comprehensive K-sweep test: {args.base_model} + {args.draft_model}"
    )
    logger.info(f"Device: {resolved_device} (requested: {args.device})")
    logger.info(f"Max tokens: {args.max_tokens}, Iterations: {args.iterations}")
    logger.info(f"Prompt suite: {len(PROMPT_SUITE)} prompts")

    # Get system info (add kernel info + deterministic)
    system_info = get_system_info(resolved_device)
    system_info["deterministic"] = args.deterministic or (
        os.getenv("SPECDEC_DETERMINISTIC", "0").lower() in ("1", "true", "yes")
    )
    system_info["kernel_backends"] = get_kernel_info()

    # Run tests
    results, detailed_results, run_meta = run_comprehensive_k_sweep(
        base_model=args.base_model,
        draft_model=args.draft_model,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
        device=args.device,
        deterministic=args.deterministic,
    )

    # Save results
    # Merge run metadata into system info for JSON
    system_info.update(run_meta)

    csv_file, json_file = save_results(
        results, detailed_results, system_info, args.output_dir, resolved_device
    )

    # Create plots (unless disabled)
    if not args.no_plots:
        create_plots(results, args.output_dir)
    else:
        logger.info("Skipping plot generation (--no-plots specified)")

    # Print summary table with real values
    print("\n" + "=" * 120, flush=True)
    print("K-SWEEP RESULTS SUMMARY", flush=True)
    print("=" * 120, flush=True)
    print(
        f"Device: {resolved_device} | Dtype: {system_info['dtype']} | "
        f"Models: {args.base_model} + {args.draft_model}",
        flush=True,
    )
    print("=" * 120, flush=True)
    print(
        f"{'K':<3} {'Samples':<8} {'Failures':<9} {'Success%':<9} "
        f"{'Latency (ms)':<20} {'Throughput (tok/s)':<20} "
        f"{'Accept Rate':<15} {'Proposed':<12} {'Accepted':<12}",
        flush=True,
    )
    print("-" * 120, flush=True)

    for result in results:
        if result["n_samples"] > 0:
            # Validate values before printing
            latency_mean = result.get("latency_ms_mean", 0.0)
            latency_std = result.get("latency_ms_std", 0.0)
            throughput_mean = result.get("tokens_per_sec_mean", 0.0)
            throughput_std = result.get("tokens_per_sec_std", 0.0)
            accept_mean = result.get("acceptance_rate_mean", 0.0)
            accept_std = result.get("acceptance_rate_std", 0.0)
            proposed_mean = result.get("proposed_mean", 0.0)
            proposed_std = result.get("proposed_std", 0.0)
            accepted_mean = result.get("accepted_mean", 0.0)
            accepted_std = result.get("accepted_std", 0.0)

            # Check for NaN/inf
            if np.isnan(latency_mean) or np.isinf(latency_mean):
                latency_mean = 0.0
            if np.isnan(latency_std) or np.isinf(latency_std):
                latency_std = 0.0
            if np.isnan(throughput_mean) or np.isinf(throughput_mean):
                throughput_mean = 0.0
            if np.isnan(throughput_std) or np.isinf(throughput_std):
                throughput_std = 0.0

            print(
                f"{result['k']:<3} {result['n_samples']:<8} {result['n_failures']:<9} "
                f"{result['success_rate']*100:.1f}%{'':<4} "
                f"{latency_mean:.1f}±{latency_std:.1f}{'':<15} "
                f"{throughput_mean:.2f}±{throughput_std:.2f}{'':<12} "
                f"{accept_mean:.3f}±{accept_std:.3f}{'':<9} "
                f"{proposed_mean:.1f}±{proposed_std:.1f}{'':<7} "
                f"{accepted_mean:.1f}±{accepted_std:.1f}",
                flush=True,
            )
        else:
            print(
                f"{result['k']:<3} {result['n_samples']:<8} {result['n_failures']:<9} "
                f"{result['success_rate']*100:.1f}%{'':<4} "
                f"{'N/A':<20} {'N/A':<20} {'N/A':<15} {'N/A':<12} {'N/A':<12}",
                flush=True,
            )

    print("=" * 120, flush=True)
    print(
        f"\n[FINAL] Comprehensive K-sweep completed. Results saved to {csv_file} and {json_file}",
        flush=True,
    )
    logger.info(
        f"Comprehensive K-sweep completed. Results saved to {csv_file} and {json_file}"
    )
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
