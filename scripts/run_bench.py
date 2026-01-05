#!/usr/bin/env python3
"""
Canonical benchmark runner for speculative decoding experiments.

Usage:
    python scripts/run_bench.py --config configs/t4_gpt2_distil.yaml

This script runs baseline (draft_model=none) and speculative decoding (draft_model set)
experiments, always using batch mode with ring-buffer KV cache.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kernels import get_kernel_info  # noqa: E402
from specdec import SpeculativePipeline  # noqa: E402

# Import runner utilities
sys.path.insert(0, str(SCRIPT_DIR))
from k_sweep.utils import get_system_info, resolve_device, set_deterministic_mode  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default prompt suite
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


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_baseline(
    pipeline: SpeculativePipeline,
    prompts: List[str],
    max_tokens: int,
    iterations: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Run baseline (non-speculative) generation."""
    logger.info("Running baseline (non-speculative) generation...")
    results = []
    
    for i in range(iterations):
        iter_start = time.time()
        batch_results = pipeline.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
        )
        iter_time = time.time() - iter_start
        
        for j, result in enumerate(batch_results):
            results.append({
                "iteration": i,
                "prompt_idx": j,
                "prompt": prompts[j],
                "text": result.get("text", ""),
                "generated_tokens": result.get("generated_tokens", []),
                "num_generated": result.get("num_generated", 0),
                "latency_ms": result.get("latency_ms", iter_time * 1000),
                "tokens_per_sec": result.get("tokens_per_sec", 0.0),
                "throughput_tokens_per_sec": result.get("throughput_tokens_per_sec", 0.0),
                "mode": "baseline",
            })
    
    return results


def run_specdec(
    pipeline: SpeculativePipeline,
    prompts: List[str],
    max_tokens: int,
    iterations: int,
    k_values: List[int],
    device: str,
) -> List[Dict[str, Any]]:
    """Run speculative decoding with multiple K values."""
    logger.info(f"Running speculative decoding with K values: {k_values}...")
    results = []
    
    for k in k_values:
        logger.info(f"Running K={k}...")
        # Configure controller with desired K value
        # Import from the installed package (works in both local and Colab)
        try:
            from specdec.policies.controllers import FixedKController
        except ImportError:
            # Fallback for direct script execution
            import sys
            from pathlib import Path
            script_dir = Path(__file__).parent
            src_dir = script_dir.parent / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from specdec.policies.controllers import FixedKController
        pipeline.controller = FixedKController(k=k)
        
        for i in range(iterations):
            iter_start = time.time()
            batch_results = pipeline.generate_batch(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
            )
            iter_time = time.time() - iter_start
            
            for j, result in enumerate(batch_results):
                results.append({
                    "iteration": i,
                    "prompt_idx": j,
                    "prompt": prompts[j],
                    "k": k,
                    "text": result.get("text", ""),
                    "generated_tokens": result.get("generated_tokens", []),
                    "num_generated": result.get("num_generated", 0),
                    "latency_ms": result.get("latency_ms", iter_time * 1000),
                    "tokens_per_sec": result.get("tokens_per_sec", 0.0),
                    "throughput_tokens_per_sec": result.get("throughput_tokens_per_sec", 0.0),
                    "acceptance_rate": result.get("acceptance_rate", 0.0),
                    "proposed": result.get("proposed", 0),
                    "accepted": result.get("accepted", 0),
                    "draft_avg_ms": result.get("draft_avg_ms", 0.0),
                    "verify_avg_ms": result.get("verify_avg_ms", 0.0),
                    "kv_append_enabled": result.get("kv_append_enabled", False),
                    "kv_append_backend": result.get("kv_append_backend", "unknown"),
                    "verify_backend": result.get("verify_backend", "unknown"),
                    "draft_generation_mode": result.get("draft_generation_mode", "unknown"),
                    "kv_appended_tokens": result.get("kv_appended_tokens", 0),
                    "kv_append_time_ms": result.get("kv_append_time_ms", 0.0),
                    "mode": "specdec",
                })
    
    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results by mode and K value."""
    summary = {}
    
    # Separate baseline and specdec
    baseline_results = [r for r in results if r.get("mode") == "baseline"]
    specdec_results = [r for r in results if r.get("mode") == "specdec"]
    
    if baseline_results:
        throughputs = [r["tokens_per_sec"] for r in baseline_results if r.get("tokens_per_sec")]
        summary["baseline"] = {
            "n_samples": len(baseline_results),
            "tokens_per_sec_mean": float(np.mean(throughputs)) if throughputs else 0.0,
            "tokens_per_sec_std": float(np.std(throughputs)) if throughputs else 0.0,
            "latency_ms_mean": float(np.mean([r["latency_ms"] for r in baseline_results])),
            "latency_ms_std": float(np.std([r["latency_ms"] for r in baseline_results])),
        }
    
    # Aggregate by K
    k_groups = {}
    for r in specdec_results:
        k = r.get("k", 0)
        if k not in k_groups:
            k_groups[k] = []
        k_groups[k].append(r)
    
    summary["specdec"] = {}
    for k, k_results in sorted(k_groups.items()):
        throughputs = [r["tokens_per_sec"] for r in k_results if r.get("tokens_per_sec")]
        acceptance_rates = [r.get("acceptance_rate", 0.0) for r in k_results]
        draft_times = [r.get("draft_avg_ms", 0.0) for r in k_results]
        verify_times = [r.get("verify_avg_ms", 0.0) for r in k_results]
        kv_times = [r.get("kv_append_time_ms", 0.0) for r in k_results]
        
        summary["specdec"][k] = {
            "n_samples": len(k_results),
            "tokens_per_sec_mean": float(np.mean(throughputs)) if throughputs else 0.0,
            "tokens_per_sec_std": float(np.std(throughputs)) if throughputs else 0.0,
            "latency_ms_mean": float(np.mean([r["latency_ms"] for r in k_results])),
            "latency_ms_std": float(np.std([r["latency_ms"] for r in k_results])),
            "acceptance_rate_mean": float(np.mean(acceptance_rates)) if acceptance_rates else 0.0,
            "acceptance_rate_std": float(np.std(acceptance_rates)) if acceptance_rates else 0.0,
            "draft_avg_ms": float(np.mean(draft_times)) if draft_times else 0.0,
            "verify_avg_ms": float(np.mean(verify_times)) if verify_times else 0.0,
            "kv_append_time_ms": float(np.mean(kv_times)) if kv_times else 0.0,
        }
    
    return summary


def print_summary_table(summary: Dict[str, Any], device: str):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print(f"Device: {device}")
    print("-" * 100)
    
    # Baseline row
    if "baseline" in summary:
        bl = summary["baseline"]
        print(f"{'Mode':<12} {'K':<4} {'Throughput (tok/s)':<20} {'Latency (ms)':<20} {'Accept Rate':<15}")
        print("-" * 100)
        print(
            f"{'Baseline':<12} {'-':<4} "
            f"{bl['tokens_per_sec_mean']:.2f}±{bl['tokens_per_sec_std']:.2f}  "
            f"{bl['latency_ms_mean']:.1f}±{bl['latency_ms_std']:.1f}  "
            f"{'-':<15}"
        )
    
    # Specdec rows
    if "specdec" in summary:
        print("-" * 100)
        for k in sorted(summary["specdec"].keys()):
            sd = summary["specdec"][k]
            print(
                f"{'SpecDec':<12} {k:<4} "
                f"{sd['tokens_per_sec_mean']:.2f}±{sd['tokens_per_sec_std']:.2f}  "
                f"{sd['latency_ms_mean']:.1f}±{sd['latency_ms_std']:.1f}  "
                f"{sd['acceptance_rate_mean']:.3f}±{sd['acceptance_rate_std']:.3f}"
            )
    
    # Timing breakdown for specdec
    if "specdec" in summary:
        print("\n" + "-" * 100)
        print("Timing Breakdown (ms):")
        print(f"{'Mode':<12} {'K':<4} {'Draft':<12} {'Verify':<12} {'KV Append':<12}")
        print("-" * 100)
        for k in sorted(summary["specdec"].keys()):
            sd = summary["specdec"][k]
            print(
                f"{'SpecDec':<12} {k:<4} "
                f"{sd['draft_avg_ms']:.2f}  "
                f"{sd['verify_avg_ms']:.2f}  "
                f"{sd['kv_append_time_ms']:.2f}"
            )
    
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Canonical benchmark runner for speculative decoding"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/t4_gpt2_distil.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<run_id>)",
    )
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Extract config values
    base_model = config.get("base_model", "gpt2")
    draft_model = config.get("draft_model", None)  # None for baseline
    device = config.get("device", "auto")
    dtype = config.get("dtype", "auto")
    batch_size = config.get("batch_size", 1)
    max_tokens = config.get("max_tokens", 32)
    iterations = config.get("iterations", 10)
    k_values = config.get("k_values", config.get("max_k", [1, 2, 3, 4]))
    if isinstance(k_values, int):
        k_values = list(range(1, k_values + 1))
    kv_append_enabled = config.get("kv_append_enabled", True)
    draft_force_hf_generate = config.get("draft_force_hf_generate", False)
    deterministic = config.get("deterministic", False)
    
    # Set environment variables from config
    if dtype != "auto":
        os.environ["SPECDEC_DTYPE"] = dtype
    os.environ["SPECDEC_BATCH_SIZE"] = str(batch_size)
    os.environ["SPECDEC_ENABLE_KV_APPEND"] = "1" if kv_append_enabled else "0"
    if draft_force_hf_generate:
        os.environ["SPECDEC_DRAFT_FORCE_HF_GENERATE"] = "1"
    if deterministic:
        os.environ["SPECDEC_DETERMINISTIC"] = "1"
    
    # Resolve device
    resolved_device = resolve_device(device)
    logger.info(f"Using device: {resolved_device}")
    
    # Set deterministic mode
    set_deterministic_mode(deterministic)
    
    # Startup logging
    print("=" * 80, flush=True)
    print("[STARTUP] Benchmark Configuration", flush=True)
    print("=" * 80, flush=True)
    print(f"[STARTUP] Base Model: {base_model}", flush=True)
    print(f"[STARTUP] Draft Model: {draft_model if draft_model else 'None (baseline)'}", flush=True)
    print(f"[STARTUP] Device: {resolved_device}", flush=True)
    print(f"[STARTUP] Batch Size: {batch_size}", flush=True)
    print(f"[STARTUP] Max Tokens: {max_tokens}", flush=True)
    print(f"[STARTUP] Iterations: {iterations}", flush=True)
    print(f"[STARTUP] K Values: {k_values}", flush=True)
    print(f"[STARTUP] KV Append Enabled: {kv_append_enabled}", flush=True)
    print(f"[STARTUP] Draft Force HF Generate: {draft_force_hf_generate}", flush=True)
    print(f"[STARTUP] Deterministic: {deterministic}", flush=True)
    print(f"[STARTUP] Generation Mode: batch (ring-buffer KV)", flush=True)
    print("=" * 80, flush=True)
    
    # Get prompts
    prompts = config.get("prompts", PROMPT_SUITE)
    if isinstance(prompts, str):
        prompts = [prompts]
    # Limit to batch_size if needed
    prompts = prompts[:batch_size]
    logger.info(f"Using {len(prompts)} prompts")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create pipeline
    logger.info("Creating pipeline...")
    # For baseline, pass None or empty string for draft_model
    draft_model_arg = draft_model if draft_model else None
    pipeline = SpeculativePipeline(
        base_model=base_model,
        draft_model=draft_model_arg,
        device=resolved_device,
        implementation="hf",  # Use HF implementation
    )
    
    all_results = []
    
    # Run baseline if draft_model is None or if config explicitly requests it
    run_baseline_flag = config.get("run_baseline", draft_model is None)
    if run_baseline_flag:
        # Create baseline pipeline (no draft model)
        baseline_pipeline = SpeculativePipeline(
            base_model=base_model,
            draft_model=None,
            device=resolved_device,
            implementation="hf",
        )
        baseline_results = run_baseline(baseline_pipeline, prompts, max_tokens, iterations, resolved_device)
        all_results.extend(baseline_results)
    
    # Run specdec if draft_model is set
    if draft_model:
        specdec_results = run_specdec(pipeline, prompts, max_tokens, iterations, k_values, resolved_device)
        all_results.extend(specdec_results)
    
    # Aggregate results
    summary = aggregate_results(all_results)
    
    # Get system info
    system_info = get_system_info(resolved_device)
    system_info.update({
        "config_file": str(config_path),
        "base_model": base_model,
        "draft_model": draft_model,
        "device": resolved_device,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "iterations": iterations,
        "k_values": k_values,
        "kv_append_enabled": kv_append_enabled,
        "draft_force_hf_generate": draft_force_hf_generate,
        "deterministic": deterministic,
        "mode": "batch",
    })
    
    # Save results
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "summary": summary,
            "detailed_results": all_results,
            "system_info": system_info,
        }, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    
    # Save CSV
    import csv
    csv_path = output_dir / "summary.csv"
    if all_results:
        # Collect all possible fieldnames from all results
        all_fieldnames = set()
        for result in all_results:
            all_fieldnames.update(result.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"CSV saved to {csv_path}")
    
    # Save system info
    system_path = output_dir / "system.json"
    with open(system_path, "w") as f:
        json.dump(system_info, f, indent=2)
    logger.info(f"System info saved to {system_path}")
    
    # Print summary table
    print_summary_table(summary, resolved_device)
    
    logger.info("Benchmark completed successfully!")


if __name__ == "__main__":
    main()

