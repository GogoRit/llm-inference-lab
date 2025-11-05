import argparse
import os
import sys
from pathlib import Path

from kernels import get_kernel_info
from specdec import SpeculativePipeline


def cmd_bench(args: argparse.Namespace) -> int:
    # Reuse the existing script logic by importing functions
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from comprehensive_k_sweep import (  # type: ignore
        get_system_info,
        resolve_device,
        run_comprehensive_k_sweep,
        save_results,
    )

    resolved_device = resolve_device(args.device)
    system_info = get_system_info(resolved_device)
    system_info["deterministic"] = args.deterministic or (
        os.getenv("SPECDEC_DETERMINISTIC", "0").lower() in ("1", "true", "yes")
    )
    system_info["kernel_backends"] = get_kernel_info()

    results, detailed, run_meta = run_comprehensive_k_sweep(
        base_model=args.base_model,
        draft_model=args.draft_model,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
        device=args.device,
        deterministic=args.deterministic,
    )

    system_info.update(run_meta)
    out_dir = args.output_dir
    save_results(results, detailed, system_info, out_dir, resolved_device)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    pipe = SpeculativePipeline(
        base_model=args.base_model,
        draft_model=args.draft_model,
        max_draft=args.k,
        implementation="hf",
        device=args.device,
        enable_optimization=True,
        draft_mode="vanilla",
    )
    res = pipe.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    kinfo = get_kernel_info()
    verify_backend = kinfo.get("verify_backend")
    kv_backend = kinfo.get("kv_append_backend")
    print(
        f"Device: {res.get('device')} | Dtype: {res.get('dtype')} | "
        f"Backends: verify={verify_backend}, kv_append={kv_backend}"
    )
    print(f"Text: {res.get('text', '')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="specdec", description="LLM Inference Lab CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bench", help="Run a K-sweep benchmark")
    pb.add_argument("--base-model", default="gpt2")
    pb.add_argument("--draft-model", default="distilgpt2")
    pb.add_argument("--max-tokens", type=int, default=32)
    pb.add_argument("--iterations", type=int, default=10)
    pb.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    pb.add_argument("--output-dir", type=Path, default=Path("results"))
    pb.add_argument("--deterministic", action="store_true")
    pb.set_defaults(func=cmd_bench)

    pr = sub.add_parser("run", help="Run a single prompt via SpeculativePipeline")
    pr.add_argument("--base-model", default="gpt2")
    pr.add_argument("--draft-model", default="distilgpt2")
    pr.add_argument("--k", type=int, default=2)
    pr.add_argument("--max-tokens", type=int, default=32)
    pr.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    pr.add_argument("--temperature", type=float, default=0.7)
    pr.add_argument("--do-sample", action="store_true")
    pr.add_argument("prompt", type=str)
    pr.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)
