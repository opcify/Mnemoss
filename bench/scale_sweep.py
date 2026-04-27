"""Scale-axis launch benchmark driver.

Runs ``bench.launch_comparison`` in scale mode for each (backend, N)
pair in the sweep grid and collates results into a single JSON.

Usage::

    python -m bench.scale_sweep \
        --embedder local \
        --sizes 500 1500 3000 5000 \
        --backends mnemoss raw_stack static_file \
        --out bench/results/scale_sweep_local.json

Each individual point is written under ``bench/results/scale_{backend}_n{N}.json``
so the raw per-run JSON is inspectable and re-runs are cheap to restart.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run_one(
    *,
    backend: str,
    scale_n: int,
    embedder: str,
    gold_conversation: str,
    out_path: Path,
) -> dict:
    """Shell out to ``bench.launch_comparison`` and read back the JSON.

    Using subprocess (rather than importing ``run_scale`` in-process)
    keeps each run in a clean Python process — important because the
    sentence-transformers model weights stick around per-process and
    a crashed run doesn't poison the next one.
    """

    cmd = [
        sys.executable,
        "-m",
        "bench.launch_comparison",
        "--backend",
        backend,
        "--embedder",
        embedder,
        "--scale-n",
        str(scale_n),
        "--gold-conversation",
        gold_conversation,
        "--out",
        str(out_path),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    wall = time.perf_counter() - t0
    payload = json.loads(out_path.read_text())
    payload["_wall_seconds"] = round(wall, 2)
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "nomic", "fake"],
        default="local",
        help="Embedder to use (default: local = LocalEmbedder / free).",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500, 1500, 3000, 5000],
        help="Corpus sizes to sweep (default: 500 1500 3000 5000).",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["mnemoss", "raw_stack", "static_file"],
        choices=[
            "mnemoss",
            "mnemoss_semantic",
            "mnemoss_fast",
            "mnemoss_prod",
            "mnemoss_rocket",
            "raw_stack",
            "static_file",
        ],
        help="Backends to run at each size.",
    )
    p.add_argument(
        "--gold-conversation",
        default="conv-26",
        help="Gold conversation ID (queries come from here).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("bench/results"),
        help="Directory for per-run JSON files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the collated sweep JSON.",
    )
    args = p.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for backend in args.backends:
        for n in args.sizes:
            # static_file ignores the embedder — tag one shared file name.
            tag = backend if backend != "static_file" else "static_file"
            single_out = args.results_dir / f"scale_{tag}_{args.embedder}_n{n}.json"
            print(f"[sweep] backend={backend} embedder={args.embedder} N={n}", flush=True)
            payload = _run_one(
                backend=backend,
                scale_n=n,
                embedder=args.embedder,
                gold_conversation=args.gold_conversation,
                out_path=single_out,
            )
            results.append(
                {
                    "backend": backend,
                    "scale_n": n,
                    "embedder": args.embedder,
                    "recall_at_k": payload["aggregate"]["mean_recall_at_k"],
                    "n_queries": payload["aggregate"]["n_queries"],
                    "duration_seconds": payload["duration_seconds"],
                    "wall_seconds": payload["_wall_seconds"],
                    "single_out": str(single_out),
                }
            )
            print(
                f"  → recall@10 = {payload['aggregate']['mean_recall_at_k']:.4f} "
                f"(duration={payload['duration_seconds']:.1f}s)",
                flush=True,
            )

    summary = {
        "chart": "scale",
        "embedder": args.embedder,
        "sizes": args.sizes,
        "backends": args.backends,
        "gold_conversation": args.gold_conversation,
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    # Terminal summary table.
    print()
    print(f"{'backend':<16} {'N':>6} {'recall@10':>10} {'wall':>8}")
    print("─" * 44)
    for r in results:
        print(
            f"{r['backend']:<16} {r['scale_n']:>6} "
            f"{r['recall_at_k']:>10.4f} {r['wall_seconds']:>8.1f}s"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
