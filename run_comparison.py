from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "qwen3-jax-vs-pytorch-benchmark")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both Qwen3 synthetic benchmarks and create a W&B report."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument(
        "--group", default=f"synthetic-qwen3-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--torchrun-gpus", type=positive_int, default=8)

    parser.add_argument("--vocab-size", type=positive_int, default=32768)
    parser.add_argument("--seq-len", type=positive_int, default=512)
    parser.add_argument("--per-device-batch-size", type=positive_int, default=2)
    parser.add_argument("--hidden-size", type=positive_int, default=768)
    parser.add_argument("--intermediate-size", type=positive_int, default=2304)
    parser.add_argument("--num-layers", type=positive_int, default=12)
    parser.add_argument("--num-attention-heads", type=positive_int, default=12)
    parser.add_argument("--num-key-value-heads", type=positive_int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def benchmark_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--project",
        args.project,
        "--group",
        args.group,
        "--wandb-mode",
        args.wandb_mode,
        "--output-dir",
        args.output_dir,
        "--vocab-size",
        str(args.vocab_size),
        "--seq-len",
        str(args.seq_len),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--hidden-size",
        str(args.hidden_size),
        "--intermediate-size",
        str(args.intermediate_size),
        "--num-layers",
        str(args.num_layers),
        "--num-attention-heads",
        str(args.num_attention_heads),
        "--num-key-value-heads",
        str(args.num_key_value_heads),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--seed",
        str(args.seed),
    ]
    if args.entity:
        result.extend(["--entity", args.entity])
    return result


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    benchmark = here / "benchmark.py"
    report = here / "create_wandb_report.py"
    common = benchmark_args(args)

    run(
        [
            sys.executable,
            str(benchmark),
            "--framework",
            "jax",
            "--run-name",
            f"jax-{args.group}",
            *common,
        ]
    )
    run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(args.torchrun_gpus),
            str(benchmark),
            "--framework",
            "torch",
            "--run-name",
            f"torch-{args.group}",
            *common,
        ]
    )
    if not args.skip_report and args.wandb_mode == "online":
        run(
            [
                sys.executable,
                str(report),
                "--entity",
                args.entity,
                "--project",
                args.project,
                "--group",
                args.group,
                "--output-dir",
                args.output_dir,
            ]
        )

    print(
        json.dumps({"group": args.group, "project": args.project, "entity": args.entity}, indent=2)
    )


if __name__ == "__main__":
    main()
