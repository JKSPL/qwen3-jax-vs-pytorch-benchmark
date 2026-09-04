from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_BENCHMARK_ARGS = [
    "--framework",
    "jax",
    "--dtype",
    "bfloat16",
    "--gradient-communication-dtype",
    "bfloat16",
    "--jax-attention-implementation",
    "cudnn_flash",
    "--run-name",
    "jax-process-per-gpu-bf16-cudnn-flash",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch the JAX benchmark with one Python process per local GPU.",
        allow_abbrev=False,
    )
    parser.add_argument("--num-processes", type=positive_int, default=8)
    parser.add_argument("--coordinator-address", default=None)
    parser.add_argument("--coordinator-bind-address", default=None)
    parser.add_argument("--initialization-timeout", type=positive_int, default=300)
    return parser.parse_known_args()


def child_env(process_id: int, num_processes: int) -> dict[str, str]:
    env = os.environ.copy()
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        device_ids = [value.strip() for value in visible_devices.split(",") if value.strip()]
        if len(device_ids) < num_processes:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES exposes {len(device_ids)} devices, "
                f"but --num-processes is {num_processes}"
            )
        selected_device = device_ids[process_id]
    else:
        selected_device = str(process_id)
    env.update(
        {
            "RANK": str(process_id),
            "LOCAL_RANK": "0",
            "WORLD_SIZE": str(num_processes),
            "JAX_PROCESS_ID": str(process_id),
            "JAX_NUM_PROCESSES": str(num_processes),
            "CUDA_VISIBLE_DEVICES": selected_device,
        }
    )
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    return env


def main() -> None:
    args, benchmark_args = parse_args()
    here = Path(__file__).resolve().parent
    benchmark = here / "benchmark.py"
    coordinator_address = args.coordinator_address or f"127.0.0.1:{find_free_port()}"

    processes: list[subprocess.Popen[bytes]] = []
    try:
        for process_id in range(args.num_processes):
            cmd = [
                sys.executable,
                str(benchmark),
                *DEFAULT_BENCHMARK_ARGS,
                *benchmark_args,
                "--jax-distributed",
                "--jax-coordinator-address",
                coordinator_address,
                "--jax-num-processes",
                str(args.num_processes),
                "--jax-process-id",
                str(process_id),
                "--jax-local-device-ids",
                "0",
                "--jax-initialization-timeout",
                str(args.initialization_timeout),
            ]
            if args.coordinator_bind_address:
                cmd.extend(["--jax-coordinator-bind-address", args.coordinator_bind_address])
            print(f"+ rank {process_id}: {' '.join(cmd)}", flush=True)
            processes.append(subprocess.Popen(cmd, env=child_env(process_id, args.num_processes)))

        exit_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise

    failed = [index for index, exit_code in enumerate(exit_codes) if exit_code != 0]
    if failed:
        raise SystemExit(
            f"JAX process-per-GPU benchmark failed for ranks {failed} with exit codes {exit_codes}"
        )


if __name__ == "__main__":
    main()
