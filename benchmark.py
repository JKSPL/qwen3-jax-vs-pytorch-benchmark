from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "qwen3-jax-vs-pytorch-benchmark")
JAX_PERFORMANCE_XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_use_memcpy_local_p2p=true",
    "--xla_gpu_all_reduce_combine_threshold_bytes=67108864",
)
JAX_PERFORMANCE_CUDA_ENV = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
}
JAX_PERFORMANCE_NCCL_ENV = {}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Qwen3-style training benchmark for JAX and PyTorch."
    )
    parser.add_argument("--framework", choices=("jax", "torch"), required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--output-dir", default="results")

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
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--gradient-communication-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--cuda-profiler-capture", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--torch-attn-implementation", default="sdpa")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-fused-adam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--torch-matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    parser.add_argument(
        "--jax-attention-implementation", choices=("explicit", "cudnn_flash"), default="explicit"
    )
    parser.add_argument(
        "--jax-performance-flags", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--jax-matmul-precision", default="high")
    parser.add_argument("--jax-preallocate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jax-distributed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jax-coordinator-address", default=None)
    parser.add_argument("--jax-coordinator-bind-address", default=None)
    parser.add_argument("--jax-num-processes", type=positive_int, default=None)
    parser.add_argument("--jax-process-id", type=int, default=None)
    parser.add_argument("--jax-local-device-ids", default=None)
    parser.add_argument("--jax-initialization-timeout", type=positive_int, default=300)
    args = parser.parse_args()

    if args.hidden_size % args.num_attention_heads != 0:
        parser.error("--hidden-size must be divisible by --num-attention-heads")
    if args.num_attention_heads % args.num_key_value_heads != 0:
        parser.error("--num-attention-heads must be divisible by --num-key-value-heads")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    if args.group is None:
        args.group = f"synthetic-qwen3-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    return args


def model_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vocab_size": args.vocab_size,
        "seq_len": args.seq_len,
        "per_device_batch_size": args.per_device_batch_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_layers": args.num_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.hidden_size // args.num_attention_heads,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_communication_dtype": args.gradient_communication_dtype,
    }


def parse_device_ids(value: str | None) -> int | list[int] | None:
    if value is None or value == "":
        return None
    device_ids = [int(part) for part in value.split(",")]
    return device_ids[0] if len(device_ids) == 1 else device_ids


def cuda_profiler_control(action: str) -> None:
    import ctypes
    import ctypes.util

    function_name = f"cudaProfiler{action}"
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
    ]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            cudart = ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        profiler_function = getattr(cudart, function_name, None)
        if profiler_function is None:
            errors.append(f"{candidate}: missing {function_name}")
            continue
        profiler_function.restype = ctypes.c_int
        status = int(profiler_function())
        if status != 0:
            raise RuntimeError(f"{function_name} returned CUDA status {status}")
        return
    raise RuntimeError(f"could not load {function_name} from CUDA runtime: {'; '.join(errors)}")


def apply_jax_performance_environment(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    existing_xla_flags = os.environ.get("XLA_FLAGS", "").split()
    existing_flag_names = {flag.split("=", 1)[0] for flag in existing_xla_flags}
    added_xla_flags = [
        flag
        for flag in JAX_PERFORMANCE_XLA_FLAGS
        if flag.split("=", 1)[0] not in existing_flag_names
    ]
    if added_xla_flags:
        os.environ["XLA_FLAGS"] = " ".join([*existing_xla_flags, *added_xla_flags]).strip()

    added_cuda_env: dict[str, str] = {}
    for key, value in JAX_PERFORMANCE_CUDA_ENV.items():
        if key not in os.environ:
            os.environ[key] = value
            added_cuda_env[key] = value

    added_nccl_env: dict[str, str] = {}
    for key, value in JAX_PERFORMANCE_NCCL_ENV.items():
        if key not in os.environ:
            os.environ[key] = value
            added_nccl_env[key] = value

    return {
        "enabled": True,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "added_xla_flags": added_xla_flags,
        "cuda_env": {key: os.environ.get(key) for key in JAX_PERFORMANCE_CUDA_ENV},
        "added_cuda_env": added_cuda_env,
        "nccl_env": {key: os.environ.get(key) for key in JAX_PERFORMANCE_NCCL_ENV},
        "added_nccl_env": added_nccl_env,
    }


def init_wandb(
    args: argparse.Namespace, framework: str, rank: int, extra_config: dict[str, Any] | None = None
):
    if args.wandb_mode == "disabled" or rank != 0:
        return None

    import wandb

    config = {
        "framework": framework,
        "model": model_config_dict(args),
    }
    if extra_config:
        config.update(extra_config)

    return wandb.init(
        project=args.project,
        entity=args.entity or None,
        group=args.group,
        name=args.run_name or f"{framework}-{args.group}",
        mode=args.wandb_mode,
        job_type=framework,
        tags=["synthetic-data", "qwen3", framework],
        config=config,
        settings=wandb.Settings(
            disable_git=True,
            save_code=False,
            x_disable_machine_info=True,
        ),
    )


def log_wandb(run, payload: dict[str, Any]) -> None:
    if run is None:
        return
    import wandb

    wandb.log(payload)
    for key, value in payload.items():
        run.summary[key] = value


def write_result(args: argparse.Namespace, result: dict[str, Any]) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"{result['framework']}_{timestamp}.json"
    latest = output_dir / f"latest_{result['framework']}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    latest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def run_torch(args: argparse.Namespace) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    from torch.nn.parallel import DistributedDataParallel
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(args.torch_matmul_precision)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    torch_compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch_autocast_enabled = args.dtype == "bfloat16" and device.type in {"cuda", "cpu"}

    config = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        head_dim=args.hidden_size // args.num_attention_heads,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.seq_len,
        use_cache=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        attn_implementation=args.torch_attn_implementation,
    )
    model = Qwen3ForCausalLM(config).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_dtype = str(next(model.parameters()).dtype).replace("torch.", "")

    if args.torch_compile:
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        if args.gradient_communication_dtype == "bfloat16":
            model.register_comm_hook(dist.group.WORLD, default_hooks.bf16_compress_hook)

    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = args.torch_fused_adam
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    input_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.per_device_batch_size, args.seq_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()

    run = init_wandb(
        args,
        framework="torch",
        rank=rank,
        extra_config={
            "torch": {
                "version": torch.__version__,
                "distributed": distributed,
                "world_size": world_size,
                "attn_implementation": args.torch_attn_implementation,
                "compile": args.torch_compile,
                "fused_adam": args.torch_fused_adam,
                "matmul_precision": args.torch_matmul_precision,
                "compute_dtype": args.dtype,
                "gradient_communication_dtype": args.gradient_communication_dtype,
                "autocast_enabled": torch_autocast_enabled,
                "parameter_dtype": parameter_dtype,
            }
        },
    )

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch_compute_dtype, enabled=torch_autocast_enabled
        ):
            output = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        optimizer.step()
        return loss.detach()

    compile_start = time.perf_counter()
    loss = None
    for _ in range(args.warmup_steps):
        loss = train_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if distributed:
        dist.barrier()
    compile_and_warmup_seconds = time.perf_counter() - compile_start

    if args.cuda_profiler_capture:
        if distributed:
            dist.barrier()
        cuda_profiler_control("Start")
        if distributed:
            dist.barrier()

    timed_start = time.perf_counter()
    for _ in range(args.timed_steps):
        loss = train_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if args.cuda_profiler_capture:
        cuda_profiler_control("Stop")
    if distributed:
        dist.barrier()
    elapsed_seconds = time.perf_counter() - timed_start

    if loss is None:
        loss = torch.tensor(float("nan"), device=device)
    mean_loss = loss.float()
    if distributed:
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)

    tokens = args.per_device_batch_size * args.seq_len * world_size * args.timed_steps
    tokens_per_second = tokens / elapsed_seconds
    result = {
        "framework": "torch",
        "group": args.group,
        "run_name": args.run_name or f"torch-{args.group}",
        "world_size": world_size,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameter_count": param_count,
        "tokens": tokens,
        "tokens_per_step": args.per_device_batch_size * args.seq_len * world_size,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": elapsed_seconds / args.timed_steps,
        "tokens_per_second": tokens_per_second,
        "loss": float(mean_loss.item()),
        "compile_and_warmup_seconds": compile_and_warmup_seconds,
        "config": model_config_dict(args),
        "framework_config": {
            "torch_version": torch.__version__,
            "attn_implementation": args.torch_attn_implementation,
            "compile": args.torch_compile,
            "fused_adam": args.torch_fused_adam,
            "matmul_precision": args.torch_matmul_precision,
            "compute_dtype": args.dtype,
            "gradient_communication_dtype": args.gradient_communication_dtype,
            "autocast_enabled": torch_autocast_enabled,
            "parameter_dtype": parameter_dtype,
            "cuda_profiler_capture": args.cuda_profiler_capture,
        },
    }

    if rank == 0:
        log_wandb(run, result)
        if run is not None:
            result["wandb_url"] = run.url
            run.finish()
        result_path = write_result(args, result)
        print(json.dumps({"result_path": str(result_path), **result}, indent=2, sort_keys=True))

    if distributed:
        dist.destroy_process_group()
    return result if rank == 0 else None


def run_jax(args: argparse.Namespace) -> dict[str, Any] | None:
    performance_environment = apply_jax_performance_environment(args.jax_performance_flags)
    if not args.jax_preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from jax import random
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from qwen3_jax import Qwen3ForCausalLM

    if args.jax_distributed:
        jax.distributed.initialize(
            coordinator_address=args.jax_coordinator_address,
            num_processes=args.jax_num_processes,
            process_id=args.jax_process_id,
            local_device_ids=parse_device_ids(args.jax_local_device_ids),
            initialization_timeout=args.jax_initialization_timeout,
            coordinator_bind_address=args.jax_coordinator_bind_address,
        )

    jax.config.update("jax_default_matmul_precision", args.jax_matmul_precision)
    jax_compute_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else None
    jax_gradient_communication_dtype = (
        jnp.bfloat16 if args.gradient_communication_dtype == "bfloat16" else None
    )

    def is_inexact_array(value: Any) -> bool:
        return hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact)

    devices = np.array(jax.local_devices())
    if devices.size == 0:
        raise RuntimeError("JAX did not report any local devices")
    mesh = Mesh(devices, ("data",))
    local_device_count = int(devices.size)
    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    world_size = int(jax.device_count())

    key = random.PRNGKey(args.seed + process_index)
    model = Qwen3ForCausalLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.seq_len,
        attention_implementation=args.jax_attention_implementation,
        dtype=jax_compute_dtype,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, state = nnx.split(model)
    param_count = sum(int(leaf.size) for leaf in jax.tree.leaves(state) if hasattr(leaf, "size"))

    optimizer = optax.adamw(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mu_dtype=jnp.float32,
    )
    opt_state = optimizer.init(state)

    def shard_for_pmap(tree):
        def shard_leaf(leaf):
            replicated = jnp.broadcast_to(leaf, (local_device_count,) + leaf.shape)
            return jax.device_put(replicated, NamedSharding(mesh, P("data", *([None] * leaf.ndim))))

        return jax.tree.map(shard_leaf, tree)

    state = shard_for_pmap(state)
    opt_state = shard_for_pmap(opt_state)

    key, data_key = random.split(key)
    input_ids = random.randint(
        data_key,
        shape=(local_device_count, args.per_device_batch_size, args.seq_len),
        minval=0,
        maxval=args.vocab_size,
        dtype=jnp.int32,
    )
    position_ids = jnp.broadcast_to(
        jnp.arange(args.seq_len, dtype=jnp.int32),
        (local_device_count, args.per_device_batch_size, args.seq_len),
    )
    batch = {
        "input_ids": jax.device_put(input_ids, NamedSharding(mesh, P("data", None, None))),
        "labels": jax.device_put(input_ids, NamedSharding(mesh, P("data", None, None))),
        "position_ids": jax.device_put(position_ids, NamedSharding(mesh, P("data", None, None))),
    }

    def loss_fn(step_state, step_batch):
        step_model = nnx.merge(graphdef, step_state)
        return step_model(step_batch).loss

    def train_step(step_state, step_opt_state, step_batch):
        loss, grads = jax.value_and_grad(loss_fn)(step_state, step_batch)
        local_grads = grads
        if jax_gradient_communication_dtype is not None:
            grads = jax.tree.map(
                lambda grad: (
                    grad.astype(jax_gradient_communication_dtype)
                    if is_inexact_array(grad)
                    else grad
                ),
                grads,
            )
        grads = jax.lax.pmean(grads, axis_name="data")
        if jax_gradient_communication_dtype is not None:
            grads = jax.tree.map(
                lambda grad, local_grad: (
                    grad.astype(local_grad.dtype) if is_inexact_array(grad) else grad
                ),
                grads,
                local_grads,
            )
        loss = jax.lax.pmean(loss, axis_name="data")
        updates, step_opt_state = optimizer.update(grads, step_opt_state, step_state)
        step_state = optax.apply_updates(step_state, updates)
        return step_state, step_opt_state, loss

    p_train_step = jax.pmap(train_step, axis_name="data")

    jax_run_config = {
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
        "backend": jax.default_backend(),
        "world_size": world_size,
        "process_count": process_count,
        "process_index": process_index,
        "local_device_count": local_device_count,
        "distributed": args.jax_distributed,
        "attention_implementation": args.jax_attention_implementation,
        "matmul_precision": args.jax_matmul_precision,
        "preallocate": args.jax_preallocate,
        "compute_dtype": args.dtype,
        "gradient_communication_dtype": args.gradient_communication_dtype,
        "parameter_dtype": "float32",
        "performance_environment": performance_environment,
        "cuda_profiler_capture": args.cuda_profiler_capture,
    }

    run = init_wandb(
        args,
        framework="jax",
        rank=process_index,
        extra_config={"jax": jax_run_config},
    )

    compile_start = time.perf_counter()
    loss = None
    for _ in range(args.warmup_steps):
        state, opt_state, loss = p_train_step(state, opt_state, batch)
        jax.block_until_ready(loss)
    if args.jax_distributed:
        multihost_utils.sync_global_devices("qwen3-benchmark-after-warmup")
    compile_and_warmup_seconds = time.perf_counter() - compile_start

    if args.cuda_profiler_capture:
        if args.jax_distributed:
            multihost_utils.sync_global_devices("qwen3-benchmark-before-profiler-start")
        cuda_profiler_control("Start")
        if args.jax_distributed:
            multihost_utils.sync_global_devices("qwen3-benchmark-after-profiler-start")

    if args.jax_distributed:
        multihost_utils.sync_global_devices("qwen3-benchmark-before-timed")
    timed_start = time.perf_counter()
    for _ in range(args.timed_steps):
        state, opt_state, loss = p_train_step(state, opt_state, batch)
    if loss is not None:
        jax.block_until_ready(loss)
    if args.cuda_profiler_capture:
        cuda_profiler_control("Stop")
    if args.jax_distributed:
        multihost_utils.sync_global_devices("qwen3-benchmark-after-timed")
    elapsed_seconds = time.perf_counter() - timed_start

    loss_value = float(jax.device_get(loss)[0]) if loss is not None else math.nan
    tokens = args.per_device_batch_size * args.seq_len * world_size * args.timed_steps
    tokens_per_second = tokens / elapsed_seconds
    result = {
        "framework": "jax",
        "group": args.group,
        "run_name": args.run_name or f"jax-{args.group}",
        "world_size": world_size,
        "device_count": world_size,
        "device_name": str(devices[0]),
        "process_count": process_count,
        "process_index": process_index,
        "local_device_count": local_device_count,
        "parameter_count": param_count,
        "tokens": tokens,
        "tokens_per_step": args.per_device_batch_size * args.seq_len * world_size,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": elapsed_seconds / args.timed_steps,
        "tokens_per_second": tokens_per_second,
        "loss": loss_value,
        "compile_and_warmup_seconds": compile_and_warmup_seconds,
        "config": model_config_dict(args),
        "framework_config": jax_run_config,
    }
    if process_index == 0:
        log_wandb(run, result)
        if run is not None:
            result["wandb_url"] = run.url
            run.finish()
        result_path = write_result(args, result)
        print(json.dumps({"result_path": str(result_path), **result}, indent=2, sort_keys=True))
    if args.jax_distributed:
        multihost_utils.sync_global_devices("qwen3-benchmark-finished")
    return result if process_index == 0 else None


def main() -> None:
    args = parse_args()
    if args.framework == "torch":
        run_torch(args)
    else:
        run_jax(args)


if __name__ == "__main__":
    main()
