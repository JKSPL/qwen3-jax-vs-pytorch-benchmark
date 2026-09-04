# Qwen3-style JAX vs PyTorch synthetic benchmark

This repository compares steady-state training throughput for equivalent small
Qwen3-style causal language models implemented with Flax NNX/JAX and Hugging
Face Transformers/PyTorch. It uses deterministic synthetic token batches, so no
dataset or model checkpoint is required.

The measured interval includes forward, backward, gradient reduction, and an
AdamW update. Throughput is computed as:

```text
per-device batch × sequence length × device count ÷ seconds per step
```

## Privacy defaults

The repository contains no private URLs, user names, e-mail addresses, host
names, GPU UUIDs, credentials, or internal package imports. W&B is disabled for
the smoke commands below. When W&B is enabled, Git capture, source-code capture,
and machine metadata are disabled by default. Set `WANDB_ENTITY` explicitly if
you want to publish runs to your own account.

## Setup

Use Python 3.12 and `uv`:

```bash
uv sync
```

Install the JAX CUDA 13 plugin that matches your platform separately, following
the official JAX installation instructions. The historical environment used
JAX/JAXlib 0.10.2 and PyTorch 2.12.1 with CUDA 13.

## CPU smoke tests

The framework paths can be checked independently with a tiny model:

```bash
uv run benchmark.py \
  --framework jax \
  --wandb-mode disabled \
  --no-jax-performance-flags \
  --hidden-size 64 \
  --intermediate-size 192 \
  --num-layers 1 \
  --num-attention-heads 4 \
  --num-key-value-heads 2 \
  --vocab-size 256 \
  --seq-len 16 \
  --per-device-batch-size 1 \
  --warmup-steps 1 \
  --timed-steps 2

uv run benchmark.py \
  --framework torch \
  --wandb-mode disabled \
  --hidden-size 64 \
  --intermediate-size 192 \
  --num-layers 1 \
  --num-attention-heads 4 \
  --num-key-value-heads 2 \
  --vocab-size 256 \
  --seq-len 16 \
  --per-device-batch-size 1 \
  --warmup-steps 1 \
  --timed-steps 2
```

## Eight-GPU comparison

`run_comparison.py` launches one JAX process controlling the visible devices,
then an eight-process PyTorch DDP run:

```bash
uv run run_comparison.py \
  --wandb-mode disabled \
  --hidden-size 768 \
  --intermediate-size 2304 \
  --num-layers 12 \
  --num-attention-heads 12 \
  --num-key-value-heads 6 \
  --vocab-size 32768 \
  --seq-len 512 \
  --per-device-batch-size 2 \
  --warmup-steps 5 \
  --timed-steps 30
```

For a matched one-process-per-GPU topology, use:

```bash
XLA_FLAGS="--xla_gpu_all_reduce_combine_threshold_bytes=29360128 --xla_gpu_enable_nccl_comm_splitting=true" \
uv run run_jax_process_per_gpu.py \
  --num-processes 8 \
  --wandb-mode disabled \
  --output-dir results/process_per_gpu \
  --dtype bfloat16 \
  --gradient-communication-dtype float32 \
  --jax-attention-implementation cudnn_flash \
  --warmup-steps 5 \
  --timed-steps 30
```

## Historical result

The original internal experiment used eight NVIDIA L40S GPUs. With FP32
gradient communication, two one-process-per-GPU JAX runs averaged 134,794.7
tokens/s (0.060774 s/step), 1.5% above the measured PyTorch DDP baseline. With
BF16 gradient communication, the best tuned JAX run reached 189,074.9 tokens/s
and PyTorch reached 200,013.8 tokens/s.

See [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md) for the complete aggregate
table and limitations. These figures came from the original internal model
implementation. The self-contained JAX model in this sanitized repository
preserves the public Qwen3 architecture and benchmark mechanics, but new results
must be treated as a fresh experiment rather than an exact reproduction.
