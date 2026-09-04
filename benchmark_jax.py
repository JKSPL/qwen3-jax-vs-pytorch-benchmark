#!/usr/bin/env python3
from __future__ import annotations

import sys

from benchmark import main

DEFAULT_ARGS = [
    "--framework",
    "jax",
    "--dtype",
    "bfloat16",
    "--gradient-communication-dtype",
    "bfloat16",
    "--jax-attention-implementation",
    "cudnn_flash",
    "--run-name",
    "jax-bf16-cudnn-flash-nvidia-tips",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()
