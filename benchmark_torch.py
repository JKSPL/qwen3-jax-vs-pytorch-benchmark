#!/usr/bin/env python3
from __future__ import annotations

import sys

from benchmark import main

DEFAULT_ARGS = [
    "--framework",
    "torch",
    "--dtype",
    "bfloat16",
    "--gradient-communication-dtype",
    "bfloat16",
    "--torch-attn-implementation",
    "sdpa",
    "--torch-compile",
    "--run-name",
    "torch-bf16-compile-sdpa",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()
