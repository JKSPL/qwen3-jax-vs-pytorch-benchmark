"""Small, self-contained Flax NNX implementation of the public Qwen3 architecture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


@dataclass(frozen=True)
class CausalLMOutput:
    logits: Array
    loss: Array | None


def _rotate_half(x: Array) -> Array:
    midpoint = x.shape[-1] // 2
    return jnp.concatenate((-x[..., midpoint:], x[..., :midpoint]), axis=-1)


def _rotary_embeddings(
    position_ids: Array, head_dim: int, rope_theta: float, dtype: jnp.dtype
) -> tuple[Array, Array]:
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    frequencies = jnp.einsum("bt,d->btd", position_ids.astype(jnp.float32), inv_freq)
    embeddings = jnp.concatenate((frequencies, frequencies), axis=-1)
    return jnp.cos(embeddings).astype(dtype), jnp.sin(embeddings).astype(dtype)


def _apply_rotary(query: Array, key: Array, cosine: Array, sine: Array) -> tuple[Array, Array]:
    cosine = cosine[:, :, None, :]
    sine = sine[:, :, None, :]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def shifted_cross_entropy(logits: Array, labels: Array) -> Array:
    """Causal language-model loss, ignoring the final label position."""
    shifted_logits = logits[:, :-1].astype(jnp.float32)
    shifted_labels = labels[:, 1:]
    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, shifted_labels[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(token_log_probs)


class RMSNorm(nnx.Module):
    def __init__(self, size: int, epsilon: float = 1e-6) -> None:
        self.weight = nnx.Param(jnp.ones((size,), dtype=jnp.float32))
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        normalized = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(normalized), axis=-1, keepdims=True)
        normalized = normalized * jax.lax.rsqrt(variance + self.epsilon)
        return (normalized * self.weight[...]).astype(dtype)


class MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        compute_dtype: jnp.dtype | None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_implementation: Literal["explicit", "cudnn_flash"],
        compute_dtype: jnp.dtype | None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.kv_repeats = num_attention_heads // num_key_value_heads
        self.attention_implementation = attention_implementation
        self.q_proj = nnx.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            num_attention_heads * self.head_dim,
            hidden_size,
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def __call__(self, x: Array, cosine: Array, sine: Array) -> Array:
        batch, length, _ = x.shape
        query = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.num_attention_heads, self.head_dim)
        )
        key = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.num_key_value_heads, self.head_dim)
        )
        value = self.v_proj(x).reshape(batch, length, self.num_key_value_heads, self.head_dim)
        query, key = _apply_rotary(query, key, cosine, sine)
        key = jnp.repeat(key, self.kv_repeats, axis=2)
        value = jnp.repeat(value, self.kv_repeats, axis=2)

        if self.attention_implementation == "cudnn_flash":
            attended = jax.nn.dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                implementation="cudnn",
            )
        else:
            query_heads = jnp.transpose(query, (0, 2, 1, 3))
            key_heads = jnp.transpose(key, (0, 2, 1, 3))
            value_heads = jnp.transpose(value, (0, 2, 1, 3))
            scores = jnp.matmul(query_heads, jnp.swapaxes(key_heads, -1, -2)) * (
                self.head_dim**-0.5
            )
            causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, -jnp.inf)
            attended = jnp.transpose(
                jnp.matmul(jax.nn.softmax(scores, axis=-1), value_heads),
                (0, 2, 1, 3),
            )
        return self.o_proj(attended.reshape(batch, length, -1))


class DecoderLayer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_implementation: Literal["explicit", "cudnn_flash"],
        compute_dtype: jnp.dtype | None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.input_norm = RMSNorm(hidden_size)
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            attention_implementation,
            compute_dtype,
            rngs=rngs,
        )
        self.post_attention_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, compute_dtype, rngs=rngs)

    def __call__(self, x: Array, cosine: Array, sine: Array) -> Array:
        x = x + self.attention(self.input_norm(x), cosine, sine)
        return x + self.mlp(self.post_attention_norm(x))


class Qwen3ForCausalLM(nnx.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        attention_implementation: Literal["explicit", "cudnn_flash"],
        dtype: jnp.dtype | None,
        rngs: nnx.Rngs,
    ) -> None:
        del max_position_embeddings
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = hidden_size // num_attention_heads
        self.rope_theta = 10_000.0
        self.dtype = dtype
        self.embedding = nnx.Embed(
            vocab_size,
            hidden_size,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.layers = nnx.List(
            [
                DecoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    num_key_value_heads,
                    attention_implementation,
                    dtype,
                    rngs=rngs,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nnx.Linear(
            hidden_size,
            vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, batch: Mapping[str, Any]) -> CausalLMOutput:
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"]) if "labels" in batch else None
        position_ids = jnp.asarray(batch.get("position_ids"))
        hidden = self.embedding(input_ids)
        if self.dtype is not None:
            hidden = hidden.astype(self.dtype)
        cosine, sine = _rotary_embeddings(
            position_ids, self.head_dim, self.rope_theta, hidden.dtype
        )
        for layer in self.layers:
            hidden = layer(hidden, cosine, sine)
        logits = self.lm_head(self.final_norm(hidden))
        loss = shifted_cross_entropy(logits, labels) if labels is not None else None
        return CausalLMOutput(logits=logits, loss=loss)
