import jax.numpy as jnp
from flax import nnx

from qwen3_jax import Qwen3ForCausalLM


def test_tiny_model_forward_and_loss_are_finite() -> None:
    model = Qwen3ForCausalLM(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8,
        attention_implementation="explicit",
        dtype=None,
        rngs=nnx.Rngs(0),
    )
    tokens = jnp.arange(8, dtype=jnp.int32)[None, :] % 32
    output = model(
        {
            "input_ids": tokens,
            "labels": tokens,
            "position_ids": jnp.arange(8, dtype=jnp.int32)[None, :],
        }
    )
    assert output.logits.shape == (1, 8, 32)
    assert output.loss is not None
    assert bool(jnp.isfinite(output.loss))
