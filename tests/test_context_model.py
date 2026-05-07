import torch

from models.context_model import (
    ContextModelConfig,
    TokenContextTransformer,
    fill_missing_with_fallback,
    flatten_token_maps,
    unflatten_token_maps,
)


def test_context_transformer_forward_and_loss_shapes():
    model = TokenContextTransformer(
        ContextModelConfig(codebook_size=32, levels=4, max_sequence_length=32, d_model=32, num_layers=1, num_heads=4)
    )
    tokens = torch.randint(0, 32, (2, 12))
    levels = torch.randint(0, 4, (2, 12))

    logits = model(tokens, levels)
    loss = model.next_token_loss(tokens, levels)

    assert logits.shape == (2, 12, 32)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_flatten_and_unflatten_token_maps_round_trip():
    maps = [
        torch.randint(0, 8, (2, 2, 2)),
        torch.randint(0, 8, (2, 4, 4)),
    ]

    flat, levels, shapes = flatten_token_maps(maps)
    restored = unflatten_token_maps(flat, shapes)

    assert flat.shape == (2, 20)
    assert levels.shape == (2, 20)
    assert shapes == [(2, 2), (4, 4)]
    assert all(torch.equal(a, b) for a, b in zip(maps, restored))


def test_fill_missing_with_fallback_appends_expected_shapes():
    received = [torch.ones(2, 2, 2, dtype=torch.long)]
    target_shapes = [(2, 2), (4, 4), (8, 8)]
    filled = fill_missing_with_fallback(received, target_shapes, [1, 3, 5])

    assert len(filled) == 3
    assert filled[1].shape == (2, 4, 4)
    assert filled[2].shape == (2, 8, 8)
    assert torch.all(filled[1] == 3)
    assert torch.all(filled[2] == 5)


def test_greedy_predict_uses_fallback_beyond_budget():
    model = TokenContextTransformer(
        ContextModelConfig(codebook_size=16, levels=4, max_sequence_length=32, d_model=32, num_layers=1, num_heads=4)
    )
    prefix = torch.randint(0, 16, (2, 4))
    prefix_levels = torch.zeros_like(prefix)
    target_levels = torch.tensor([1, 1, 2, 2, 2])
    fallback = torch.tensor([7, 7, 9, 9, 9])

    predicted = model.greedy_predict(prefix, prefix_levels, target_levels, fallback, max_autoregressive_steps=2)

    assert predicted.shape == (2, 5)
    assert torch.all(predicted[:, 2:] == torch.tensor([9, 9, 9]))
