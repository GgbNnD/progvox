import torch

from models.tokenizer import MultiScaleResidualTokenizer, TokenizerConfig


def make_test_frame(batch_size=2, size=64):
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing="ij",
    )
    frame = torch.stack([xx, yy, (xx + yy) / 2], dim=0)
    return frame.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def mse(a, b):
    return torch.mean((a - b).pow(2)).item()


def test_tokenizer_outputs_four_token_maps_with_expected_shapes():
    tokenizer = MultiScaleResidualTokenizer(TokenizerConfig(levels=4, codebook_size=512))
    x = make_test_frame(size=64)

    output = tokenizer(x)

    assert len(output.tokens) == 4
    assert output.shapes == [(4, 4), (8, 8), (16, 16), (32, 32)]
    assert [tuple(token.shape) for token in output.tokens] == [
        (2, 4, 4),
        (2, 8, 8),
        (2, 16, 16),
        (2, 32, 32),
    ]
    for token in output.tokens:
        assert token.dtype == torch.long
        assert token.min() >= 0
        assert token.max() < 512


def test_reconstruction_from_tokens_matches_saved_reconstructions():
    tokenizer = MultiScaleResidualTokenizer(TokenizerConfig(levels=4, codebook_size=512))
    x = make_test_frame(size=64)
    output = tokenizer(x)

    for level in range(4):
        reconstructed = tokenizer.reconstruct(output.tokens, output_shape=(64, 64), max_level=level)
        assert torch.allclose(reconstructed, output.reconstructions[level], atol=1e-6)


def test_finer_token_prefix_improves_reconstruction_quality():
    tokenizer = MultiScaleResidualTokenizer(TokenizerConfig(levels=4, codebook_size=512))
    x = make_test_frame(size=64)
    output = tokenizer(x)

    errors = [mse(x, reconstruction) for reconstruction in output.reconstructions]

    assert errors[-1] < errors[0]
    assert errors[-1] < 0.01


def test_rate_accounting_increases_with_each_layer():
    tokenizer = MultiScaleResidualTokenizer(TokenizerConfig(levels=4, codebook_size=512))
    output = tokenizer(make_test_frame(size=64))

    bit_counts = [output.rate_bits(level) for level in range(4)]

    assert bit_counts == sorted(bit_counts)
    assert output.bits_per_token == 9
    assert bit_counts[0] == 2 * 4 * 4 * 9
    assert bit_counts[-1] == sum(token.numel() for token in output.tokens) * 9
