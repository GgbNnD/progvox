from pathlib import Path

import torch

from run_progvc_codec import fallback_predict_tokens, parse_raw_clip_name
from models.context_model import ContextModelConfig, TokenContextTransformer


def test_parse_raw_clip_name_reads_compressai_style_metadata():
    meta = parse_raw_clip_name(Path("foreman_cif_352x288_29.97fps_8bit_P420.yuv"))

    assert meta["width"] == 352
    assert meta["height"] == 288
    assert meta["fps"] == 29.97
    assert meta["bitdepth"] == 8


def test_fallback_predict_tokens_returns_full_scale_set():
    model = TokenContextTransformer(
        ContextModelConfig(codebook_size=16, levels=3, max_sequence_length=128, d_model=32, num_layers=1, num_heads=4)
    )
    received = [torch.ones(1, 2, 2, dtype=torch.long)]
    shapes = [(2, 2), (4, 4), (8, 8)]

    filled = fallback_predict_tokens(model, received, shapes, [1, 3, 5])

    assert len(filled) == 3
    assert torch.all(filled[1] == 3)
    assert torch.all(filled[2] == 5)
