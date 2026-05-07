from pathlib import Path

import torch

from train.train_context_model import collect_frame_refs, parse_raw_clip_name


def test_parse_raw_clip_name_for_context_training():
    meta = parse_raw_clip_name(Path("news_cif_352x288_29.97fps_8bit_P420.yuv"))

    assert meta["width"] == 352
    assert meta["height"] == 288
    assert meta["fps"] == 29.97


def test_context_training_random_batch_shape():
    from train.train_context_model import random_batch

    tokens = torch.arange(40).view(4, 10)
    levels = torch.zeros_like(tokens)
    batch_tokens, batch_levels = random_batch(tokens, levels, 2)

    assert batch_tokens.shape == (2, 10)
    assert batch_levels.shape == (2, 10)
