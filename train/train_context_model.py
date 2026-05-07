#!/usr/bin/env python3
"""Train the autoregressive token context model on tokenizer sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.context_model import ContextModelConfig, TokenContextTransformer, flatten_token_maps  # noqa: E402
from models.tokenizer import MultiScaleResidualTokenizer, TokenizerConfig  # noqa: E402


YUV_NAME_RE = re.compile(
    r"(?P<width>\d+)x(?P<height>\d+)_(?P<fps>[\d.]+)fps_(?P<bitdepth>\d+)bit_(?P<format>P420|I420|yuv420p)",
    re.IGNORECASE,
)


def parse_raw_clip_name(path: Path) -> dict[str, Any]:
    match = YUV_NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse raw clip metadata from {path.name}")
    return {
        "width": int(match.group("width")),
        "height": int(match.group("height")),
        "fps": float(match.group("fps")),
    }


def read_yuv420_frame(path: Path, frame_index: int) -> np.ndarray:
    meta = parse_raw_clip_name(path)
    width, height = meta["width"], meta["height"]
    frame_size = width * height * 3 // 2
    with path.open("rb") as handle:
        handle.seek(frame_index * frame_size)
        raw = handle.read(frame_size)
    if len(raw) != frame_size:
        raise ValueError(f"{path.name} does not contain frame {frame_index}")
    yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)


def collect_frame_refs(input_dir: Path, count: int) -> list[tuple[Path, int]]:
    clips = sorted(input_dir.rglob("*.yuv"))
    if not clips:
        raise RuntimeError(f"No .yuv clips found in {input_dir}")
    refs: list[tuple[Path, int]] = []
    per_clip = max(1, math.ceil(count / len(clips)))
    for clip in clips:
        meta = parse_raw_clip_name(clip)
        frame_size = meta["width"] * meta["height"] * 3 // 2
        total_frames = clip.stat().st_size // frame_size
        indices = np.linspace(0, max(0, total_frames - 1), num=min(per_clip, total_frames), dtype=int)
        refs.extend((clip, int(index)) for index in indices)
    return refs[:count]


def refs_to_tensor(refs: list[tuple[Path, int]], resize: int, device: torch.device) -> Tensor:
    frames = []
    for path, index in refs:
        rgb = read_yuv420_frame(path, index)
        rgb = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(rgb.astype(np.float32) / 255.0)
    data = np.stack(frames, axis=0)
    return torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().to(device)


@torch.no_grad()
def make_token_sequences(frames: Tensor, tokenizer_config: dict[str, Any]) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    tokenizer = MultiScaleResidualTokenizer(
        TokenizerConfig(
            levels=int(tokenizer_config["levels"]),
            codebook_size=int(tokenizer_config["codebook_size"]),
            max_token_resolution=int(tokenizer_config["max_token_resolution"]),
        )
    ).to(frames.device)
    tokenizer.eval()
    output = tokenizer(frames)
    tokens, levels, shapes = flatten_token_maps(output.tokens)
    return tokens.detach(), levels.detach(), shapes


def random_batch(tokens: Tensor, levels: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, tokens.size(0), (batch_size,), device=tokens.device)
    return tokens[indices], levels[indices]


@torch.no_grad()
def evaluate(model: TokenContextTransformer, tokens: Tensor, levels: Tensor, batch_size: int = 4) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for start in range(0, tokens.size(0), batch_size):
        x = tokens[start : start + batch_size]
        l = levels[start : start + batch_size]
        loss = model.next_token_loss(x, l)
        logits = model(x[:, :-1], l[:, :-1])
        pred = logits.argmax(dim=-1)
        target = x[:, 1:]
        correct += int((pred == target).sum().item())
        total += int(target.numel())
        losses.append(float(loss.item()))
    loss_value = float(np.mean(losses))
    return {
        "loss": loss_value,
        "perplexity": float(math.exp(min(loss_value, 20.0))),
        "accuracy": correct / total if total else 0.0,
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "val_loss", "val_perplexity", "val_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    metrics_csv: Path,
    checkpoint: Path,
    sequence_length: int,
    token_shapes: list[tuple[int, int]],
) -> None:
    final = rows[-1]
    best = min(rows, key=lambda row: row["val_loss"])
    lines = [
        "# Context Model Training",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input: `{config['input']}`",
        f"- Frames: {config['sample_frames']} total, {config['train_frames']} train",
        f"- Token sequence length: {sequence_length}",
        f"- Token shapes: `{token_shapes}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- TensorBoard logs: `{config['log_dir']}`",
        f"- Metrics CSV: `{metrics_csv}`",
        "",
        "## Final Metrics",
        "",
        f"- Final train loss: {final['train_loss']:.4f}",
        f"- Final val loss: {final['val_loss']:.4f}",
        f"- Final val perplexity: {final['val_perplexity']:.3f}",
        f"- Final val next-token accuracy: {final['val_accuracy']:.4f}",
        f"- Best val loss: {best['val_loss']:.4f} at step {best['step']}",
        "",
        "## Metrics By Evaluation Step",
        "",
        "| step | train loss | val loss | val perplexity | val accuracy |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_perplexity']:.3f} | {row['val_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is the first formal training run on the local Xiph sample set.",
            "- The model trains next-token prediction over flattened multi-scale tokenizer outputs.",
            "- Checkpoints and TensorBoard event files are intentionally ignored by git.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("train/train_context_model.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    refs = collect_frame_refs(Path(config["input"]), int(config["sample_frames"]))
    frames = refs_to_tensor(refs, int(config["resize"]), device)
    tokens, levels, token_shapes = make_token_sequences(frames, config["tokenizer"])
    train_count = int(config["train_frames"])
    train_tokens, val_tokens = tokens[:train_count], tokens[train_count:]
    train_levels, val_levels = levels[:train_count], levels[train_count:]

    context_cfg = ContextModelConfig(
        codebook_size=int(config["tokenizer"]["codebook_size"]),
        levels=int(config["tokenizer"]["levels"]),
        max_sequence_length=tokens.size(1),
        d_model=int(config["context_model"]["d_model"]),
        num_layers=int(config["context_model"]["num_layers"]),
        num_heads=int(config["context_model"]["num_heads"]),
        dropout=float(config["context_model"]["dropout"]),
    )
    model = TokenContextTransformer(context_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    writer = SummaryWriter(str(config["log_dir"]))
    rows: list[dict[str, Any]] = []

    for step in range(1, int(config["train_steps"]) + 1):
        model.train()
        batch_tokens, batch_levels = random_batch(train_tokens, train_levels, int(config["batch_size"]))
        loss = model.next_token_loss(batch_tokens, batch_levels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        writer.add_scalar("loss/train", float(loss.item()), step)

        if step == 1 or step % int(config["eval_every"]) == 0 or step == int(config["train_steps"]):
            val = evaluate(model, val_tokens, val_levels)
            writer.add_scalar("loss/val", val["loss"], step)
            writer.add_scalar("accuracy/val", val["accuracy"], step)
            rows.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": val["loss"],
                    "val_perplexity": val["perplexity"],
                    "val_accuracy": val["accuracy"],
                }
            )
    writer.close()

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "context_model.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(context_cfg),
            "token_shapes": token_shapes,
            "training_config": config,
        },
        checkpoint,
    )
    report_dir = Path(config["report_dir"])
    metrics_csv = report_dir / "context_model_training_metrics.csv"
    metrics_json = report_dir / "context_model_training_metrics.json"
    write_csv(rows, metrics_csv)
    metrics_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "sequence_length": int(tokens.size(1)),
                "token_shapes": token_shapes,
                "checkpoint": str(checkpoint),
                "rows": rows,
                "config": config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(rows, config, Path(config["docs_report"]), metrics_csv, checkpoint, tokens.size(1), token_shapes)
    print(json.dumps({"ok": True, "device": str(device), "steps": config["train_steps"], "checkpoint": str(checkpoint)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
