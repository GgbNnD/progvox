#!/usr/bin/env python3
"""Run the offline ProGVC tokenizer -> context -> generator codec prototype."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import Tensor

from models.context_model import (
    ContextModelConfig,
    TokenContextTransformer,
    flatten_token_maps,
    unflatten_token_maps,
)
from models.generator import CNNGeneratorConfig, DetailSynthesisCNN
from models.tokenizer import MultiScaleResidualTokenizer, TokenizerConfig


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
        "bitdepth": int(match.group("bitdepth")),
    }


def choose_input(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*_352x288_*.yuv")) or sorted(path.rglob("*.yuv"))
    if not candidates:
        raise RuntimeError(f"No .yuv clips found in {path}")
    return candidates[0]


def read_yuv420_frames(path: Path, frames: int, resize: int, device: torch.device) -> tuple[Tensor, float]:
    meta = parse_raw_clip_name(path)
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    frame_size = width * height * 3 // 2
    available = path.stat().st_size // frame_size
    frames = min(frames, available)
    images = []
    with path.open("rb") as handle:
        for _ in range(frames):
            raw = handle.read(frame_size)
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            rgb = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_AREA)
            images.append(rgb.astype(np.float32) / 255.0)
    tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).contiguous().to(device)
    return tensor, fps


def psnr(pred: Tensor, target: Tensor) -> float:
    mse = torch.mean((pred - target).pow(2), dim=(1, 2, 3)).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).mean().item())


def ssim(pred: Tensor, target: Tensor) -> float:
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    values = [
        structural_similarity(target_np[i], pred_np[i], channel_axis=-1, data_range=1.0)
        for i in range(pred_np.shape[0])
    ]
    return float(np.mean(values))


def make_lpips(device: torch.device):
    try:
        import lpips

        metric = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).to(device).eval()
        return metric, "lpips-alex-rand"
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return None, f"unavailable: {exc}"


def lpips_value(metric, pred: Tensor, target: Tensor) -> float:
    if metric is None:
        return float("nan")
    with torch.no_grad():
        return float(metric(pred * 2 - 1, target * 2 - 1).mean().item())


def load_generator(checkpoint: Path, device: torch.device) -> DetailSynthesisCNN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = CNNGeneratorConfig(**payload.get("config", {}))
    model = DetailSynthesisCNN(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def fallback_predict_tokens(
    context_model: TokenContextTransformer,
    received_tokens: list[Tensor],
    target_shapes: list[tuple[int, int]],
    fallback_token_ids: list[int],
) -> list[Tensor]:
    """Use the context-model API with zero AR budget for deterministic filling."""

    flat_prefix, level_prefix, _ = flatten_token_maps(received_tokens)
    missing_shapes = target_shapes[len(received_tokens) :]
    if not missing_shapes:
        return received_tokens

    missing_level_ids = []
    missing_fallback_ids = []
    for level, (height, width) in enumerate(missing_shapes, start=len(received_tokens)):
        count = height * width
        missing_level_ids.extend([level] * count)
        missing_fallback_ids.extend([fallback_token_ids[level]] * count)
    level_ids = torch.tensor(missing_level_ids, device=flat_prefix.device, dtype=torch.long)
    fallback_ids = torch.tensor(missing_fallback_ids, device=flat_prefix.device, dtype=torch.long)
    predicted = context_model.greedy_predict(
        flat_prefix,
        level_prefix,
        level_ids,
        fallback_token_ids=fallback_ids,
        max_autoregressive_steps=0,
    )
    missing_maps = unflatten_token_maps(predicted, missing_shapes)
    return [*received_tokens, *missing_maps]


def write_video(frames: Tensor, output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    array = (frames.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255).astype(np.uint8)
    height, width = array.shape[1:3]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in array:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_sample_grid(target: Tensor, variants: dict[str, Tensor], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["target", *variants.keys()]
    images = [target[0], *[value[0] for value in variants.values()]]
    plt.figure(figsize=(3.0 * len(images), 3.2))
    for index, (name, image) in enumerate(zip(names, images), start=1):
        plt.subplot(1, len(images), index)
        plt.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def load_ssf2020_reference(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] == "aggregate":
                rows.append(
                    {
                        "quality": int(row["quality"]),
                        "bitrate_kbps": float(row["bitrate_kbps"]),
                        "psnr_y": float(row["psnr_y"]),
                        "ms_ssim_rgb": float(row["ms_ssim_rgb"]),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "max_level", "bitrate_kbps", "psnr_rgb", "ssim_rgb", "lpips", "video_path"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, Any]],
    ssf_rows: list[dict[str, Any]],
    output: Path,
    metrics_csv: Path,
    sample_image: Path,
    input_clip: Path,
    lpips_mode: str,
) -> None:
    lines = [
        "# ProGVC Integration Test",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input clip: `{input_clip}`",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Sample image: `{sample_image}`",
        f"- LPIPS mode: `{lpips_mode}`",
        "- Context predictor: untrained Transformer interface with zero-residual fallback for missing token layers.",
        "",
        "## ProGVC Prototype Results",
        "",
        "| variant | max layer | bitrate kbps | PSNR-RGB | SSIM-RGB | LPIPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['max_level']} | {row['bitrate_kbps']:.3f} | {row['psnr_rgb']:.3f} | {row['ssim_rgb']:.4f} | {row['lpips']:.4f} |"
        )
    if ssf_rows:
        lines.extend(["", "## SSF2020 Reference From Phase 1", "", "| quality | bitrate kbps | PSNR-Y | MS-SSIM-RGB |", "| ---: | ---: | ---: | ---: |"])
        for row in ssf_rows:
            lines.append(
                f"| {row['quality']} | {row['bitrate_kbps']:.3f} | {row['psnr_y']:.3f} | {row['ms_ssim_rgb']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is an offline smoke integration of tokenizer, context-model API, and generator, not the final trained ProGVC result.",
            "- Layer-2 mode exercises missing-token completion; Layer-3 mode is the full-token reconstruction path near the first-stage 300-500 kbps target.",
            "- Generated videos are written under `reports/videos/` and intentionally ignored by git.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/xiph_small_clips"))
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--max-levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/generator/cnn_gan.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/progvc_codec"))
    parser.add_argument("--video-dir", type=Path, default=Path("reports/videos/progvc_codec"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("reports/progvc_integration_metrics.csv"))
    parser.add_argument("--metrics-json", type=Path, default=Path("reports/progvc_integration_metrics.json"))
    parser.add_argument("--sample-image", type=Path, default=Path("reports/progvc_integration_samples.png"))
    parser.add_argument("--report", type=Path, default=Path("docs/progvc_integration_test.md"))
    parser.add_argument("--ssf2020-csv", type=Path, default=Path("reports/ssf2020_rd_points.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_clip = choose_input(args.input)
    target, fps = read_yuv420_frames(input_clip, args.frames, args.resize, device)
    tokenizer = MultiScaleResidualTokenizer(TokenizerConfig(levels=4, codebook_size=512, max_token_resolution=32)).to(device)
    context_model = TokenContextTransformer(
        ContextModelConfig(codebook_size=512, levels=4, max_sequence_length=2048, d_model=64, num_layers=1, num_heads=4)
    ).to(device)
    context_model.eval()
    generator = load_generator(args.generator_checkpoint, device)
    lpips_metric, lpips_mode = make_lpips(device)

    rows = []
    sample_variants: dict[str, Tensor] = {}
    with torch.no_grad():
        tokenized = tokenizer(target)
        zero_ids = tokenizer.zero_token_ids()
        for max_level in args.max_levels:
            received = [token.clone() for token in tokenized.tokens[: max_level + 1]]
            filled = fallback_predict_tokens(context_model, received, tokenized.shapes, zero_ids)
            condition = tokenizer.reconstruct(filled, output_shape=(args.resize, args.resize), max_level=len(filled) - 1)
            generated = generator(condition)
            bitrate = tokenized.rate_bits(max_level) / target.size(0) * fps / 1000.0
            variant = f"progvc_l{max_level}_cnn_gan"
            video_path = args.video_dir / f"{variant}.mp4"
            write_video(generated, video_path, fps)
            sample_variants[f"cond_l{max_level}"] = condition.detach().cpu()
            sample_variants[f"gen_l{max_level}"] = generated.detach().cpu()
            rows.append(
                {
                    "variant": variant,
                    "max_level": max_level,
                    "bitrate_kbps": bitrate,
                    "psnr_rgb": psnr(generated, target),
                    "ssim_rgb": ssim(generated, target),
                    "lpips": lpips_value(lpips_metric, generated, target),
                    "video_path": str(video_path),
                }
            )

    write_sample_grid(target.detach().cpu(), sample_variants, args.sample_image)
    write_csv(rows, args.metrics_csv)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "input_clip": str(input_clip),
        "frames": int(target.size(0)),
        "resize": args.resize,
        "fps": fps,
        "rows": rows,
        "lpips_mode": lpips_mode,
        "context_mode": "untrained_transformer_zero_residual_fallback",
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(
        rows,
        load_ssf2020_reference(args.ssf2020_csv),
        args.report,
        args.metrics_csv,
        args.sample_image,
        input_clip,
        lpips_mode,
    )
    print(json.dumps({"ok": True, "device": str(device), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
