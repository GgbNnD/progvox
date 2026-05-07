#!/usr/bin/env python3
"""Analyze multi-scale tokenizer quality/rate trade-offs on prepared clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def frames_to_tensor(frames: list[np.ndarray], resize: int, device: torch.device) -> torch.Tensor:
    arrays = []
    for frame in frames:
        resized = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        arrays.append(resized.astype(np.float32) / 255.0)
    batch = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((reference - reconstruction).pow(2), dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_batch(reference: torch.Tensor, reconstruction: torch.Tensor) -> list[float]:
    ref = reference.detach().cpu().permute(0, 2, 3, 1).numpy()
    rec = reconstruction.detach().cpu().permute(0, 2, 3, 1).numpy()
    return [
        float(structural_similarity(ref[i], rec[i], channel_axis=-1, data_range=1.0))
        for i in range(ref.shape[0])
    ]


def write_visual_comparison(
    original: torch.Tensor,
    reconstructions: list[torch.Tensor],
    output: Path,
) -> None:
    images = [original[0], *[reconstruction[0] for reconstruction in reconstructions]]
    titles = ["Original", *[f"Layer 0-{i}" for i in range(len(reconstructions))]]
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.0 * len(images), 3.2))
    for index, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), index)
        plt.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "max_level",
                "frames",
                "token_bits_per_frame",
                "estimated_kbps_30fps",
                "psnr_rgb",
                "ssim_rgb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output: Path, visual: Path, sample_count: int) -> None:
    lines = [
        "# Tokenizer Analysis",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Sampled frames: {sample_count}",
        f"- Visual comparison: `{visual}`",
        "",
        "| max layer | token bits/frame | est. kbps @30fps | PSNR-RGB | SSIM-RGB |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {max_level} | {token_bits_per_frame} | {estimated_kbps_30fps:.3f} | {psnr_rgb:.3f} | {ssim_rgb:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Layer prefixes monotonically increase token rate and generally improve reconstruction fidelity.",
            "- This tokenizer uses an untrained uniform RGB residual codebook, so the results are a functional baseline rather than the final compression quality target.",
            "- The same API supports learnable codebooks for the later tokenizer training deliverable.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def write_notebook(output: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Tokenizer Analysis\n\n"
            "This notebook records the phase-2.1 tokenizer validation workflow. "
            "The committed CSV/Markdown/PNG artifacts are generated by `scripts/analyze_tokenizer.py`."
        ),
        nbf.v4.new_code_cell(
            "!conda run -n alg python scripts/analyze_tokenizer.py "
            "--input data/processed/xiph_small_clips --sample-count 50 --resize 128"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "pd.read_csv('reports/tokenizer_analysis.csv')"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image('reports/tokenizer_visual_comparison.png'))"
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/xiph_small_clips"))
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--csv-output", type=Path, default=Path("reports/tokenizer_analysis.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/tokenizer_analysis.md"))
    parser.add_argument("--visual-output", type=Path, default=Path("reports/tokenizer_visual_comparison.png"))
    parser.add_argument("--notebook-output", type=Path, default=Path("analysis_tokenizer.ipynb"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/tokenizer_analysis.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MultiScaleResidualTokenizer(
        TokenizerConfig(levels=4, codebook_size=512, learnable_codebooks=False)
    ).to(device)
    tokenizer.eval()

    refs = collect_frame_refs(args.input, args.sample_count)
    level_psnr: list[list[float]] = [[] for _ in range(tokenizer.config.levels)]
    level_ssim: list[list[float]] = [[] for _ in range(tokenizer.config.levels)]
    token_bits = [0 for _ in range(tokenizer.config.levels)]
    first_original = None
    first_reconstructions = None

    with torch.no_grad():
        for start in range(0, len(refs), args.batch_size):
            batch_refs = refs[start : start + args.batch_size]
            frames = [read_yuv420_frame(path, index) for path, index in batch_refs]
            x = frames_to_tensor(frames, args.resize, device)
            output = tokenizer(x)
            for level, reconstruction in enumerate(output.reconstructions):
                level_psnr[level].extend(psnr(x, reconstruction).detach().cpu().tolist())
                level_ssim[level].extend(ssim_batch(x, reconstruction))
                if start == 0:
                    token_bits[level] = output.rate_bits(level) // x.size(0)
            if first_original is None:
                first_original = x.detach().cpu()
                first_reconstructions = [reconstruction.detach().cpu() for reconstruction in output.reconstructions]

    rows = []
    for level in range(tokenizer.config.levels):
        bits_per_frame = token_bits[level]
        rows.append(
            {
                "max_level": level,
                "frames": len(refs),
                "token_bits_per_frame": bits_per_frame,
                "estimated_kbps_30fps": bits_per_frame * 30.0 / 1000.0,
                "psnr_rgb": float(np.mean(level_psnr[level])),
                "ssim_rgb": float(np.mean(level_ssim[level])),
            }
        )

    assert first_original is not None and first_reconstructions is not None
    write_csv(rows, args.csv_output)
    write_markdown(rows, args.markdown_output, args.visual_output, len(refs))
    write_visual_comparison(first_original, first_reconstructions, args.visual_output)
    write_notebook(args.notebook_output)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "sample_count": len(refs),
        "resize": args.resize,
        "rows": rows,
        "input_refs": [{"path": str(path), "frame_index": index} for path, index in refs],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "frames": len(refs), "device": str(device)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
