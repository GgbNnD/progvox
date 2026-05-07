#!/usr/bin/env python3
"""Train a small generator ablation suite for phase 2.2."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import linalg
from skimage.metrics import structural_similarity
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator import (  # noqa: E402
    CNNGeneratorConfig,
    DetailSynthesisCNN,
    DiffusionConfig,
    PatchDiscriminator,
    TinyConditionalDiffusion,
)
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


def make_condition(target: Tensor, condition_level: int) -> Tensor:
    tokenizer = MultiScaleResidualTokenizer(
        TokenizerConfig(levels=4, codebook_size=512, max_token_resolution=32)
    ).to(target.device)
    tokenizer.eval()
    with torch.no_grad():
        output = tokenizer(target)
        return output.reconstructions[condition_level].detach()


def random_batch(condition: Tensor, target: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, target.size(0), (batch_size,), device=target.device)
    return condition[indices], target[indices]


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


def feature_stats(images: Tensor) -> np.ndarray:
    images_np = images.detach().cpu().numpy()
    means = images_np.mean(axis=(2, 3))
    stds = images_np.std(axis=(2, 3))
    pooled = F.adaptive_avg_pool2d(images.detach().cpu(), (4, 4)).reshape(images.size(0), -1).numpy()
    return np.concatenate([means, stds, pooled], axis=1)


def frechet_distance(pred: Tensor, target: Tensor) -> float:
    x = feature_stats(pred)
    y = feature_stats(target)
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
    cov_y = np.cov(y, rowvar=False) + np.eye(y.shape[1]) * 1e-6
    covmean = linalg.sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x + cov_y - 2 * covmean))


def make_lpips(config: dict[str, Any], device: torch.device):
    if not config.get("enabled", True):
        return None, "disabled"
    try:
        import lpips

        metric = lpips.LPIPS(net="alex", pnet_rand=bool(config.get("pnet_rand", True)), verbose=False)
        metric.to(device).eval()
        return metric, "lpips-alex-rand" if config.get("pnet_rand", True) else "lpips-alex"
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return None, f"unavailable: {exc}"


def lpips_value(metric, pred: Tensor, target: Tensor) -> float:
    if metric is None:
        return float("nan")
    with torch.no_grad():
        return float(metric(pred * 2 - 1, target * 2 - 1).mean().item())


def train_pure_cnn(
    train_condition: Tensor,
    train_target: Tensor,
    config: dict[str, Any],
    writer: SummaryWriter,
) -> DetailSynthesisCNN:
    model = DetailSynthesisCNN(
        CNNGeneratorConfig(base_channels=int(config["base_channels"]))
    ).to(train_target.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    steps = int(config["models"]["pure_cnn"]["train_steps"])
    for step in range(steps):
        condition, target = random_batch(train_condition, train_target, int(config["batch_size"]))
        pred = model(condition)
        loss = F.l1_loss(pred, target) + 0.25 * F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/reconstruction", float(loss.item()), step)
    return model


def train_cnn_gan(
    train_condition: Tensor,
    train_target: Tensor,
    config: dict[str, Any],
    writer: SummaryWriter,
) -> tuple[DetailSynthesisCNN, PatchDiscriminator]:
    generator = DetailSynthesisCNN(
        CNNGeneratorConfig(base_channels=int(config["base_channels"]))
    ).to(train_target.device)
    discriminator = PatchDiscriminator(base_channels=int(config["base_channels"])).to(train_target.device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=float(config["learning_rate"]))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=float(config["gan_learning_rate"]))
    steps = int(config["models"]["cnn_gan"]["train_steps"])
    gan_weight = float(config["gan_weight"])
    for step in range(steps):
        condition, target = random_batch(train_condition, train_target, int(config["batch_size"]))
        with torch.no_grad():
            fake = generator(condition)
        real_logits = discriminator(condition, target)
        fake_logits = discriminator(condition, fake.detach())
        d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        fake = generator(condition)
        fake_logits = discriminator(condition, fake)
        recon_loss = F.l1_loss(fake, target) + 0.25 * F.mse_loss(fake, target)
        g_loss = recon_loss + gan_weight * F.softplus(-fake_logits).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        writer.add_scalar("loss/generator", float(g_loss.item()), step)
        writer.add_scalar("loss/discriminator", float(d_loss.item()), step)
    return generator, discriminator


def train_diffusion(
    train_condition: Tensor,
    train_target: Tensor,
    config: dict[str, Any],
    writer: SummaryWriter,
) -> TinyConditionalDiffusion:
    diffusion = TinyConditionalDiffusion(
        DiffusionConfig(
            base_channels=int(config["base_channels"]),
            timesteps=int(config["diffusion"].get("timesteps", 16)),
        )
    ).to(train_target.device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=float(config["learning_rate"]))
    steps = int(config["diffusion"]["train_steps"])
    for step in range(steps):
        condition, target = random_batch(train_condition, train_target, int(config["batch_size"]))
        loss = diffusion.training_loss(condition, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/noise_prediction", float(loss.item()), step)
    return diffusion


@torch.no_grad()
def evaluate_direct(model: DetailSynthesisCNN, condition: Tensor) -> Tensor:
    model.eval()
    return model(condition)


@torch.no_grad()
def evaluate_diffusion(model: TinyConditionalDiffusion, condition: Tensor, steps: int) -> Tensor:
    model.eval()
    return model.sample(condition, steps=steps)


def save_sample_grid(
    target: Tensor,
    condition: Tensor,
    predictions: dict[str, Tensor],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["target", "condition", *predictions.keys()]
    images = [target[0], condition[0], *[value[0] for value in predictions.values()]]
    plt.figure(figsize=(3.0 * len(images), 3.2))
    for index, (name, image) in enumerate(zip(names, images), start=1):
        plt.subplot(1, len(images), index)
        plt.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "psnr", "ssim", "lpips", "fid_lite"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    csv_path: Path,
    sample_image: Path,
    checkpoint_paths: dict[str, str],
    lpips_mode: str,
) -> None:
    best = max(rows, key=lambda row: (row["psnr"], -row["lpips"] if not math.isnan(row["lpips"]) else 0))
    lines = [
        "# Generator Ablation",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input clips: `{config['input']}`",
        f"- Resize: {config['resize']} px",
        f"- Condition: tokenizer reconstruction through layer {config['condition_level']}",
        f"- TensorBoard logs: `{config['log_dir']}`",
        f"- Local checkpoints: `{config['checkpoint_dir']}`",
        f"- Metrics CSV: `{csv_path}`",
        f"- Sample image: `{sample_image}`",
        f"- LPIPS mode: `{lpips_mode}`",
        "",
        "| model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID-lite ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['psnr']:.3f} | {row['ssim']:.4f} | {row['lpips']:.4f} | {row['fid_lite']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Current recommended generator: `{best['model']}` for this smoke-scale run.",
            "- This is a bootstrap ablation on a tiny local dataset, not a final convergence run.",
            "- Model weights and TensorBoard event files are intentionally kept out of git.",
            "",
            "## Checkpoints",
            "",
        ]
    )
    for name, path in checkpoint_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Metric Notes",
            "",
            "- `FID-lite` is a Fréchet distance over color and pooled image features, used here as a deterministic no-download proxy.",
            "- LPIPS uses random AlexNet perceptual weights by default to avoid external downloads in this environment; flip `lpips.pnet_rand` to `false` for publication-grade LPIPS.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("train/train_generator.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_frames = int(config["train_frames"]) + int(config["val_frames"])
    refs = collect_frame_refs(Path(config["input"]), total_frames)
    target = refs_to_tensor(refs, int(config["resize"]), device)
    condition = make_condition(target, int(config["condition_level"]))
    train_target = target[: int(config["train_frames"])]
    val_target = target[int(config["train_frames"]) :]
    train_condition = condition[: int(config["train_frames"])]
    val_condition = condition[int(config["train_frames"]) :]

    log_dir = Path(config["log_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lpips_metric, lpips_mode = make_lpips(config.get("lpips", {}), device)
    predictions: dict[str, Tensor] = {}
    checkpoint_paths: dict[str, str] = {}

    with SummaryWriter(str(log_dir / "pure_cnn")) as writer:
        pure_cnn = train_pure_cnn(train_condition, train_target, config, writer)
    pure_path = checkpoint_dir / "pure_cnn.pth"
    torch.save({"model": pure_cnn.state_dict(), "config": asdict(pure_cnn.config)}, pure_path)
    checkpoint_paths["pure_cnn"] = str(pure_path)
    predictions["pure_cnn"] = evaluate_direct(pure_cnn, val_condition)

    with SummaryWriter(str(log_dir / "cnn_gan")) as writer:
        cnn_gan, discriminator = train_cnn_gan(train_condition, train_target, config, writer)
    gan_path = checkpoint_dir / "cnn_gan.pth"
    disc_path = checkpoint_dir / "cnn_gan_discriminator.pth"
    torch.save({"model": cnn_gan.state_dict(), "config": asdict(cnn_gan.config)}, gan_path)
    torch.save({"model": discriminator.state_dict()}, disc_path)
    checkpoint_paths["cnn_gan"] = str(gan_path)
    checkpoint_paths["cnn_gan_discriminator"] = str(disc_path)
    predictions["cnn_gan"] = evaluate_direct(cnn_gan, val_condition)

    with SummaryWriter(str(log_dir / "tiny_diffusion")) as writer:
        diffusion = train_diffusion(train_condition, train_target, config, writer)
    diffusion_path = checkpoint_dir / "tiny_diffusion.pth"
    torch.save({"model": diffusion.state_dict(), "config": asdict(diffusion.config)}, diffusion_path)
    checkpoint_paths["tiny_diffusion"] = str(diffusion_path)
    predictions["tiny_diffusion"] = evaluate_diffusion(
        diffusion,
        val_condition,
        int(config["diffusion"]["sample_steps"]),
    )

    rows = []
    for name, pred in predictions.items():
        rows.append(
            {
                "model": name,
                "psnr": psnr(pred, val_target),
                "ssim": ssim(pred, val_target),
                "lpips": lpips_value(lpips_metric, pred, val_target),
                "fid_lite": frechet_distance(pred, val_target),
            }
        )

    report_dir = Path(config["report_dir"])
    csv_path = report_dir / "generator_ablation_metrics.csv"
    json_path = report_dir / "generator_ablation_metrics.json"
    sample_image = Path(config["sample_image"])
    write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "config": config,
                "rows": rows,
                "checkpoints": checkpoint_paths,
                "lpips_mode": lpips_mode,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    save_sample_grid(val_target, val_condition, predictions, sample_image)
    write_report(
        rows,
        config,
        Path(config["docs_report"]),
        csv_path,
        sample_image,
        checkpoint_paths,
        lpips_mode,
    )
    print(json.dumps({"ok": True, "device": str(device), "models": [row["model"] for row in rows]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
