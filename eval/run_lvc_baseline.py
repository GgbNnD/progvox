#!/usr/bin/env python3
"""Run CompressAI SSF2020 on prepared YUV clips and summarize RD points."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


QUALITY_RE = re.compile(r"ssf2020-[^-]+-(?P<quality>\d+)-")
YUV_NAME_RE = re.compile(r"(?P<width>\d+)x(?P<height>\d+)_")


def run_command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "args": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def quality_from_name(name: str) -> int | None:
    match = QUALITY_RE.search(name)
    return int(match.group("quality")) if match else None


def dimensions_from_name(path: Path) -> tuple[int, int] | None:
    match = YUV_NAME_RE.search(path.name)
    if not match:
        return None
    return int(match.group("width")), int(match.group("height"))


def prepare_eval_input(input_dir: Path, work_dir: Path, min_side: int) -> tuple[Path, list[dict[str, Any]]]:
    clips = sorted(input_dir.rglob("*.yuv"))
    skipped = []
    selected = []
    for clip in clips:
        dims = dimensions_from_name(clip)
        if dims is None:
            skipped.append({"path": str(clip), "reason": "missing dimensions in filename"})
            continue
        width, height = dims
        if min(width, height) < min_side:
            skipped.append(
                {
                    "path": str(clip),
                    "reason": f"short side {min(width, height)} < required {min_side}",
                }
            )
            continue
        selected.append(clip)

    if not selected:
        raise RuntimeError(f"No compatible YUV clips found in {input_dir}")

    if not skipped:
        return input_dir, skipped

    work_dir.mkdir(parents=True, exist_ok=True)
    for old in work_dir.glob("*.yuv"):
        old.unlink()
    for clip in selected:
        target = work_dir / clip.name
        target.symlink_to(clip.resolve())
    return work_dir, skipped


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_aggregate_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    results = data["results"]
    rows = []
    for index, model_name in enumerate(results["q"]):
        rows.append(
            {
                "scope": "aggregate",
                "source": "all",
                "quality": quality_from_name(model_name),
                "model": model_name,
                "bitrate_kbps": results.get("bitrate", [None])[index],
                "psnr_y": results.get("psnr-y", [None])[index],
                "psnr_yuv": results.get("psnr-yuv", [None])[index],
                "psnr_rgb": results.get("psnr-rgb", [None])[index],
                "ms_ssim_rgb": results.get("ms-ssim-rgb", [None])[index],
                "mse_rgb": results.get("mse-rgb", [None])[index],
            }
        )
    return rows


def collect_sequence_rows(output_dir: Path, aggregate_path: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*.json")):
        if path == aggregate_path:
            continue
        data = load_json(path)
        model_name = data["description"].removeprefix("Inference (")
        results = data["results"]
        rows.append(
            {
                "scope": "sequence",
                "source": data["source"],
                "quality": quality_from_name(path.stem),
                "model": path.stem,
                "bitrate_kbps": results.get("bitrate"),
                "psnr_y": results.get("psnr-y"),
                "psnr_yuv": results.get("psnr-yuv"),
                "psnr_rgb": results.get("psnr-rgb"),
                "ms_ssim_rgb": results.get("ms-ssim-rgb"),
                "mse_rgb": results.get("mse-rgb"),
                "description": model_name,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "source",
        "quality",
        "model",
        "bitrate_kbps",
        "psnr_y",
        "psnr_yuv",
        "psnr_rgb",
        "ms_ssim_rgb",
        "mse_rgb",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(rows: list[dict[str, Any]], output: Path) -> None:
    aggregate = sorted(
        [row for row in rows if row["scope"] == "aggregate"],
        key=lambda row: row["quality"] or 0,
    )
    if not aggregate:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(
        [row["bitrate_kbps"] for row in aggregate],
        [row["psnr_y"] for row in aggregate],
        marker="o",
        label="PSNR-Y",
    )
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("PSNR-Y (dB)")
    plt.title("SSF2020 RD Points on Xiph Small")
    plt.grid(True, alpha=0.3)
    for row in aggregate:
        plt.annotate(f"q{row['quality']}", (row["bitrate_kbps"], row["psnr_y"]))
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def summarize_stderr(stderr: str, max_lines: int = 80) -> str:
    keep = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "it/s" in stripped or "B/s" in stripped:
            continue
        keep.append(stripped)
    return "\n".join(keep[-max_lines:])


def write_report(
    rows: list[dict[str, Any]],
    command: dict[str, Any],
    output: Path,
    csv_path: Path,
    aggregate_path: Path,
    plot_path: Path,
    entropy_estimation: bool,
    skipped: list[dict[str, Any]],
) -> None:
    aggregate = sorted(
        [row for row in rows if row["scope"] == "aggregate"],
        key=lambda row: row["quality"] or 0,
    )
    lines = [
        "# SSF2020 LVC Baseline",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Status: {'PASS' if command['returncode'] == 0 else 'FAIL'}",
        f"- Mode: {'entropy estimation' if entropy_estimation else 'actual entropy coding'}",
        f"- Aggregate JSON: `{aggregate_path}`",
        f"- CSV: `{csv_path}`",
        f"- RD plot: `{plot_path}`",
        f"- Skipped clips: {len(skipped)}",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command["args"]),
        "```",
        "",
        "## Aggregate RD Points",
        "",
        "| quality | bitrate kbps | PSNR-Y | PSNR-YUV | PSNR-RGB | MS-SSIM-RGB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {quality} | {bitrate_kbps:.3f} | {psnr_y:.3f} | {psnr_yuv:.3f} | {psnr_rgb:.3f} | {ms_ssim_rgb:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Input clips are the 64-frame YUV420 files generated by `scripts/prepare_test_videos.py`.",
            "- CompressAI's video eval path computes MS-SSIM with four downsamplings, so clips with short side < 161 are skipped by default.",
            "- The first run may download CompressAI pretrained SSF2020 weights into the local torch cache.",
        ]
    )
    if skipped:
        lines.extend(["", "## Skipped Clips", ""])
        lines.extend(f"- `{item['path']}`: {item['reason']}" for item in skipped)
    stderr_summary = summarize_stderr(command["stderr"])
    if stderr_summary:
        lines.extend(["", "## Stderr Summary", "", "```text", stderr_summary, "```"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/xiph_small_clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/lvc_baseline_raw"))
    parser.add_argument("--qualities", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--csv-output", type=Path, default=Path("reports/ssf2020_rd_points.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/lvc_baseline.md"))
    parser.add_argument("--plot-output", type=Path, default=Path("reports/ssf2020_rd_curve.png"))
    parser.add_argument("--eval-input-dir", type=Path, default=Path("data/tmp/lvc_baseline_input"))
    parser.add_argument("--min-side", type=int, default=161)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="estimate bitrate from likelihoods instead of writing entropy-coded bitstreams",
    )
    args = parser.parse_args()

    aggregate_name = "ssf2020_rd_points_raw"
    eval_input, skipped = prepare_eval_input(args.input, args.eval_input_dir, args.min_side)
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    command = [
        sys.executable,
        "-m",
        "compressai.utils.video.eval_model",
        "pretrained",
        str(eval_input),
        str(args.output_dir),
        "-a",
        "ssf2020",
        "-q",
        ",".join(str(q) for q in args.qualities),
        "-d",
        str(args.output_dir),
        "-o",
        aggregate_name,
    ]
    if args.force:
        command.append("--force")
    if args.entropy_estimation:
        command.append("--entropy-estimation")
    if args.half:
        command.append("--half")
    if not args.cpu and torch.cuda.is_available():
        command.append("--cuda")

    result = run_command(command)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command_stdout.txt").write_text(result["stdout"], encoding="utf-8")
    (args.output_dir / "command_stderr.txt").write_text(result["stderr"], encoding="utf-8")
    if result["returncode"] != 0:
        write_report(
            [],
            result,
            args.report_output,
            args.csv_output,
            args.output_dir / f"{aggregate_name}.json",
            args.plot_output,
            args.entropy_estimation,
            skipped,
        )
        print(result["stderr"], file=sys.stderr)
        return result["returncode"]

    aggregate_path = args.output_dir / f"{aggregate_name}.json"
    rows = collect_aggregate_rows(aggregate_path) + collect_sequence_rows(args.output_dir, aggregate_path)
    write_csv(rows, args.csv_output)
    write_plot(rows, args.plot_output)
    write_report(
        rows,
        result,
        args.report_output,
        args.csv_output,
        aggregate_path,
        args.plot_output,
        args.entropy_estimation,
        skipped,
    )
    print(json.dumps({"ok": True, "rows": len(rows), "qualities": args.qualities}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
