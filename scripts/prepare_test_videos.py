#!/usr/bin/env python3
"""Convert raw sample videos into fixed-length YUV420 clips for CompressAI."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".y4m", ".mp4", ".mkv", ".mov", ".avi", ".webm"}


def run_json(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "command failed")
    return json.loads(completed.stdout)


def run_command(args: list[str]) -> None:
    completed = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "command failed")


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    return float(Fraction(value))


def slugify(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def probe_video(path: Path) -> dict[str, Any]:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = data["streams"][0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate")) or 30.0
    frames = int(stream["nb_frames"]) if stream.get("nb_frames") else None
    duration = float(stream["duration"]) if stream.get("duration") else None
    if frames is None and duration is not None:
        frames = round(duration * fps)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
    }


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def output_name(path: Path, stats: dict[str, Any], fps_label: str) -> str:
    return (
        f"{slugify(path.stem)}_{stats['width']}x{stats['height']}_"
        f"{fps_label}fps_8bit_P420.yuv"
    )


def prepare_clip(path: Path, output: Path, frames: int, force: bool) -> dict[str, Any]:
    stats = probe_video(path)
    available = stats.get("frames")
    if available is not None and available < frames:
        raise RuntimeError(f"{path.name} has only {available} frames, need {frames}")

    fps_label = f"{stats['fps']:.3f}".rstrip("0").rstrip(".")
    target = output / output_name(path, stats, fps_label)
    if target.exists() and not force:
        status = "present"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-frames:v",
                str(frames),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rawvideo",
                str(target),
            ]
        )
        status = "written"

    frame_size = stats["width"] * stats["height"] * 3 // 2
    written_frames = target.stat().st_size // frame_size
    return {
        "source": str(path),
        "output": str(target),
        "status": status,
        "width": stats["width"],
        "height": stats["height"],
        "fps": stats["fps"],
        "frames": written_frames,
        "bytes": target.stat().st_size,
        "ok": written_frames >= frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/xiph_small"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/xiph_small_clips"))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--report", type=Path, default=Path("reports/prepare_test_videos.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = []
    for path in collect_inputs(args.input):
        records.append(prepare_clip(path, args.output, args.frames, args.force))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "requested_frames": args.frames,
        "records": records,
        "ok": bool(records) and all(record["ok"] for record in records),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "clips": len(records)}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
