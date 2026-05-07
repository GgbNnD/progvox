#!/usr/bin/env python3
"""Collect video counts, dimensions, frame counts, and duration summaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".y4m", ".yuv", ".mp4", ".mkv", ".mov", ".avi", ".webm"}


RAW_NAME_RE = re.compile(
    r"(?P<width>\d+)x(?P<height>\d+)_(?P<fps>[\d.]+)fps_(?P<bitdepth>\d+)bit_(?P<format>P420|I420|yuv420p)",
    re.IGNORECASE,
)


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


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    return float(Fraction(value))


def probe_with_ffprobe(path: Path) -> dict[str, Any]:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,pix_fmt",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = data["streams"][0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate")) or 0.0
    frames = int(stream["nb_frames"]) if stream.get("nb_frames") else None
    duration = float(stream["duration"]) if stream.get("duration") else None
    if frames is None and duration is not None and fps:
        frames = round(duration * fps)
    if duration is None and frames is not None and fps:
        duration = frames / fps
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration,
        "pix_fmt": stream.get("pix_fmt"),
    }


def probe_raw_yuv(path: Path) -> dict[str, Any]:
    match = RAW_NAME_RE.search(path.name)
    if not match:
        raise RuntimeError("raw .yuv filename must include WIDTHxHEIGHT_FPSfps_8bit_P420")
    width = int(match.group("width"))
    height = int(match.group("height"))
    fps = float(match.group("fps"))
    bitdepth = int(match.group("bitdepth"))
    if bitdepth != 8:
        raise RuntimeError("only 8-bit yuv420 raw clips are supported in phase one")
    frame_size = width * height * 3 // 2
    frames = path.stat().st_size // frame_size
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps if fps else None,
        "pix_fmt": "yuv420p",
    }


def collect_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def collect_stats(input_path: Path) -> dict[str, Any]:
    files = collect_files(input_path)
    records = []
    for path in files:
        try:
            stats = probe_raw_yuv(path) if path.suffix.lower() == ".yuv" else probe_with_ffprobe(path)
            status = "ok"
            error = None
        except Exception as exc:  # pragma: no cover - diagnostic path
            stats = {}
            status = "error"
            error = str(exc)
        records.append(
            {
                "path": str(path),
                "name": path.name,
                "bytes": path.stat().st_size,
                "status": status,
                "error": error,
                **stats,
            }
        )

    ok_records = [record for record in records if record["status"] == "ok"]
    resolution_counts = Counter(f"{record['width']}x{record['height']}" for record in ok_records)
    total_duration = sum(record.get("duration_seconds") or 0.0 for record in ok_records)
    total_frames = sum(record.get("frames") or 0 for record in ok_records)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "video_count": len(ok_records),
        "error_count": len(records) - len(ok_records),
        "total_duration_seconds": round(total_duration, 3),
        "total_frames": total_frames,
        "resolution_distribution": dict(sorted(resolution_counts.items())),
        "records": records,
        "ok": len(ok_records) == len(records) and bool(records),
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Dataset Readme",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Input: `{report['input']}`",
        f"- Valid videos: {report['video_count']}",
        f"- Errors: {report['error_count']}",
        f"- Total frames: {report['total_frames']}",
        f"- Total duration: {report['total_duration_seconds']:.3f}s",
        "",
        "## Resolution Distribution",
        "",
    ]
    for resolution, count in report["resolution_distribution"].items():
        lines.append(f"- `{resolution}`: {count}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| file | resolution | fps | frames | duration | status |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for record in report["records"]:
        resolution = (
            f"{record.get('width')}x{record.get('height')}"
            if record["status"] == "ok"
            else "-"
        )
        fps = f"{record.get('fps', 0):.3f}" if record["status"] == "ok" else "-"
        frames = record.get("frames", "-") if record["status"] == "ok" else "-"
        duration = (
            f"{record.get('duration_seconds') or 0:.3f}"
            if record["status"] == "ok"
            else "-"
        )
        lines.append(f"| `{record['name']}` | {resolution} | {fps} | {frames} | {duration} | {record['status']} |")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/xiph_small"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_stats.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/dataset_readme.md"))
    args = parser.parse_args()

    report = collect_stats(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.markdown_output)
    print(json.dumps({"ok": report["ok"], "videos": report["video_count"]}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
