#!/usr/bin/env python3
"""Compare layered token schedulers on a deterministic weak-network trace."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transport.scheduler import (  # noqa: E402
    GreedyScheduler,
    SlidingWindowScheduler,
    SimulationResult,
    default_video_call_layers,
    make_bursty_trace,
    simulate,
)


def scalar_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "layer_counts"}


def write_summary_csv(results: list[SimulationResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [scalar_summary(result.summary()) for result in results]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_frame_csv(results: list[SimulationResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "frame_id", "deadline_ms", "max_layer", "quality_psnr", "delivered_bits", "stalled"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for frame in result.frames:
                writer.writerow(
                    {
                        "policy": result.policy,
                        "frame_id": frame.frame_id,
                        "deadline_ms": f"{frame.deadline_ms:.2f}",
                        "max_layer": frame.max_layer,
                        "quality_psnr": f"{frame.quality_psnr:.3f}",
                        "delivered_bits": frame.delivered_bits,
                        "stalled": int(frame.stalled),
                    }
                )


def write_json(results: list[SimulationResult], trace: list[Any], output: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace": [
            {
                "slot": index,
                "bandwidth_kbps": point.bandwidth_kbps,
                "loss_rate": point.loss_rate,
                "duration_ms": point.duration_ms,
                "raw_capacity_bits": point.raw_capacity_bits,
                "effective_capacity_bits": point.effective_capacity_bits,
            }
            for index, point in enumerate(trace)
        ],
        "layer_specs": [
            {
                "level": spec.level,
                "bits": spec.bits,
                "quality_psnr": spec.quality_psnr,
                "label": spec.label,
            }
            for spec in results[0].layer_specs
        ],
        "results": [
            {
                "summary": result.summary(),
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "deadline_ms": frame.deadline_ms,
                        "max_layer": frame.max_layer,
                        "quality_psnr": frame.quality_psnr,
                        "delivered_bits": frame.delivered_bits,
                        "stalled": frame.stalled,
                    }
                    for frame in result.frames
                ],
            }
            for result in results
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_plot(results: list[SimulationResult], trace: list[Any], output: Path, fps: float) -> None:
    frame_times = [frame.frame_id / fps for frame in results[0].frames]
    trace_times = [index / fps for index in range(len(trace))]
    bandwidth = [point.bandwidth_kbps for point in trace]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(trace_times, bandwidth, color="#2f6f9f", linewidth=1.8)
    axes[0].set_ylabel("Bandwidth kbps")
    axes[0].grid(True, alpha=0.25)

    colors = {"greedy": "#b34747", "sliding_window": "#2c8c59"}
    for result in results:
        qualities = [frame.quality_psnr for frame in result.frames]
        axes[1].plot(frame_times, qualities, label=result.policy, color=colors.get(result.policy), linewidth=1.6)
        stalled_times = [frame.frame_id / fps for frame in result.frames if frame.stalled]
        if stalled_times:
            axes[1].scatter(stalled_times, [0.0] * len(stalled_times), color=colors.get(result.policy), s=14)
    axes[1].set_ylabel("Render PSNR proxy")
    axes[1].set_xlabel("Time seconds")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_report(
    results: list[SimulationResult],
    report: Path,
    summary_csv: Path,
    frames_csv: Path,
    metrics_json: Path,
    plot: Path,
    command: str,
    fps: float,
    playback_delay_ms: float,
) -> None:
    layer_lines = []
    for spec in results[0].layer_specs:
        layer_lines.append(f"- L{spec.level}: {spec.bits} bits/frame, PSNR proxy {spec.quality_psnr:.2f} dB")

    lines = [
        "# Scheduler Comparison",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Command: `{command}`",
        f"- Source FPS: {fps:.2f}",
        f"- Playback delay: {playback_delay_ms:.1f} ms",
        f"- Summary CSV: `{summary_csv}`",
        f"- Per-frame CSV: `{frames_csv}`",
        f"- Metrics JSON: `{metrics_json}`",
        f"- Trace plot: `{plot}`",
        "",
        "## Token Layer Model",
        "",
        *layer_lines,
        "",
        "## Results",
        "",
        "| policy | stall rate | decoded PSNR mean | render PSNR mean | PSNR fluctuation | PSNR std | sent kbps | avg layer | utilization | layer counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        summary = result.summary()
        layer_counts = ", ".join(f"{level}:{count}" for level, count in summary["layer_counts"].items())
        lines.append(
            "| {policy} | {stall} | {decoded:.2f} | {render:.2f} | {fluct:.2f} | {std:.2f} | {kbps:.1f} | {avg_layer:.2f} | {util} | {counts} |".format(
                policy=summary["policy"],
                stall=format_percent(float(summary["stall_rate"])),
                decoded=float(summary["decoded_psnr_mean"]),
                render=float(summary["render_psnr_mean"]),
                fluct=float(summary["psnr_fluctuation"]),
                std=float(summary["psnr_std"]),
                kbps=float(summary["sent_bitrate_kbps"]),
                avg_layer=float(summary["average_layer"]),
                util=format_percent(float(summary["network_utilization"])),
                counts=layer_counts,
            )
        )

    greedy = next(result.summary() for result in results if result.policy == "greedy")
    sliding = next(result.summary() for result in results if result.policy == "sliding_window")
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "- Greedy scheduling spends early bandwidth finishing enhancement layers for the oldest frame, which increases frame-to-frame quality swings when the trace drops.",
            "- Sliding-window scheduling first spreads lower layers across frames inside the playback window, reducing deadline misses under the same trace.",
            f"- In this run, sliding-window stall rate changed from {format_percent(float(greedy['stall_rate']))} to {format_percent(float(sliding['stall_rate']))}, and PSNR fluctuation changed from {float(greedy['psnr_fluctuation']):.2f} to {float(sliding['psnr_fluctuation']):.2f}.",
            "",
            "## Failure Notes",
            "",
            "- This is a trace-level simulator, not a packet-level WebRTC implementation yet.",
            "- The PSNR values are layer quality proxies from the current 64x64 tokenizer/generator experiments; the next step is to connect these decisions to actual reconstructed video frames.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--playback-delay-ms", type=float, default=133.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    interval_ms = 1000.0 / args.fps
    drain_slots = int(args.playback_delay_ms / interval_ms) + 3
    trace = make_bursty_trace(args.frames + drain_slots, duration_ms=interval_ms, seed=args.seed)
    layer_specs = default_video_call_layers()
    results = [
        simulate(GreedyScheduler(), trace, layer_specs, args.frames, args.fps, args.playback_delay_ms),
        simulate(SlidingWindowScheduler(window_frames=5), trace, layer_specs, args.frames, args.fps, args.playback_delay_ms),
    ]

    summary_csv = args.reports_dir / "scheduler_comparison.csv"
    frames_csv = args.reports_dir / "scheduler_frames.csv"
    metrics_json = args.reports_dir / "scheduler_comparison.json"
    plot = args.reports_dir / "scheduler_trace.png"
    report = args.docs_dir / "scheduler_comparison.md"

    write_summary_csv(results, summary_csv)
    write_frame_csv(results, frames_csv)
    write_json(results, trace, metrics_json)
    write_plot(results, trace, plot, args.fps)
    write_report(
        results,
        report,
        summary_csv,
        frames_csv,
        metrics_json,
        plot,
        " ".join(shlex.quote(part) for part in ["python", *sys.argv]),
        args.fps,
        args.playback_delay_ms,
    )

    print(json.dumps({"ok": True, "summaries": [result.summary() for result in results]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
