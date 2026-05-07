#!/usr/bin/env python3
"""Run trace-driven ABR decisions and capped scheduler simulation."""

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

from models.abr_controller import (  # noqa: E402
    ABRDecision,
    NetworkObservation,
    RuleBasedABRConfig,
    RuleBasedABRController,
    observations_from_trace,
)
from transport.scheduler import SlidingWindowScheduler, default_video_call_layers, make_bursty_trace, simulate  # noqa: E402


def layer_distribution(decisions: list[ABRDecision]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for decision in decisions:
        counts[decision.selected_level] = counts.get(decision.selected_level, 0) + 1
    return dict(sorted(counts.items()))


def write_decisions_csv(
    decisions: list[ABRDecision],
    observations: list[NetworkObservation],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "timestamp_ms",
        "throughput_kbps",
        "estimated_throughput_kbps",
        "rtt_ms",
        "loss_rate",
        "queue_delay_ms",
        "selected_level",
        "budget_bits_per_frame",
        "target_bits_per_frame",
        "quality_psnr",
        "reason",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for decision, observation in zip(decisions, observations):
            writer.writerow(
                {
                    "frame_id": decision.frame_id,
                    "timestamp_ms": f"{decision.timestamp_ms:.2f}",
                    "throughput_kbps": f"{observation.throughput_kbps:.3f}",
                    "estimated_throughput_kbps": f"{decision.estimated_throughput_kbps:.3f}",
                    "rtt_ms": f"{observation.rtt_ms:.3f}",
                    "loss_rate": f"{observation.loss_rate:.5f}",
                    "queue_delay_ms": f"{observation.queue_delay_ms:.3f}",
                    "selected_level": decision.selected_level,
                    "budget_bits_per_frame": f"{decision.budget_bits_per_frame:.2f}",
                    "target_bits_per_frame": decision.target_bits_per_frame,
                    "quality_psnr": f"{decision.quality_psnr:.3f}",
                    "reason": decision.reason,
                }
            )


def write_summary_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    decisions: list[ABRDecision],
    observations: list[NetworkObservation],
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary_rows,
        "decision_layer_distribution": layer_distribution(decisions),
        "decisions": [
            {
                "frame_id": decision.frame_id,
                "timestamp_ms": decision.timestamp_ms,
                "throughput_kbps": observation.throughput_kbps,
                "estimated_throughput_kbps": decision.estimated_throughput_kbps,
                "rtt_ms": observation.rtt_ms,
                "loss_rate": observation.loss_rate,
                "queue_delay_ms": observation.queue_delay_ms,
                "selected_level": decision.selected_level,
                "budget_bits_per_frame": decision.budget_bits_per_frame,
                "target_bits_per_frame": decision.target_bits_per_frame,
                "quality_psnr": decision.quality_psnr,
                "reason": decision.reason,
            }
            for decision, observation in zip(decisions, observations)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_plot(
    decisions: list[ABRDecision],
    observations: list[NetworkObservation],
    output: Path,
    fps: float,
) -> None:
    times = [decision.frame_id / fps for decision in decisions]
    throughput = [observation.throughput_kbps for observation in observations]
    estimated = [decision.estimated_throughput_kbps for decision in decisions]
    rtt = [observation.rtt_ms for observation in observations]
    levels = [decision.selected_level for decision in decisions]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, throughput, color="#2f6f9f", linewidth=1.4, label="instant")
    axes[0].plot(times, estimated, color="#8a4f9f", linewidth=1.4, label="ewma")
    axes[0].set_ylabel("kbps")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, rtt, color="#b35b2d", linewidth=1.4)
    axes[1].set_ylabel("RTT ms")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(times, levels, where="post", color="#2c8c59", linewidth=1.8)
    axes[2].set_ylabel("max layer")
    axes[2].set_xlabel("Time seconds")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_notebook(output: Path, decisions_csv: Path, summary_csv: Path, plot: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ABR Trace-Driven Analysis\n",
                    "\n",
                    "This notebook is generated by `scripts/simulate_abr.py` and records the phase 3.2 ABR deliverable.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    f"decisions = pd.read_csv('{decisions_csv.as_posix()}')\n",
                    f"summary = pd.read_csv('{summary_csv.as_posix()}')\n",
                    "display(summary)\n",
                    "display(decisions.head())\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "from IPython.display import Image, display\n",
                    f"display(Image(filename='{plot.as_posix()}'))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "The ABR controller uses throughput EWMA for the primary layer choice, then applies RTT/loss penalties and upshift hysteresis. The generated CSV contains the per-frame reason field for debugging.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_report(
    decisions: list[ABRDecision],
    summary_rows: list[dict[str, Any]],
    output: Path,
    decisions_csv: Path,
    summary_csv: Path,
    metrics_json: Path,
    plot: Path,
    notebook: Path,
    command: str,
) -> None:
    distribution = layer_distribution(decisions)
    lines = [
        "# ABR Analysis",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Command: `{command}`",
        f"- Decisions CSV: `{decisions_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        f"- Metrics JSON: `{metrics_json}`",
        f"- Decision curve: `{plot}`",
        f"- Notebook: `{notebook}`",
        "",
        "## Decision Distribution",
        "",
        "| max layer | frames |",
        "| ---: | ---: |",
    ]
    for level, count in distribution.items():
        lines.append(f"| {level} | {count} |")
    lines.extend(
        [
            "",
            "## Scheduler Impact",
            "",
            "| policy | stall rate | render PSNR mean | PSNR fluctuation | PSNR std | sent kbps | avg layer |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {policy} | {stall} | {render:.2f} | {fluct:.2f} | {std:.2f} | {kbps:.1f} | {avg_layer:.2f} |".format(
                policy=row["policy"],
                stall=format_percent(float(row["stall_rate"])),
                render=float(row["render_psnr_mean"]),
                fluct=float(row["psnr_fluctuation"]),
                std=float(row["psnr_std"]),
                kbps=float(row["sent_bitrate_kbps"]),
                avg_layer=float(row["average_layer"]),
            )
        )
    uncapped = summary_rows[0]
    capped = summary_rows[1]
    bitrate_reduction = 1.0 - float(capped["sent_bitrate_kbps"]) / max(float(uncapped["sent_bitrate_kbps"]), 1e-9)
    render_delta = float(capped["render_psnr_mean"]) - float(uncapped["render_psnr_mean"])
    fluctuation_delta = float(capped["psnr_fluctuation"]) - float(uncapped["psnr_fluctuation"])
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "- The controller lowers the maximum transmitted layer during low-throughput and high-delay windows, then requires several stable observations before upshifting.",
            f"- Compared with uncapped sliding-window scheduling, ABR-capped scheduling changes stall rate from {format_percent(float(uncapped['stall_rate']))} to {format_percent(float(capped['stall_rate']))}.",
            f"- ABR sent bitrate is {float(capped['sent_bitrate_kbps']):.1f} kbps, a {format_percent(bitrate_reduction)} reduction from {float(uncapped['sent_bitrate_kbps']):.1f} kbps for the uncapped scheduler.",
            f"- The cost in this conservative rule set is {render_delta:.2f} dB render-PSNR proxy and {fluctuation_delta:+.2f} PSNR fluctuation on the same trace.",
            "",
            "## Failure Notes",
            "",
            "- This is a rule-based controller; it does not yet use learned QoE optimization.",
            "- RTT values are derived from the deterministic weak-network trace for repeatable offline tests. Live WebRTC RTT/loss will replace this source in phase 4.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--playback-delay-ms", type=float, default=133.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--notebook", type=Path, default=Path("analysis_abr.ipynb"))
    args = parser.parse_args()

    interval_ms = 1000.0 / args.fps
    drain_slots = int(args.playback_delay_ms / interval_ms) + 3
    trace = make_bursty_trace(args.frames + drain_slots, duration_ms=interval_ms, seed=args.seed)
    observations = observations_from_trace(trace[: args.frames], fps=args.fps)
    layers = default_video_call_layers()
    controller = RuleBasedABRController(
        layers,
        RuleBasedABRConfig(
            fps=args.fps,
            max_level=len(layers) - 1,
            safety_factor=1.0,
            high_rtt_ms=220.0,
            queue_delay_high_ms=150.0,
            stable_upshift_windows=2,
        ),
    )
    decisions = controller.run(observations)
    frame_caps = [decision.selected_level for decision in decisions]

    uncapped = simulate(
        SlidingWindowScheduler(window_frames=5),
        trace,
        layers,
        args.frames,
        args.fps,
        args.playback_delay_ms,
    )
    capped = simulate(
        SlidingWindowScheduler(window_frames=5, name="sliding_window_abr"),
        trace,
        layers,
        args.frames,
        args.fps,
        args.playback_delay_ms,
        frame_max_levels=frame_caps,
    )
    summary_rows = [
        {key: value for key, value in uncapped.summary().items() if key != "layer_counts"},
        {key: value for key, value in capped.summary().items() if key != "layer_counts"},
    ]

    decisions_csv = args.reports_dir / "abr_decisions.csv"
    summary_csv = args.reports_dir / "abr_simulation_summary.csv"
    metrics_json = args.reports_dir / "abr_analysis.json"
    plot = args.reports_dir / "abr_decision_curve.png"
    report = args.docs_dir / "abr_analysis.md"

    write_decisions_csv(decisions, observations, decisions_csv)
    write_summary_csv(summary_rows, summary_csv)
    write_json(decisions, observations, summary_rows, metrics_json)
    write_plot(decisions, observations, plot, args.fps)
    write_notebook(args.notebook, decisions_csv, summary_csv, plot)
    write_report(
        decisions,
        summary_rows,
        report,
        decisions_csv,
        summary_csv,
        metrics_json,
        plot,
        args.notebook,
        " ".join(shlex.quote(part) for part in ["python", *sys.argv]),
    )

    print(
        json.dumps(
            {
                "ok": True,
                "decision_layer_distribution": layer_distribution(decisions),
                "summary": summary_rows,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
