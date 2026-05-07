#!/usr/bin/env python3
"""Run the offline ProGVC sender/protocol/receiver loopback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transport.offline_loopback import OfflineLoopbackConfig, run_offline_loopback, write_loopback_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/xiph_small_clips"))
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--playback-delay-ms", type=float, default=133.0)
    parser.add_argument("--packet-loss-rate", type=float, default=0.0)
    parser.add_argument("--max-payload-size", type=int, default=300)
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/generator/cnn_gan.pth"))
    parser.add_argument("--context-checkpoint", type=Path, default=Path("checkpoints/context_model/context_model.pth"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("reports/offline_loopback_metrics.csv"))
    parser.add_argument("--metrics-json", type=Path, default=Path("reports/offline_loopback_metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/offline_loopback_report.md"))
    parser.add_argument("--video-output", type=Path, default=Path("reports/videos/offline_loopback/reconstruction.mp4"))
    parser.add_argument("--sample-output", type=Path, default=Path("reports/offline_loopback_samples.png"))
    args = parser.parse_args()

    summary = run_offline_loopback(
        OfflineLoopbackConfig(
            input_path=args.input,
            frames=args.frames,
            resize=args.resize,
            playback_delay_ms=args.playback_delay_ms,
            max_payload_size=args.max_payload_size,
            packet_loss_rate=args.packet_loss_rate,
            generator_checkpoint=args.generator_checkpoint,
            context_checkpoint=args.context_checkpoint,
            video_output=args.video_output,
            sample_output=args.sample_output,
        )
    )
    write_loopback_outputs(summary, args.metrics_csv, args.metrics_json, args.report)
    print(json.dumps({"ok": True, "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
