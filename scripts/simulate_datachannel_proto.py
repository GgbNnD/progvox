#!/usr/bin/env python3
"""Simulate DataChannel token packet loss, reordering and reassembly."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transport.datachannel_proto import FrameReassembler, fragment_payload, pack_packet  # noqa: E402
from transport.scheduler import default_video_call_layers  # noqa: E402


def deterministic_payload(frame_id: int, layer_id: int, size: int) -> bytes:
    return bytes((frame_id * 17 + layer_id * 31 + index) % 256 for index in range(size))


def make_packets(frames: int, fps: float, max_payload_size: int) -> list[dict[str, Any]]:
    interval_ms = 1000.0 / fps
    layer_specs = default_video_call_layers()
    packet_rows = []
    for frame_id in range(frames):
        deadline_ms = int(round((frame_id + 4) * interval_ms))
        send_base_ms = int(round(frame_id * interval_ms))
        for spec in layer_specs:
            payload_size = max(1, (spec.bits + 7) // 8)
            payload = deterministic_payload(frame_id, spec.level, payload_size)
            packets = fragment_payload(frame_id, spec.level, deadline_ms, payload, max_payload_size=max_payload_size)
            for packet in packets:
                packet_rows.append(
                    {
                        "send_ms": send_base_ms + spec.level * 2 + packet.chunk_id,
                        "packet": packet,
                        "raw": pack_packet(packet),
                    }
                )
    return packet_rows


def simulate_loss(
    loss_rate: float,
    frames: int,
    fps: float,
    max_payload_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = make_packets(frames, fps, max_payload_size)
    delivered = []
    for row in rows:
        if rng.random() < loss_rate:
            continue
        jitter_ms = rng.randint(0, 35)
        delivered.append((row["send_ms"] + jitter_ms, row["raw"]))
    delivered.sort(key=lambda item: (item[0], rng.random()))

    reassembler = FrameReassembler(timeout_ms=180)
    completed_layers: dict[int, set[int]] = {}
    for arrival_ms, raw in delivered:
        layer = reassembler.push(raw, now_ms=arrival_ms)
        if layer is not None:
            completed_layers.setdefault(layer.frame_id, set()).add(layer.layer_id)
        reassembler.expire(arrival_ms)
    reassembler.expire(int(round((frames + 8) * 1000.0 / fps)))

    decodable = sum(1 for frame_id in range(frames) if 0 in completed_layers.get(frame_id, set()))
    full = sum(1 for frame_id in range(frames) if completed_layers.get(frame_id, set()) == {0, 1, 2, 3})
    completed_layer_count = sum(len(layers) for layers in completed_layers.values())
    return {
        "loss_rate": loss_rate,
        "frames": frames,
        "packets_sent": len(rows),
        "packets_delivered": len(delivered),
        "packet_delivery_rate": len(delivered) / len(rows),
        "decodable_frames": decodable,
        "decodable_frame_rate": decodable / frames,
        "full_frames": full,
        "full_frame_rate": full / frames,
        "completed_layers": completed_layer_count,
        "completed_layer_rate": completed_layer_count / (frames * 4),
        "expired_assemblies": reassembler.expired_assemblies,
        "duplicate_chunks": reassembler.duplicate_chunks,
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], output: Path, csv_path: Path, json_path: Path, command: str) -> None:
    lines = [
        "# DataChannel Protocol Robustness",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Command: `{command}`",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        "",
        "## Results",
        "",
        "| loss | packet delivery | decodable frames | full frames | completed layers | expired assemblies |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {loss:.1%} | {delivery:.1%} | {decodable:.1%} | {full:.1%} | {layers:.1%} | {expired} |".format(
                loss=float(row["loss_rate"]),
                delivery=float(row["packet_delivery_rate"]),
                decodable=float(row["decodable_frame_rate"]),
                full=float(row["full_frame_rate"]),
                layers=float(row["completed_layer_rate"]),
                expired=int(row["expired_assemblies"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Decodable frames require Layer 0 to reassemble before deadline.",
            "- Full frames require all four token layers to reassemble; this is naturally more sensitive to packet loss because enhancement layers fragment into more chunks.",
            "- The next step is to send these binary packets over the aiortc DataChannel loopback instead of this in-memory packet-loss harness.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-payload-size", type=int, default=300)
    parser.add_argument("--loss-rates", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    rows = [
        simulate_loss(loss, args.frames, args.fps, args.max_payload_size, args.seed + index)
        for index, loss in enumerate(args.loss_rates)
    ]
    csv_path = args.reports_dir / "datachannel_protocol_robustness.csv"
    json_path = args.reports_dir / "datachannel_protocol_robustness.json"
    report = args.docs_dir / "protocol_robustness.md"
    write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        rows,
        report,
        csv_path,
        json_path,
        " ".join(shlex.quote(part) for part in ["python", *sys.argv]),
    )
    print(json.dumps({"ok": True, "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
