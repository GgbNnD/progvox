#!/usr/bin/env python3
"""Run a live aiortc DataChannel loopback for binary token packets."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transport.webrtc_token_loopback import TokenLoopbackConfig, run_token_loopback  # noqa: E402


def write_report(result: dict[str, object], output: Path, metrics_json: Path, command: str) -> None:
    lines = [
        "# WebRTC Token DataChannel Loopback",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Command: `{command}`",
        f"- Metrics JSON: `{metrics_json}`",
        "",
        "## Result",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in result.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This sends real binary ProGVC token packets over an aiortc DataChannel and reassembles them with `FrameReassembler`.",
            "- The test still runs in one process; Janus/browser signaling can replace the in-process offer/answer without changing the packet format.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


async def async_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-payload-size", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    result = await run_token_loopback(
        TokenLoopbackConfig(
            frames=args.frames,
            layers=args.layers,
            max_payload_size=args.max_payload_size,
            timeout_s=args.timeout_s,
        )
    )
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "result": result}
    metrics_json = args.reports_dir / "webrtc_token_loopback.json"
    report = args.docs_dir / "webrtc_token_loopback_report.md"
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(
        result,
        report,
        metrics_json,
        " ".join(shlex.quote(part) for part in ["python", *sys.argv]),
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
