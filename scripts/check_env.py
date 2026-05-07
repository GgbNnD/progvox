#!/usr/bin/env python3
"""Verify the phase-one ProGVC development environment.

The script is intentionally dependency-light except for the packages it checks.
Run it inside the shared conda environment:

    conda run -n alg python scripts/check_env.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "torchaudio",
    "compressai",
    "cv2",
    "av",
    "numpy",
    "pandas",
    "matplotlib",
    "skimage",
    "pytorch_msssim",
    "lpips",
    "bjontegaard",
    "yaml",
    "tqdm",
    "aiortc",
    "websockets",
]


def run_command(args: list[str]) -> dict[str, Any]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False, "error": f"{args[0]} not found on PATH"}

    completed = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def module_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostics path
        return {"ok": False, "error": repr(exc)}

    version = getattr(module, "__version__", None)
    return {"ok": True, "version": version}


def torch_status() -> dict[str, Any]:
    status = module_status("torch")
    if not status["ok"]:
        return status

    import torch

    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024**2)),
                    "capability": f"{props.major}.{props.minor}",
                }
            )

    return {
        **status,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def build_report() -> dict[str, Any]:
    modules = {name: module_status(name) for name in REQUIRED_MODULES}
    modules["torch"] = torch_status()

    ffmpeg = run_command(["ffmpeg", "-version"])
    nvidia_smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    gpu_names = [device["name"] for device in modules["torch"].get("devices", [])]
    checks = {
        "python_3_10": sys.version_info[:2] == (3, 10),
        "cuda_available": bool(modules["torch"].get("cuda_available")),
        "rtx_4060_visible": any("RTX 4060" in name for name in gpu_names),
        "compressai_imports": modules["compressai"]["ok"],
        "ffmpeg_available": ffmpeg.get("available") and ffmpeg.get("returncode") == 0,
        "bdrate_library_available": modules["bjontegaard"]["ok"],
        "webrtc_stack_available": modules["aiortc"]["ok"] and modules["websockets"]["ok"] and modules["av"]["ok"],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "system": platform.platform(),
        },
        "modules": modules,
        "commands": {
            "ffmpeg": ffmpeg,
            "nvidia_smi": nvidia_smi,
        },
        "checks": checks,
        "ok": all(checks.values()),
        "notes": [
            "python-bdrate was unavailable for this Python/PyPI combination; bjontegaard==1.3.0 is used instead.",
            "OpenCV is pinned to 4.10.0.84 because CompressAI 1.2.8 requires numpy<2.0.",
            "aiortc==1.14.0 pins av<17, so av==16.1.0 is recorded for WebRTC loopback work.",
        ],
    }


def write_text_summary(report: dict[str, Any], path: Path) -> None:
    torch_info = report["modules"]["torch"]
    ffmpeg_stdout = report["commands"]["ffmpeg"].get("stdout", "")
    ffmpeg_first_line = ffmpeg_stdout.splitlines()[0] if ffmpeg_stdout else "not available"

    lines = [
        "# Environment Check",
        "",
        f"- Overall status: {'PASS' if report['ok'] else 'FAIL'}",
        f"- Python: {report['platform']['python'].split()[0]}",
        f"- Torch: {torch_info.get('version')} / CUDA available: {torch_info.get('cuda_available')}",
        f"- CUDA runtime reported by torch: {torch_info.get('cuda_version')}",
        f"- GPU devices: {', '.join(device['name'] for device in torch_info.get('devices', [])) or 'none'}",
        f"- CompressAI: {report['modules']['compressai'].get('version')}",
        f"- OpenCV: {report['modules']['cv2'].get('version')}",
        f"- PyAV: {report['modules']['av'].get('version')}",
        f"- NumPy: {report['modules']['numpy'].get('version')}",
        f"- BD-Rate helper: bjontegaard {report['modules']['bjontegaard'].get('version')}",
        f"- ffmpeg: {ffmpeg_first_line}",
        f"- aiortc: {report['modules']['aiortc'].get('version')}",
        f"- websockets: {report['modules']['websockets'].get('version')}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- [{'x' if value else ' '}] {name}" for name, value in report["checks"].items())
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=Path("reports/env_check.json"))
    parser.add_argument("--text-output", type=Path, default=Path("reports/env_check.txt"))
    args = parser.parse_args()

    report = build_report()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_summary(report, args.text_output)

    print(json.dumps({"ok": report["ok"], "checks": report["checks"]}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
