#!/usr/bin/env python3
"""Download the small Xiph sample dataset declared in a YAML manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


def read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stream_download(
    url: str,
    output: Path,
    expected_bytes: int = 0,
    chunk_size: int = 1024 * 1024,
    force: bool = False,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    if force:
        output.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)

    resume_from = tmp.stat().st_size if tmp.exists() else 0
    digest = hashlib.sha256()
    bytes_written = resume_from
    start = time.perf_counter()

    headers = {"User-Agent": "progvox-dataset-prep/0.1"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        if resume_from and response.status != 206:
            resume_from = 0
            bytes_written = 0
            mode = "wb"
        else:
            mode = "ab" if resume_from else "wb"
        with tmp.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)

    if expected_bytes and bytes_written != expected_bytes:
        raise RuntimeError(
            f"{output.name}: expected {expected_bytes} bytes, got {bytes_written}"
        )
    tmp.replace(output)
    digest_value = file_sha256(output)
    elapsed = time.perf_counter() - start
    return {
        "path": str(output),
        "bytes": bytes_written,
        "sha256": digest_value,
        "elapsed_seconds": round(elapsed, 3),
        "resumed_from": resume_from,
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/xiph_small.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/xiph_small"))
    parser.add_argument("--report", type=Path, default=Path("reports/download_sample_data.json"))
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--workers", type=int, default=3, help="parallel download workers")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    def fetch(video: dict) -> dict:
        target = args.output_root / video["filename"]
        expected_bytes = int(video.get("expected_bytes") or 0)
        if target.exists() and not args.force:
            size = target.stat().st_size
            status = "present" if not expected_bytes or size == expected_bytes else "size_mismatch"
            return {
                "id": video["id"],
                "status": status,
                "path": str(target),
                "bytes": size,
                "sha256": file_sha256(target) if status == "present" else None,
            }

        try:
            result = stream_download(
                video["url"],
                target,
                expected_bytes=expected_bytes,
                force=args.force,
            )
            return {"id": video["id"], "status": "downloaded", **result}
        except Exception as exc:  # pragma: no cover - network diagnostic path
            return {"id": video["id"], "status": "error", "path": str(target), "error": str(exc)}

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(fetch, video): video for video in manifest["videos"]}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"{record['id']}: {record['status']}")

    records.sort(key=lambda record: [video["id"] for video in manifest["videos"]].index(record["id"]))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest["dataset"],
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "records": records,
        "ok": all(record["status"] in {"present", "downloaded"} for record in records),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "files": len(records)}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
