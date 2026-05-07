"""Offline end-to-end ProGVC loopback over the DataChannel packet protocol."""

from __future__ import annotations

import csv
import json
import random
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from models.abr_controller import RuleBasedABRConfig, RuleBasedABRController, observations_from_trace
from models.context_model import ContextModelConfig, TokenContextTransformer
from models.generator import DetailSynthesisCNN
from models.tokenizer import MultiScaleResidualTokenizer, TokenizerConfig
from run_progvc_codec import (
    choose_input,
    fallback_predict_tokens,
    load_generator,
    psnr,
    read_yuv420_frames,
    ssim,
    write_sample_grid,
    write_video,
)
from transport.datachannel_proto import FrameReassembler, fragment_payload, pack_packet
from transport.scheduler import SlidingWindowScheduler, default_video_call_layers, make_bursty_trace, simulate


TOKEN_PAYLOAD_HEADER = struct.Struct("!HH")


@dataclass(frozen=True)
class OfflineLoopbackConfig:
    """Configuration for the local end-to-end loopback."""

    input_path: Path = Path("data/processed/xiph_small_clips")
    frames: int = 12
    resize: int = 64
    playback_delay_ms: float = 133.0
    max_payload_size: int = 300
    packet_loss_rate: float = 0.0
    seed: int = 17
    generator_checkpoint: Path = Path("checkpoints/generator/cnn_gan.pth")
    context_checkpoint: Path = Path("checkpoints/context_model/context_model.pth")
    video_output: Path = Path("reports/videos/offline_loopback/reconstruction.mp4")
    sample_output: Path = Path("reports/offline_loopback_samples.png")


def token_map_to_payload(token_map: Tensor) -> bytes:
    """Serialize one HxW token map into a compact network payload."""

    if token_map.dim() == 3:
        token_map = token_map.squeeze(0)
    if token_map.dim() != 2:
        raise ValueError("token_map must be HxW or 1xHxW")
    height, width = token_map.shape
    if height > 0xFFFF or width > 0xFFFF:
        raise ValueError("token map is too large for the protocol payload")
    array = token_map.detach().cpu().numpy().astype(">u2", copy=False)
    return TOKEN_PAYLOAD_HEADER.pack(int(height), int(width)) + array.tobytes()


def payload_to_token_map(payload: bytes, device: torch.device | None = None) -> Tensor:
    """Deserialize a token layer payload into a 1xHxW tensor."""

    if len(payload) < TOKEN_PAYLOAD_HEADER.size:
        raise ValueError("token payload is shorter than its header")
    height, width = TOKEN_PAYLOAD_HEADER.unpack(payload[: TOKEN_PAYLOAD_HEADER.size])
    expected = TOKEN_PAYLOAD_HEADER.size + height * width * 2
    if len(payload) != expected:
        raise ValueError("token payload length does not match shape")
    array = np.frombuffer(payload[TOKEN_PAYLOAD_HEADER.size :], dtype=">u2").astype(np.int64)
    tensor = torch.from_numpy(array.reshape(1, height, width)).long()
    return tensor if device is None else tensor.to(device)


def load_context_model(checkpoint: Path, device: torch.device) -> TokenContextTransformer:
    """Load the trained context model when available, otherwise use the default interface."""

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = ContextModelConfig(**payload.get("config", {}))
        model = TokenContextTransformer(config).to(device)
        model.load_state_dict(payload["model"])
    else:
        model = TokenContextTransformer(
            ContextModelConfig(codebook_size=512, levels=4, max_sequence_length=2048, d_model=64, num_layers=1, num_heads=4)
        ).to(device)
    model.eval()
    return model


def _gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _write_metrics_csv(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frames",
        "stall_rate",
        "sent_bitrate_kbps",
        "psnr_rgb",
        "ssim_rgb",
        "avg_selected_layer",
        "avg_delivered_layer",
        "encode_ms_per_frame",
        "packetize_ms_per_frame",
        "reassemble_ms_per_frame",
        "decode_ms_per_frame",
        "generator_ms_per_frame",
        "estimated_e2e_latency_ms",
        "gpu_peak_memory_mb",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: summary[field] for field in fields})


def write_loopback_report(summary: dict[str, Any], output: Path, metrics_csv: Path, metrics_json: Path) -> None:
    lines = [
        "# Offline End-to-End Loopback",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input clip: `{summary['input_clip']}`",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Metrics JSON: `{metrics_json}`",
        f"- Sample image: `{summary['sample_output']}`",
        f"- Reconstruction video: `{summary['video_output']}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in [
        "frames",
        "stall_frames",
        "stall_rate",
        "sent_packets",
        "sent_bytes",
        "sent_bitrate_kbps",
        "psnr_rgb",
        "ssim_rgb",
        "avg_selected_layer",
        "avg_delivered_layer",
        "encode_ms_per_frame",
        "packetize_ms_per_frame",
        "reassemble_ms_per_frame",
        "decode_ms_per_frame",
        "generator_ms_per_frame",
        "estimated_e2e_latency_ms",
        "gpu_peak_memory_mb",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Sender side: tokenizer encodes frames, ABR selects maximum token layer, scheduler simulation determines delivered layer depth, and token maps are fragmented into DataChannel packets.",
            "- Receiver side: packets are reassembled by frame/layer, missing layers use the context-model fallback path, and the trained CNN-GAN generator reconstructs frames.",
            "- This is still an offline loopback. The next step is to move the same packet payloads through the live aiortc DataChannel.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run_offline_loopback(config: OfflineLoopbackConfig | None = None) -> dict[str, Any]:
    """Run the offline sender/protocol/receiver/generator path and return metrics."""

    cfg = config or OfflineLoopbackConfig()
    random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    input_clip = choose_input(cfg.input_path)
    target, fps = read_yuv420_frames(input_clip, cfg.frames, cfg.resize, device)
    frame_count = int(target.size(0))
    interval_ms = 1000.0 / fps

    tokenizer = MultiScaleResidualTokenizer(TokenizerConfig(levels=4, codebook_size=512, max_token_resolution=32)).to(device)
    context_model = load_context_model(cfg.context_checkpoint, device)
    generator: DetailSynthesisCNN = load_generator(cfg.generator_checkpoint, device)

    trace = make_bursty_trace(frame_count + int(cfg.playback_delay_ms / interval_ms) + 3, interval_ms, cfg.seed)
    observations = observations_from_trace(trace[:frame_count], fps=fps)
    layers = default_video_call_layers()
    abr = RuleBasedABRController(
        layers,
        RuleBasedABRConfig(
            fps=fps,
            max_level=len(layers) - 1,
            safety_factor=1.0,
            high_rtt_ms=220.0,
            queue_delay_high_ms=150.0,
            stable_upshift_windows=2,
        ),
    )
    decisions = abr.run(observations)
    frame_caps = [decision.selected_level for decision in decisions]
    scheduler_result = simulate(
        SlidingWindowScheduler(window_frames=5, name="offline_loopback_scheduler"),
        trace,
        layers,
        frame_count,
        fps,
        cfg.playback_delay_ms,
        frame_max_levels=frame_caps,
    )
    delivered_max_layers = [frame.max_layer for frame in scheduler_result.frames]

    encode_start = time.perf_counter()
    with torch.no_grad():
        tokenized = tokenizer(target)
    encode_s = time.perf_counter() - encode_start

    packetize_s = 0.0
    reassemble_s = 0.0
    decode_s = 0.0
    generator_s = 0.0
    sent_packets = 0
    sent_bytes = 0
    dropped_packets = 0
    completed_layers: dict[int, dict[int, Tensor]] = {frame_id: {} for frame_id in range(frame_count)}
    reassembler = FrameReassembler(timeout_ms=int(cfg.playback_delay_ms + 80))

    for frame_id, max_layer in enumerate(delivered_max_layers):
        if max_layer < 0:
            continue
        deadline_ms = int(round((frame_id * interval_ms) + cfg.playback_delay_ms))
        for layer_id in range(max_layer + 1):
            start = time.perf_counter()
            payload = token_map_to_payload(tokenized.tokens[layer_id][frame_id])
            packets = fragment_payload(frame_id, layer_id, deadline_ms, payload, max_payload_size=cfg.max_payload_size)
            raw_packets = [pack_packet(packet) for packet in packets]
            packetize_s += time.perf_counter() - start
            for chunk_offset, raw in enumerate(raw_packets):
                sent_packets += 1
                sent_bytes += len(raw)
                if rng.random() < cfg.packet_loss_rate:
                    dropped_packets += 1
                    continue
                arrival_ms = int(round(frame_id * interval_ms + layer_id * 2 + chunk_offset))
                start = time.perf_counter()
                completed = reassembler.push(raw, now_ms=arrival_ms)
                reassembler.expire(arrival_ms)
                reassemble_s += time.perf_counter() - start
                if completed is not None:
                    completed_layers[completed.frame_id][completed.layer_id] = payload_to_token_map(completed.payload, device)
    reassembler.expire(int(round((frame_count + 8) * interval_ms)))

    zero_ids = tokenizer.zero_token_ids()
    decoded_frames = []
    condition_frames = []
    stall_frames = 0
    with torch.no_grad():
        for frame_id in range(frame_count):
            layers_for_frame = completed_layers[frame_id]
            contiguous = []
            for layer_id in range(len(tokenized.shapes)):
                if layer_id not in layers_for_frame:
                    break
                contiguous.append(layers_for_frame[layer_id])
            if not contiguous:
                stall_frames += 1
                condition = torch.zeros((1, 3, cfg.resize, cfg.resize), device=device)
                generated = condition
            else:
                start = time.perf_counter()
                filled = fallback_predict_tokens(context_model, contiguous, tokenized.shapes, zero_ids)
                condition = tokenizer.reconstruct(filled, output_shape=(cfg.resize, cfg.resize), max_level=len(filled) - 1)
                decode_s += time.perf_counter() - start
                start = time.perf_counter()
                generated = generator(condition)
                generator_s += time.perf_counter() - start
            condition_frames.append(condition)
            decoded_frames.append(generated)

    output = torch.cat(decoded_frames, dim=0).clamp(0, 1)
    conditions = torch.cat(condition_frames, dim=0).clamp(0, 1)
    cfg.video_output.parent.mkdir(parents=True, exist_ok=True)
    write_video(output, cfg.video_output, fps)
    write_sample_grid(
        target.detach().cpu(),
        {
            "condition": conditions.detach().cpu(),
            "generated": output.detach().cpu(),
        },
        cfg.sample_output,
    )

    duration_s = frame_count / fps
    delivered_non_stall = [level for level in delivered_max_layers if level >= 0]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_clip": str(input_clip),
        "frames": frame_count,
        "fps": fps,
        "device": str(device),
        "stall_frames": stall_frames,
        "stall_rate": stall_frames / frame_count,
        "sent_packets": sent_packets,
        "dropped_packets": dropped_packets,
        "sent_bytes": sent_bytes,
        "sent_bitrate_kbps": sent_bytes * 8 / max(duration_s, 1e-9) / 1000.0,
        "psnr_rgb": psnr(output, target),
        "ssim_rgb": ssim(output, target),
        "avg_selected_layer": sum(frame_caps) / len(frame_caps),
        "avg_delivered_layer": sum(delivered_non_stall) / len(delivered_non_stall) if delivered_non_stall else -1.0,
        "completed_layers": sum(len(value) for value in completed_layers.values()),
        "expired_assemblies": reassembler.expired_assemblies,
        "duplicate_chunks": reassembler.duplicate_chunks,
        "encode_ms_per_frame": encode_s * 1000.0 / frame_count,
        "packetize_ms_per_frame": packetize_s * 1000.0 / frame_count,
        "reassemble_ms_per_frame": reassemble_s * 1000.0 / frame_count,
        "decode_ms_per_frame": decode_s * 1000.0 / frame_count,
        "generator_ms_per_frame": generator_s * 1000.0 / frame_count,
        "estimated_e2e_latency_ms": cfg.playback_delay_ms + (decode_s + generator_s) * 1000.0 / frame_count,
        "gpu_peak_memory_mb": _gpu_memory_mb(device),
        "video_output": str(cfg.video_output),
        "sample_output": str(cfg.sample_output),
        "selected_layers": frame_caps,
        "delivered_layers": delivered_max_layers,
    }
    return summary


def write_loopback_outputs(summary: dict[str, Any], metrics_csv: Path, metrics_json: Path, report: Path) -> None:
    _write_metrics_csv(summary, metrics_csv)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_loopback_report(summary, report, metrics_csv, metrics_json)
