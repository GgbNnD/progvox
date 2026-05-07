"""aiortc DataChannel loopback for ProGVC binary token packets."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from aiortc import RTCPeerConnection

from transport.datachannel_proto import FrameReassembler, fragment_payload, pack_packet
from transport.scheduler import default_video_call_layers


@dataclass(frozen=True)
class TokenLoopbackConfig:
    """Parameters for binary token-packet DataChannel validation."""

    frames: int = 8
    layers: int = 4
    max_payload_size: int = 300
    timeout_s: float = 10.0


def deterministic_payload(frame_id: int, layer_id: int, size: int) -> bytes:
    return bytes((frame_id * 17 + layer_id * 31 + index) % 256 for index in range(size))


def build_token_packets(config: TokenLoopbackConfig) -> list[bytes]:
    """Build deterministic serialized token packets for the loopback."""

    if config.frames <= 0:
        raise ValueError("frames must be positive")
    if config.layers <= 0:
        raise ValueError("layers must be positive")
    layer_specs = default_video_call_layers()[: config.layers]
    raw_packets = []
    for frame_id in range(config.frames):
        deadline_ms = 1000 + frame_id * 33
        for spec in layer_specs:
            payload_size = max(1, (spec.bits + 7) // 8)
            payload = deterministic_payload(frame_id, spec.level, payload_size)
            for packet in fragment_payload(frame_id, spec.level, deadline_ms, payload, config.max_payload_size):
                raw_packets.append(pack_packet(packet))
    return raw_packets


async def run_token_loopback(config: TokenLoopbackConfig | None = None) -> dict[str, Any]:
    """Send binary token packets over a real aiortc DataChannel and reassemble them."""

    cfg = config or TokenLoopbackConfig()
    raw_packets = build_token_packets(cfg)
    expected_layers = cfg.frames * cfg.layers
    pc_sender = RTCPeerConnection()
    pc_receiver = RTCPeerConnection()
    channel = pc_sender.createDataChannel("progvc-tokens")
    reassembler = FrameReassembler(timeout_ms=1000)
    start = time.perf_counter()
    channel_open = asyncio.Event()
    completed_event = asyncio.Event()
    received_packets = 0
    received_bytes = 0
    completed_layers: dict[tuple[int, int], int] = {}

    @channel.on("open")
    def on_open() -> None:
        channel_open.set()
        for raw in raw_packets:
            channel.send(raw)

    @pc_receiver.on("datachannel")
    def on_datachannel(receiver_channel: Any) -> None:
        @receiver_channel.on("message")
        def on_message(message: bytes) -> None:
            nonlocal received_packets, received_bytes
            if isinstance(message, str):
                message = message.encode("utf-8")
            received_packets += 1
            received_bytes += len(message)
            now_ms = int((time.perf_counter() - start) * 1000.0)
            completed = reassembler.push(message, now_ms=now_ms)
            if completed is not None:
                completed_layers[(completed.frame_id, completed.layer_id)] = len(completed.payload)
            if len(completed_layers) >= expected_layers:
                completed_event.set()

    offer = await pc_sender.createOffer()
    await pc_sender.setLocalDescription(offer)
    await pc_receiver.setRemoteDescription(pc_sender.localDescription)
    answer = await pc_receiver.createAnswer()
    await pc_receiver.setLocalDescription(answer)
    await pc_sender.setRemoteDescription(pc_receiver.localDescription)

    try:
        await asyncio.wait_for(channel_open.wait(), timeout=cfg.timeout_s)
        await asyncio.wait_for(completed_event.wait(), timeout=cfg.timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "datachannel_label": channel.label,
            "frames": cfg.frames,
            "layers": cfg.layers,
            "packets_sent": len(raw_packets),
            "packets_received": received_packets,
            "bytes_received": received_bytes,
            "completed_layers": len(completed_layers),
            "expected_layers": expected_layers,
            "elapsed_ms": elapsed_ms,
            "sender_connection_state": pc_sender.connectionState,
            "receiver_connection_state": pc_receiver.connectionState,
            "expired_assemblies": reassembler.expired_assemblies,
            "duplicate_chunks": reassembler.duplicate_chunks,
        }
    finally:
        await pc_sender.close()
        await pc_receiver.close()
