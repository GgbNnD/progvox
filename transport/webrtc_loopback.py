"""Local aiortc loopback used as the first WebRTC integration check."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from av import VideoFrame


@dataclass(frozen=True)
class LoopbackConfig:
    """Parameters for the in-process peer connection smoke test."""

    message_count: int = 5
    video_frames: int = 12
    width: int = 160
    height: int = 90
    fps: int = 15
    timeout_s: float = 10.0


class SyntheticVideoTrack(VideoStreamTrack):
    """Small generated RGB video track for local loopback validation."""

    def __init__(self, frames: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.frames = frames
        self.width = width
        self.height = height
        self.fps = fps
        self.index = 0

    async def recv(self) -> VideoFrame:
        await asyncio.sleep(1.0 / self.fps)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:, :, 0] = (self.index * 17) % 255
        image[:, :, 1] = np.linspace(0, 255, self.width, dtype=np.uint8)
        image[:, :, 2] = np.linspace(255, 0, self.height, dtype=np.uint8)[:, None]
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self.index
        frame.time_base = Fraction(1, self.fps)
        self.index = (self.index + 1) % max(1, self.frames)
        return frame


async def run_loopback(config: LoopbackConfig | None = None) -> dict[str, Any]:
    """Run a local offer/answer exchange, DataChannel echo and video receive test."""

    cfg = config or LoopbackConfig()
    if cfg.message_count <= 0:
        raise ValueError("message_count must be positive")
    if cfg.video_frames <= 0:
        raise ValueError("video_frames must be positive")

    pc_sender = RTCPeerConnection()
    pc_receiver = RTCPeerConnection()
    channel = pc_sender.createDataChannel("progvc-control")
    start = time.perf_counter()
    acked_messages: list[int] = []
    received_messages: list[int] = []
    received_frames = 0
    channel_open = asyncio.Event()
    acks_done = asyncio.Event()
    frames_done = asyncio.Event()

    @channel.on("open")
    def on_open() -> None:
        channel_open.set()
        for index in range(cfg.message_count):
            channel.send(json.dumps({"type": "ping", "index": index}))

    @channel.on("message")
    def on_sender_message(message: str) -> None:
        payload = json.loads(message)
        if payload.get("type") == "ack":
            acked_messages.append(int(payload["index"]))
        if len(acked_messages) >= cfg.message_count:
            acks_done.set()

    @pc_receiver.on("datachannel")
    def on_datachannel(receiver_channel: Any) -> None:
        @receiver_channel.on("message")
        def on_receiver_message(message: str) -> None:
            payload = json.loads(message)
            received_messages.append(int(payload["index"]))
            receiver_channel.send(json.dumps({"type": "ack", "index": payload["index"]}))

    @pc_receiver.on("track")
    def on_track(track: Any) -> None:
        async def consume() -> None:
            nonlocal received_frames
            if track.kind != "video":
                return
            while received_frames < cfg.video_frames:
                await track.recv()
                received_frames += 1
            frames_done.set()

        asyncio.create_task(consume())

    pc_sender.addTrack(SyntheticVideoTrack(cfg.video_frames, cfg.width, cfg.height, cfg.fps))
    offer = await pc_sender.createOffer()
    await pc_sender.setLocalDescription(offer)
    await pc_receiver.setRemoteDescription(pc_sender.localDescription)
    answer = await pc_receiver.createAnswer()
    await pc_receiver.setLocalDescription(answer)
    await pc_sender.setRemoteDescription(pc_receiver.localDescription)

    try:
        await asyncio.wait_for(channel_open.wait(), timeout=cfg.timeout_s)
        await asyncio.wait_for(asyncio.gather(acks_done.wait(), frames_done.wait()), timeout=cfg.timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "datachannel_label": channel.label,
            "messages_sent": cfg.message_count,
            "messages_received": len(received_messages),
            "messages_acknowledged": len(acked_messages),
            "video_frames_received": received_frames,
            "elapsed_ms": elapsed_ms,
            "sender_connection_state": pc_sender.connectionState,
            "receiver_connection_state": pc_receiver.connectionState,
        }
    finally:
        await pc_sender.close()
        await pc_receiver.close()
