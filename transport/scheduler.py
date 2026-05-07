"""Layered token scheduling and trace-driven network simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TokenLayerSpec:
    """Transmission cost and quality proxy for one progressive token layer."""

    level: int
    bits: int
    quality_psnr: float
    label: str = ""


@dataclass(frozen=True)
class TracePoint:
    """One network slot in a bandwidth/loss trace."""

    bandwidth_kbps: float
    duration_ms: float
    loss_rate: float = 0.0

    @property
    def raw_capacity_bits(self) -> int:
        return max(0, int(round(self.bandwidth_kbps * self.duration_ms)))

    @property
    def effective_capacity_bits(self) -> int:
        delivered = self.raw_capacity_bits * (1.0 - self.loss_rate)
        return max(0, int(math.floor(delivered)))


@dataclass
class FrameState:
    """Mutable delivery state for a frame before its playback deadline."""

    frame_id: int
    generated_at_ms: float
    deadline_ms: float
    delivered_bits: list[int]
    finalized: bool = False

    def max_complete_layer(self, layer_specs: Sequence[TokenLayerSpec]) -> int:
        complete = -1
        for index, spec in enumerate(layer_specs):
            if self.delivered_bits[index] >= spec.bits:
                complete = index
            else:
                break
        return complete

    def next_missing_layer(self, layer_specs: Sequence[TokenLayerSpec]) -> int | None:
        for index, spec in enumerate(layer_specs):
            if self.delivered_bits[index] < spec.bits:
                return index
        return None

    def remaining_bits(self, level: int, layer_specs: Sequence[TokenLayerSpec]) -> int:
        return max(0, layer_specs[level].bits - self.delivered_bits[level])

    def delivered_total_bits(self) -> int:
        return int(sum(self.delivered_bits))


@dataclass(frozen=True)
class FrameDelivery:
    """Final receiver-visible state for one frame at its deadline."""

    frame_id: int
    deadline_ms: float
    max_layer: int
    quality_psnr: float
    delivered_bits: int
    stalled: bool


@dataclass
class SimulationResult:
    """Aggregate result for one scheduler policy."""

    policy: str
    frames: list[FrameDelivery]
    raw_network_bits: int
    sent_bits: int
    source_duration_s: float
    layer_specs: list[TokenLayerSpec] = field(default_factory=list)

    def layer_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {-1: 0}
        for spec in self.layer_specs:
            counts[spec.level] = 0
        for frame in self.frames:
            counts[frame.max_layer] = counts.get(frame.max_layer, 0) + 1
        return counts

    def summary(self) -> dict[str, float | int | str | dict[int, int]]:
        if not self.frames:
            return {
                "policy": self.policy,
                "frames": 0,
                "stall_rate": 0.0,
                "decoded_psnr_mean": 0.0,
                "render_psnr_mean": 0.0,
                "psnr_fluctuation": 0.0,
                "psnr_std": 0.0,
                "average_layer": -1.0,
                "sent_bitrate_kbps": 0.0,
                "network_utilization": 0.0,
                "layer_counts": self.layer_counts(),
            }

        qualities = [frame.quality_psnr for frame in self.frames]
        decoded = [frame.quality_psnr for frame in self.frames if not frame.stalled]
        layers = [frame.max_layer for frame in self.frames]
        diffs = [abs(qualities[index] - qualities[index - 1]) for index in range(1, len(qualities))]
        stall_count = sum(1 for frame in self.frames if frame.stalled)
        mean_quality = sum(qualities) / len(qualities)
        variance = sum((quality - mean_quality) ** 2 for quality in qualities) / len(qualities)
        return {
            "policy": self.policy,
            "frames": len(self.frames),
            "stall_rate": stall_count / len(self.frames),
            "decoded_psnr_mean": sum(decoded) / len(decoded) if decoded else 0.0,
            "render_psnr_mean": mean_quality,
            "psnr_fluctuation": sum(diffs) / len(diffs) if diffs else 0.0,
            "psnr_std": math.sqrt(variance),
            "average_layer": sum(layers) / len(layers),
            "sent_bitrate_kbps": self.sent_bits / max(self.source_duration_s, 1e-9) / 1000.0,
            "network_utilization": self.sent_bits / self.raw_network_bits if self.raw_network_bits else 0.0,
            "layer_counts": self.layer_counts(),
        }


class BaseScheduler:
    """Base class for token layer schedulers."""

    name = "base"

    def choose_frame(
        self,
        pending: Sequence[FrameState],
        layer_specs: Sequence[TokenLayerSpec],
        now_ms: float,
    ) -> FrameState | None:
        raise NotImplementedError

    def schedule(
        self,
        pending: Sequence[FrameState],
        layer_specs: Sequence[TokenLayerSpec],
        budget_bits: int,
        now_ms: float,
    ) -> int:
        remaining = int(budget_bits)
        sent = 0
        while remaining > 0:
            frame = self.choose_frame(pending, layer_specs, now_ms)
            if frame is None:
                break
            level = frame.next_missing_layer(layer_specs)
            if level is None:
                break
            chunk = min(frame.remaining_bits(level, layer_specs), remaining)
            if chunk <= 0:
                break
            frame.delivered_bits[level] += chunk
            remaining -= chunk
            sent += chunk
        return sent


class GreedyScheduler(BaseScheduler):
    """Fill the earliest pending frame to the highest layer before moving on."""

    name = "greedy"

    def choose_frame(
        self,
        pending: Sequence[FrameState],
        layer_specs: Sequence[TokenLayerSpec],
        now_ms: float,
    ) -> FrameState | None:
        candidates = [frame for frame in pending if frame.next_missing_layer(layer_specs) is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda frame: (frame.frame_id, frame.deadline_ms))


@dataclass
class SlidingWindowScheduler(BaseScheduler):
    """Spread lower layers across the playback window before enhancements."""

    window_frames: int = 5
    name: str = "sliding_window"

    def choose_frame(
        self,
        pending: Sequence[FrameState],
        layer_specs: Sequence[TokenLayerSpec],
        now_ms: float,
    ) -> FrameState | None:
        candidates = [frame for frame in pending if frame.next_missing_layer(layer_specs) is not None]
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda frame: (frame.deadline_ms, frame.frame_id))
        if self.window_frames > 0:
            candidates = candidates[: self.window_frames]
        return min(
            candidates,
            key=lambda frame: (
                frame.next_missing_layer(layer_specs),
                frame.deadline_ms,
                frame.frame_id,
            ),
        )


def default_video_call_layers(
    bits_per_token: int = 9,
    token_shapes: Sequence[tuple[int, int]] = ((4, 4), (8, 8), (16, 16), (32, 32)),
    quality_psnr: Sequence[float] = (13.06, 15.90, 19.50, 23.70),
) -> list[TokenLayerSpec]:
    """Return default 64x64 ProGVC layer costs from the current tokenizer."""

    if len(token_shapes) != len(quality_psnr):
        raise ValueError("token_shapes and quality_psnr must have the same length")
    specs = []
    for level, ((height, width), quality) in enumerate(zip(token_shapes, quality_psnr)):
        specs.append(
            TokenLayerSpec(
                level=level,
                bits=int(height * width * bits_per_token),
                quality_psnr=float(quality),
                label=f"L{level} {height}x{width}",
            )
        )
    return specs


def make_bursty_trace(
    slots: int,
    duration_ms: float,
    seed: int = 7,
) -> list[TracePoint]:
    """Create a deterministic weak-network trace with bandwidth drops."""

    rng = random.Random(seed)
    segments = [
        (16, 420.0, 0.010),
        (18, 85.0, 0.025),
        (14, 220.0, 0.015),
        (12, 3.0, 0.050),
        (18, 480.0, 0.010),
        (16, 90.0, 0.030),
        (14, 680.0, 0.008),
    ]
    trace: list[TracePoint] = []
    segment_index = 0
    while len(trace) < slots:
        length, base_kbps, base_loss = segments[segment_index % len(segments)]
        for _ in range(length):
            if len(trace) >= slots:
                break
            slot = len(trace)
            wave = 1.0 + 0.10 * math.sin(slot * 0.47) + rng.uniform(-0.08, 0.08)
            bandwidth = max(1.0, base_kbps * wave)
            loss = min(0.15, max(0.0, base_loss + rng.uniform(-0.006, 0.006)))
            trace.append(TracePoint(bandwidth_kbps=bandwidth, duration_ms=duration_ms, loss_rate=loss))
        segment_index += 1
    return trace


def simulate(
    scheduler: BaseScheduler,
    trace: Iterable[TracePoint],
    layer_specs: Sequence[TokenLayerSpec],
    num_frames: int,
    fps: float = 30.0,
    playback_delay_ms: float = 133.0,
) -> SimulationResult:
    """Run a deadline-based progressive token delivery simulation."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if playback_delay_ms <= 0:
        raise ValueError("playback_delay_ms must be positive")
    if not layer_specs:
        raise ValueError("layer_specs cannot be empty")

    interval_ms = 1000.0 / fps
    states: list[FrameState] = []
    deliveries: list[FrameDelivery] = []
    raw_network_bits = 0
    sent_bits = 0

    def finalize_until(cutoff_ms: float) -> None:
        for state in states:
            if state.finalized or state.deadline_ms > cutoff_ms + 1e-6:
                continue
            max_layer = state.max_complete_layer(layer_specs)
            stalled = max_layer < 0
            deliveries.append(
                FrameDelivery(
                    frame_id=state.frame_id,
                    deadline_ms=state.deadline_ms,
                    max_layer=max_layer,
                    quality_psnr=0.0 if stalled else layer_specs[max_layer].quality_psnr,
                    delivered_bits=state.delivered_total_bits(),
                    stalled=stalled,
                )
            )
            state.finalized = True

    for slot, point in enumerate(trace):
        now_ms = slot * interval_ms
        finalize_until(now_ms)

        if slot < num_frames:
            states.append(
                FrameState(
                    frame_id=slot,
                    generated_at_ms=now_ms,
                    deadline_ms=now_ms + playback_delay_ms,
                    delivered_bits=[0 for _ in layer_specs],
                )
            )

        pending = [
            state
            for state in states
            if not state.finalized and state.generated_at_ms <= now_ms + 1e-6 and state.deadline_ms > now_ms + 1e-6
        ]
        raw_network_bits += point.raw_capacity_bits
        sent_bits += scheduler.schedule(pending, layer_specs, point.effective_capacity_bits, now_ms)
        finalize_until(now_ms + point.duration_ms)

        if len(deliveries) >= num_frames:
            break

    finalize_until(float("inf"))
    deliveries = sorted(deliveries, key=lambda frame: frame.frame_id)[:num_frames]
    return SimulationResult(
        policy=scheduler.name,
        frames=deliveries,
        raw_network_bits=raw_network_bits,
        sent_bits=sent_bits,
        source_duration_s=num_frames / fps,
        layer_specs=list(layer_specs),
    )
