"""Rule-based adaptive bitrate controller for progressive token layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from transport.scheduler import TokenLayerSpec, TracePoint


@dataclass(frozen=True)
class NetworkObservation:
    """Network state observed before choosing the next frame token depth."""

    timestamp_ms: float
    throughput_kbps: float
    rtt_ms: float
    loss_rate: float
    queue_delay_ms: float = 0.0


@dataclass(frozen=True)
class ABRDecision:
    """Controller output for one frame or scheduling slot."""

    frame_id: int
    timestamp_ms: float
    selected_level: int
    estimated_throughput_kbps: float
    budget_bits_per_frame: float
    target_bits_per_frame: int
    quality_psnr: float
    reason: str


@dataclass(frozen=True)
class RuleBasedABRConfig:
    """Tuning knobs for :class:`RuleBasedABRController`."""

    fps: float = 30.0
    min_level: int = 0
    max_level: int = 3
    ewma_alpha: float = 0.35
    safety_factor: float = 0.82
    high_rtt_ms: float = 180.0
    high_loss_rate: float = 0.04
    emergency_loss_rate: float = 0.08
    queue_delay_high_ms: float = 80.0
    stable_upshift_windows: int = 4


class RuleBasedABRController:
    """Select the maximum token layer from throughput, RTT and loss signals."""

    def __init__(self, layer_specs: Sequence[TokenLayerSpec], config: RuleBasedABRConfig | None = None) -> None:
        if not layer_specs:
            raise ValueError("layer_specs cannot be empty")
        self.layer_specs = list(layer_specs)
        self.config = config or RuleBasedABRConfig(max_level=len(layer_specs) - 1)
        if self.config.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= self.config.min_level <= self.config.max_level < len(layer_specs):
            raise ValueError("invalid min/max level range")
        if not 0 < self.config.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self._estimated_throughput_kbps: float | None = None
        self._selected_level = self.config.min_level
        self._upshift_candidate: int | None = None
        self._upshift_count = 0

    @property
    def cumulative_bits(self) -> list[int]:
        total = 0
        bits = []
        for spec in self.layer_specs:
            total += spec.bits
            bits.append(total)
        return bits

    def reset(self) -> None:
        self._estimated_throughput_kbps = None
        self._selected_level = self.config.min_level
        self._upshift_candidate = None
        self._upshift_count = 0

    def _update_throughput(self, observation: NetworkObservation) -> float:
        effective = max(1.0, observation.throughput_kbps * (1.0 - observation.loss_rate))
        if self._estimated_throughput_kbps is None:
            self._estimated_throughput_kbps = effective
        else:
            alpha = self.config.ewma_alpha
            self._estimated_throughput_kbps = alpha * effective + (1.0 - alpha) * self._estimated_throughput_kbps
        return self._estimated_throughput_kbps

    def _throughput_candidate(self, budget_bits_per_frame: float) -> int:
        candidate = self.config.min_level
        for level, bits in enumerate(self.cumulative_bits):
            if level > self.config.max_level:
                break
            if bits <= budget_bits_per_frame:
                candidate = max(candidate, level)
        return candidate

    def _apply_congestion_penalty(self, candidate: int, observation: NetworkObservation) -> tuple[int, str]:
        reasons = []
        if observation.loss_rate >= self.config.emergency_loss_rate:
            candidate = min(candidate, self._selected_level, 1)
            reasons.append("emergency_loss")
        elif observation.loss_rate >= self.config.high_loss_rate:
            candidate = min(candidate, self._selected_level)
            candidate = max(self.config.min_level, candidate - 1)
            reasons.append("high_loss")

        if observation.rtt_ms >= self.config.high_rtt_ms or observation.queue_delay_ms >= self.config.queue_delay_high_ms:
            candidate = max(self.config.min_level, min(candidate, self._selected_level) - 1)
            reasons.append("high_delay")

        return candidate, "+".join(reasons) if reasons else "throughput"

    def _apply_hysteresis(self, candidate: int, reason: str) -> tuple[int, str]:
        if candidate <= self._selected_level:
            self._selected_level = candidate
            self._upshift_candidate = None
            self._upshift_count = 0
            return self._selected_level, reason

        if self._upshift_candidate == candidate:
            self._upshift_count += 1
        else:
            self._upshift_candidate = candidate
            self._upshift_count = 1

        if self._upshift_count >= self.config.stable_upshift_windows:
            self._selected_level = candidate
            self._upshift_candidate = None
            self._upshift_count = 0
            return self._selected_level, f"{reason}+stable_upshift"
        return self._selected_level, f"{reason}+hold_upshift"

    def decide(self, frame_id: int, observation: NetworkObservation) -> ABRDecision:
        estimated = self._update_throughput(observation)
        budget = estimated * 1000.0 / self.config.fps * self.config.safety_factor
        candidate = self._throughput_candidate(budget)
        candidate, reason = self._apply_congestion_penalty(candidate, observation)
        selected, reason = self._apply_hysteresis(candidate, reason)
        return ABRDecision(
            frame_id=frame_id,
            timestamp_ms=observation.timestamp_ms,
            selected_level=selected,
            estimated_throughput_kbps=estimated,
            budget_bits_per_frame=budget,
            target_bits_per_frame=self.cumulative_bits[selected],
            quality_psnr=self.layer_specs[selected].quality_psnr,
            reason=reason,
        )

    def run(self, observations: Sequence[NetworkObservation]) -> list[ABRDecision]:
        return [self.decide(index, observation) for index, observation in enumerate(observations)]


def observations_from_trace(
    trace: Sequence[TracePoint],
    fps: float = 30.0,
    base_rtt_ms: float = 55.0,
    low_bandwidth_kbps: float = 240.0,
) -> list[NetworkObservation]:
    """Derive RTT/loss observations from the deterministic scheduler trace."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    interval_ms = 1000.0 / fps
    observations = []
    for index, point in enumerate(trace):
        scarcity = max(0.0, low_bandwidth_kbps - point.bandwidth_kbps)
        rtt_ms = base_rtt_ms + scarcity * 0.45 + point.loss_rate * 900.0
        queue_delay_ms = max(0.0, rtt_ms - base_rtt_ms)
        observations.append(
            NetworkObservation(
                timestamp_ms=index * interval_ms,
                throughput_kbps=point.bandwidth_kbps,
                rtt_ms=rtt_ms,
                loss_rate=point.loss_rate,
                queue_delay_ms=queue_delay_ms,
            )
        )
    return observations
