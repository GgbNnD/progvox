"""Binary token packet protocol for WebRTC DataChannel transport."""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterable


MAGIC = b"PGVC"
VERSION = 1
MAX_LAYER_ID = 255
MAX_CHUNKS = 65535
HEADER = struct.Struct("!4sBBIBHHIII")
HEADER_SIZE = HEADER.size


class ProtocolError(ValueError):
    """Raised when a DataChannel packet is malformed."""


@dataclass(frozen=True)
class TokenPacket:
    """One fragmented token payload packet."""

    frame_id: int
    layer_id: int
    chunk_id: int
    chunk_count: int
    deadline_ms: int
    payload: bytes
    flags: int = 0


@dataclass(frozen=True)
class CompletedLayer:
    """Reassembled payload for one frame/layer pair."""

    frame_id: int
    layer_id: int
    deadline_ms: int
    payload: bytes


@dataclass
class PacketAssembly:
    """Mutable state for a partially received fragmented payload."""

    frame_id: int
    layer_id: int
    deadline_ms: int
    chunk_count: int
    first_seen_ms: int
    chunks: dict[int, bytes] = field(default_factory=dict)

    def add(self, packet: TokenPacket) -> None:
        if packet.chunk_count != self.chunk_count:
            raise ProtocolError("chunk_count changed for an existing assembly")
        if packet.deadline_ms != self.deadline_ms:
            raise ProtocolError("deadline changed for an existing assembly")
        self.chunks.setdefault(packet.chunk_id, packet.payload)

    def complete(self) -> bool:
        return len(self.chunks) == self.chunk_count

    def payload(self) -> bytes:
        if not self.complete():
            raise ProtocolError("assembly is incomplete")
        return b"".join(self.chunks[index] for index in range(self.chunk_count))


def _validate_uint(name: str, value: int, maximum: int) -> None:
    if not 0 <= value <= maximum:
        raise ProtocolError(f"{name} out of range")


def pack_packet(packet: TokenPacket) -> bytes:
    """Serialize one token packet to bytes."""

    _validate_uint("frame_id", packet.frame_id, 0xFFFFFFFF)
    _validate_uint("layer_id", packet.layer_id, MAX_LAYER_ID)
    _validate_uint("chunk_id", packet.chunk_id, MAX_CHUNKS)
    _validate_uint("chunk_count", packet.chunk_count, MAX_CHUNKS)
    _validate_uint("deadline_ms", packet.deadline_ms, 0xFFFFFFFF)
    _validate_uint("flags", packet.flags, 0xFF)
    if packet.chunk_count == 0:
        raise ProtocolError("chunk_count must be positive")
    if packet.chunk_id >= packet.chunk_count:
        raise ProtocolError("chunk_id must be smaller than chunk_count")
    payload_len = len(packet.payload)
    _validate_uint("payload_len", payload_len, 0xFFFFFFFF)
    checksum = zlib.crc32(packet.payload) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        VERSION,
        packet.flags,
        packet.frame_id,
        packet.layer_id,
        packet.chunk_id,
        packet.chunk_count,
        packet.deadline_ms,
        payload_len,
        checksum,
    )
    return header + packet.payload


def unpack_packet(data: bytes) -> TokenPacket:
    """Parse one token packet from bytes."""

    if len(data) < HEADER_SIZE:
        raise ProtocolError("packet shorter than header")
    magic, version, flags, frame_id, layer_id, chunk_id, chunk_count, deadline_ms, payload_len, checksum = HEADER.unpack(
        data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ProtocolError("invalid packet magic")
    if version != VERSION:
        raise ProtocolError("unsupported packet version")
    if chunk_count == 0:
        raise ProtocolError("chunk_count must be positive")
    if chunk_id >= chunk_count:
        raise ProtocolError("chunk_id must be smaller than chunk_count")
    payload = data[HEADER_SIZE:]
    if len(payload) != payload_len:
        raise ProtocolError("payload length mismatch")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise ProtocolError("payload checksum mismatch")
    return TokenPacket(
        frame_id=frame_id,
        layer_id=layer_id,
        chunk_id=chunk_id,
        chunk_count=chunk_count,
        deadline_ms=deadline_ms,
        payload=payload,
        flags=flags,
    )


def fragment_payload(
    frame_id: int,
    layer_id: int,
    deadline_ms: int,
    payload: bytes,
    max_payload_size: int = 900,
    flags: int = 0,
) -> list[TokenPacket]:
    """Split a token layer payload into protocol packets."""

    if max_payload_size <= 0:
        raise ProtocolError("max_payload_size must be positive")
    if not payload:
        return [
            TokenPacket(
                frame_id=frame_id,
                layer_id=layer_id,
                chunk_id=0,
                chunk_count=1,
                deadline_ms=deadline_ms,
                payload=b"",
                flags=flags,
            )
        ]
    chunk_count = math.ceil(len(payload) / max_payload_size)
    if chunk_count > MAX_CHUNKS:
        raise ProtocolError("payload requires too many chunks")
    packets = []
    for chunk_id in range(chunk_count):
        start = chunk_id * max_payload_size
        packets.append(
            TokenPacket(
                frame_id=frame_id,
                layer_id=layer_id,
                chunk_id=chunk_id,
                chunk_count=chunk_count,
                deadline_ms=deadline_ms,
                payload=payload[start : start + max_payload_size],
                flags=flags,
            )
        )
    return packets


class FrameReassembler:
    """Reassemble fragmented token layer packets with deadline and timeout drops."""

    def __init__(self, timeout_ms: int = 250) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.timeout_ms = timeout_ms
        self._assemblies: dict[tuple[int, int], PacketAssembly] = {}
        self.expired_assemblies = 0
        self.duplicate_chunks = 0

    @property
    def pending_count(self) -> int:
        return len(self._assemblies)

    def push(self, raw: bytes, now_ms: int) -> CompletedLayer | None:
        packet = unpack_packet(raw)
        if now_ms > packet.deadline_ms:
            self.expired_assemblies += 1
            return None
        key = (packet.frame_id, packet.layer_id)
        assembly = self._assemblies.get(key)
        if assembly is None:
            assembly = PacketAssembly(
                frame_id=packet.frame_id,
                layer_id=packet.layer_id,
                deadline_ms=packet.deadline_ms,
                chunk_count=packet.chunk_count,
                first_seen_ms=now_ms,
            )
            self._assemblies[key] = assembly
        elif packet.chunk_id in assembly.chunks:
            self.duplicate_chunks += 1
        assembly.add(packet)
        if not assembly.complete():
            return None
        completed = CompletedLayer(
            frame_id=packet.frame_id,
            layer_id=packet.layer_id,
            deadline_ms=packet.deadline_ms,
            payload=assembly.payload(),
        )
        del self._assemblies[key]
        return completed

    def expire(self, now_ms: int) -> int:
        expired = [
            key
            for key, assembly in self._assemblies.items()
            if now_ms > assembly.deadline_ms or now_ms - assembly.first_seen_ms > self.timeout_ms
        ]
        for key in expired:
            del self._assemblies[key]
        self.expired_assemblies += len(expired)
        return len(expired)

    def push_many(self, packets: Iterable[bytes], now_ms: int) -> list[CompletedLayer]:
        completed = []
        for raw in packets:
            layer = self.push(raw, now_ms)
            if layer is not None:
                completed.append(layer)
        return completed
