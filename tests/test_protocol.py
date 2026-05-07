import pytest

from transport.datachannel_proto import (
    FrameReassembler,
    ProtocolError,
    TokenPacket,
    fragment_payload,
    pack_packet,
    unpack_packet,
)


def test_packet_roundtrip():
    packet = TokenPacket(
        frame_id=42,
        layer_id=2,
        chunk_id=0,
        chunk_count=1,
        deadline_ms=1000,
        payload=b"token-bytes",
    )

    restored = unpack_packet(pack_packet(packet))

    assert restored == packet


def test_fragmented_payload_reassembles_out_of_order():
    payload = b"abcdef" * 100
    packets = fragment_payload(7, 3, 1000, payload, max_payload_size=128)
    reassembler = FrameReassembler(timeout_ms=500)
    completed = None

    for packet in reversed(packets):
        completed = reassembler.push(pack_packet(packet), now_ms=100)

    assert completed is not None
    assert completed.frame_id == 7
    assert completed.layer_id == 3
    assert completed.payload == payload
    assert reassembler.pending_count == 0


def test_reassembler_expires_incomplete_layers():
    payload = b"x" * 300
    packets = fragment_payload(1, 0, 1000, payload, max_payload_size=100)
    reassembler = FrameReassembler(timeout_ms=50)

    assert reassembler.push(pack_packet(packets[0]), now_ms=10) is None
    assert reassembler.expire(now_ms=100) == 1
    assert reassembler.pending_count == 0


def test_unpack_rejects_corrupt_payload():
    raw = bytearray(pack_packet(TokenPacket(1, 0, 0, 1, 1000, b"abc")))
    raw[-1] ^= 0xFF

    with pytest.raises(ProtocolError, match="checksum"):
        unpack_packet(bytes(raw))


def test_pack_rejects_invalid_chunk_index():
    with pytest.raises(ProtocolError, match="chunk_id"):
        pack_packet(TokenPacket(1, 0, 2, 2, 1000, b"bad"))
