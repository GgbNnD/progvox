import asyncio

import pytest

from transport.webrtc_token_loopback import TokenLoopbackConfig, build_token_packets, run_token_loopback


pytest.importorskip("aiortc")


def test_build_token_packets_produces_fragmented_binary_payloads():
    packets = build_token_packets(TokenLoopbackConfig(frames=2, layers=2, max_payload_size=32))

    assert packets
    assert all(isinstance(packet, bytes) for packet in packets)


def test_webrtc_token_datachannel_loopback():
    result = asyncio.run(run_token_loopback(TokenLoopbackConfig(frames=2, layers=2, max_payload_size=64)))

    assert result["ok"] is True
    assert result["packets_received"] == result["packets_sent"]
    assert result["completed_layers"] == result["expected_layers"]
