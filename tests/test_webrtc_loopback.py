import asyncio

import pytest

from transport.webrtc_loopback import LoopbackConfig, run_loopback


pytest.importorskip("aiortc")


def test_local_webrtc_loopback_smoke():
    result = asyncio.run(run_loopback(LoopbackConfig(message_count=2, video_frames=3, timeout_s=10.0)))

    assert result["ok"] is True
    assert result["messages_acknowledged"] == 2
    assert result["messages_received"] == 2
    assert result["video_frames_received"] >= 3
