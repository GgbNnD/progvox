# WebRTC Loopback Report

- Generated at: `2026-05-07T19:57:27.069257+00:00`
- Command: `python scripts/run_webrtc_loopback.py`
- Metrics JSON: `reports/webrtc_loopback.json`

## Result

| metric | value |
| --- | ---: |
| ok | True |
| datachannel_label | progvc-control |
| messages_sent | 5 |
| messages_received | 5 |
| messages_acknowledged | 5 |
| video_frames_received | 12 |
| elapsed_ms | 1108.848620991921 |
| sender_connection_state | connected |
| receiver_connection_state | connected |

## Notes

- This is an in-process aiortc loopback that validates offer/answer, ICE, DataChannel echo and video track receive without Janus.
- It is the phase 4.1 smoke test before moving to Janus/browser signaling and the custom token packet protocol.