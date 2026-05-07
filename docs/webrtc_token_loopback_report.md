# WebRTC Token DataChannel Loopback

- Generated at: `2026-05-07T20:09:58.261899+00:00`
- Command: `python scripts/run_webrtc_token_loopback.py`
- Metrics JSON: `reports/webrtc_token_loopback.json`

## Result

| metric | value |
| --- | ---: |
| ok | True |
| datachannel_label | progvc-tokens |
| frames | 8 |
| layers | 4 |
| packets_sent | 56 |
| packets_received | 56 |
| bytes_received | 13752 |
| completed_layers | 32 |
| expected_layers | 32 |
| elapsed_ms | 285.7193450035993 |
| sender_connection_state | connected |
| receiver_connection_state | connected |
| expired_assemblies | 0 |
| duplicate_chunks | 0 |

## Notes

- This sends real binary ProGVC token packets over an aiortc DataChannel and reassembles them with `FrameReassembler`.
- The test still runs in one process; Janus/browser signaling can replace the in-process offer/answer without changing the packet format.