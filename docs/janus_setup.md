# Janus/WebRTC Setup Notes

Phase 4.1 starts with a local `aiortc` loopback smoke test, then moves the same offer/answer and DataChannel flow behind Janus.

## Local Smoke Test

```bash
conda activate alg
python scripts/run_webrtc_loopback.py
```

Expected result:

- one in-process offer/answer exchange
- one `progvc-control` DataChannel
- JSON ping/ack messages delivered
- synthetic video frames received by the answer peer

The latest local result is archived in `docs/webrtc_loopback_report.md` and `reports/webrtc_loopback.json`.

## Janus Deployment Sketch

Install system dependencies and Janus Gateway outside the conda environment. On Ubuntu-like systems:

```bash
sudo apt-get update
sudo apt-get install -y janus libmicrohttpd-dev libjansson-dev libssl-dev libsrtp2-dev libsofia-sip-ua-dev libglib2.0-dev
```

For a source build, enable WebSockets and DataChannels:

```bash
./configure --prefix=/opt/janus --enable-websockets --enable-data-channels
make -j"$(nproc)"
sudo make install
sudo make configs
```

Runtime checklist:

- enable the Janus WebSockets transport
- enable the VideoRoom or EchoTest plugin for first browser/Python connectivity checks
- keep the ProGVC token channel on a named WebRTC DataChannel, separate from browser camera media
- record Janus URL, plugin, room id, ICE server config and client command in the experiment report

## Next Integration Step

Replace the in-process offer/answer exchange in `transport/webrtc_loopback.py` with a signaling client that exchanges SDP through Janus. Keep `scripts/run_webrtc_loopback.py` as the regression smoke test because it runs without external services.
