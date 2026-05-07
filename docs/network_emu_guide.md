# Weak Network Emulation Guide

This project uses `scripts/simulate_scheduler.py` for deterministic trace-driven tests, then validates real transport code with Linux traffic shaping once WebRTC/DataChannel is wired in.

## Option A: tc netem

Run the sender/receiver over a dedicated network interface or a local veth pair. Do not apply these commands to your main network interface unless you are ready to remove the rule immediately.

```bash
sudo tc qdisc add dev <iface> root handle 1: netem delay 80ms 20ms loss 2%
sudo tc qdisc add dev <iface> parent 1:1 handle 10: tbf rate 300kbit burst 32kbit latency 200ms
```

Remove the rule:

```bash
sudo tc qdisc del dev <iface> root
```

Suggested presets:

| name | bandwidth | delay | jitter | loss |
| --- | ---: | ---: | ---: | ---: |
| mild call | 500 kbit/s | 50 ms | 10 ms | 1% |
| weak mobile | 180 kbit/s | 90 ms | 25 ms | 3% |
| collapse | 70 kbit/s | 140 ms | 40 ms | 5% |
| short outage | 3 kbit/s | 140 ms | 40 ms | 5% |

## Option B: mahimahi

Install `mahimahi` from the system package manager, then place a one-column bandwidth trace in `reports/` with bytes-per-ms or packet events depending on the tool mode you choose.

```bash
mm-delay 80 mm-link uplink.trace downlink.trace -- python sender_main.py
```

For this phase, keep the simulator trace as the canonical comparison source and use `tc`/`mahimahi` only to sanity-check that the same scheduling behavior appears under live sockets.

## Measurement Checklist

- Record the exact command, trace path, and scheduler policy.
- Capture per-frame delivered max layer, stall flag, sent bits, and render-quality proxy.
- Report average decoded PSNR, average PSNR fluctuation, stall rate, and sent bitrate.
- Keep raw video, live packet captures, and large logs outside git; commit only compact CSV/JSON summaries and plots.
