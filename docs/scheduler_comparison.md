# Scheduler Comparison

- Generated at: `2026-05-07T19:45:02.115248+00:00`
- Command: `python scripts/simulate_scheduler.py`
- Source FPS: 30.00
- Playback delay: 133.0 ms
- Summary CSV: `reports/scheduler_comparison.csv`
- Per-frame CSV: `reports/scheduler_frames.csv`
- Metrics JSON: `reports/scheduler_comparison.json`
- Trace plot: `reports/scheduler_trace.png`

## Token Layer Model

- L0: 144 bits/frame, PSNR proxy 13.06 dB
- L1: 576 bits/frame, PSNR proxy 15.90 dB
- L2: 2304 bits/frame, PSNR proxy 19.50 dB
- L3: 9216 bits/frame, PSNR proxy 23.70 dB

## Results

| policy | stall rate | decoded PSNR mean | render PSNR mean | PSNR fluctuation | PSNR std | sent kbps | avg layer | utilization | layer counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| greedy | 10.0% | 21.47 | 19.32 | 0.71 | 7.05 | 251.2 | 2.10 | 81.8% | -1:12, 0:0, 1:18, 2:24, 3:66 |
| sliding_window | 3.3% | 21.58 | 20.86 | 0.47 | 4.70 | 248.0 | 2.36 | 80.8% | -1:4, 0:5, 1:0, 2:46, 3:65 |

## Takeaways

- Greedy scheduling spends early bandwidth finishing enhancement layers for the oldest frame, which increases frame-to-frame quality swings when the trace drops.
- Sliding-window scheduling first spreads lower layers across frames inside the playback window, reducing deadline misses under the same trace.
- In this run, sliding-window stall rate changed from 10.0% to 3.3%, and PSNR fluctuation changed from 0.71 to 0.47.

## Failure Notes

- This is a trace-level simulator, not a packet-level WebRTC implementation yet.
- The PSNR values are layer quality proxies from the current 64x64 tokenizer/generator experiments; the next step is to connect these decisions to actual reconstructed video frames.