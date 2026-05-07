# ABR Analysis

- Generated at: `2026-05-07T19:52:41.680696+00:00`
- Command: `python scripts/simulate_abr.py`
- Decisions CSV: `reports/abr_decisions.csv`
- Summary CSV: `reports/abr_simulation_summary.csv`
- Metrics JSON: `reports/abr_analysis.json`
- Decision curve: `reports/abr_decision_curve.png`
- Notebook: `analysis_abr.ipynb`

## Decision Distribution

| max layer | frames |
| ---: | ---: |
| 0 | 14 |
| 1 | 19 |
| 2 | 35 |
| 3 | 52 |

## Scheduler Impact

| policy | stall rate | render PSNR mean | PSNR fluctuation | PSNR std | sent kbps | avg layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_window | 3.3% | 20.86 | 0.47 | 4.70 | 248.0 | 2.36 |
| sliding_window_abr | 3.3% | 19.56 | 0.68 | 5.08 | 189.5 | 2.01 |

## Takeaways

- The controller lowers the maximum transmitted layer during low-throughput and high-delay windows, then requires several stable observations before upshifting.
- Compared with uncapped sliding-window scheduling, ABR-capped scheduling changes stall rate from 3.3% to 3.3%.
- ABR sent bitrate is 189.5 kbps, a 23.6% reduction from 248.0 kbps for the uncapped scheduler.
- The cost in this conservative rule set is -1.29 dB render-PSNR proxy and +0.21 PSNR fluctuation on the same trace.

## Failure Notes

- This is a rule-based controller; it does not yet use learned QoE optimization.
- RTT values are derived from the deterministic weak-network trace for repeatable offline tests. Live WebRTC RTT/loss will replace this source in phase 4.