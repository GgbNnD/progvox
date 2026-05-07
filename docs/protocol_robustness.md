# DataChannel Protocol Robustness

- Generated at: `2026-05-07T20:01:25.958029+00:00`
- Command: `python scripts/simulate_datachannel_proto.py`
- CSV: `reports/datachannel_protocol_robustness.csv`
- JSON: `reports/datachannel_protocol_robustness.json`

## Results

| loss | packet delivery | decodable frames | full frames | completed layers | expired assemblies |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0% | 99.3% | 99.2% | 95.0% | 98.8% | 4 |
| 3.0% | 97.3% | 96.7% | 82.5% | 95.2% | 11 |
| 5.0% | 93.9% | 92.5% | 65.0% | 89.8% | 21 |

## Notes

- Decodable frames require Layer 0 to reassemble before deadline.
- Full frames require all four token layers to reassemble; this is naturally more sensitive to packet loss because enhancement layers fragment into more chunks.
- The next step is to send these binary packets over the aiortc DataChannel loopback instead of this in-memory packet-loss harness.