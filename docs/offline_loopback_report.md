# Offline End-to-End Loopback

- Generated at: `2026-05-07T20:07:03.101190+00:00`
- Input clip: `data/processed/xiph_small_clips/foreman_cif_352x288_29.97fps_8bit_P420.yuv`
- Metrics CSV: `reports/offline_loopback_metrics.csv`
- Metrics JSON: `reports/offline_loopback_metrics.json`
- Sample image: `reports/offline_loopback_samples.png`
- Reconstruction video: `reports/videos/offline_loopback/reconstruction.mp4`

## Metrics

| metric | value |
| --- | ---: |
| frames | 64 |
| stall_frames | 4 |
| stall_rate | 0.0625 |
| sent_packets | 291 |
| sent_bytes | 67005 |
| sent_bitrate_kbps | 251.01748124999995 |
| psnr_rgb | 17.583234786987305 |
| ssim_rgb | 0.517462968826294 |
| avg_selected_layer | 1.625 |
| avg_delivered_layer | 1.7166666666666666 |
| encode_ms_per_frame | 1.7725867655826733 |
| packetize_ms_per_frame | 0.12373654772090958 |
| reassemble_ms_per_frame | 0.017796452084439807 |
| decode_ms_per_frame | 1.3633970943374152 |
| generator_ms_per_frame | 1.841577796767524 |
| estimated_e2e_latency_ms | 136.20497489110494 |
| gpu_peak_memory_mb | 284.107421875 |

## Notes

- Sender side: tokenizer encodes frames, ABR selects maximum token layer, scheduler simulation determines delivered layer depth, and token maps are fragmented into DataChannel packets.
- Receiver side: packets are reassembled by frame/layer, missing layers use the context-model fallback path, and the trained CNN-GAN generator reconstructs frames.
- This is still an offline loopback. The next step is to move the same packet payloads through the live aiortc DataChannel.