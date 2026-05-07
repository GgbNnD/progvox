# ProGVC Integration Test

- Generated at: `2026-05-07T18:06:01.450974+00:00`
- Input clip: `data/processed/xiph_small_clips/foreman_cif_352x288_29.97fps_8bit_P420.yuv`
- Metrics CSV: `reports/progvc_integration_metrics.csv`
- Sample image: `reports/progvc_integration_samples.png`
- LPIPS mode: `lpips-alex-rand`
- Context predictor: untrained Transformer interface with zero-residual fallback for missing token layers.

## ProGVC Prototype Results

| variant | max layer | bitrate kbps | PSNR-RGB | SSIM-RGB | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| progvc_l2_cnn_gan | 2 | 90.629 | 19.501 | 0.6417 | 0.0171 |
| progvc_l3_cnn_gan | 3 | 366.833 | 23.701 | 0.8815 | 0.0061 |

## SSF2020 Reference From Phase 1

| quality | bitrate kbps | PSNR-Y | MS-SSIM-RGB |
| ---: | ---: | ---: | ---: |
| 1 | 128.457 | 29.553 | 0.952402 |
| 2 | 190.311 | 30.850 | 0.963154 |
| 3 | 296.428 | 32.583 | 0.972269 |
| 4 | 403.875 | 34.219 | 0.979772 |

## Notes

- This is an offline smoke integration of tokenizer, context-model API, and generator, not the final trained ProGVC result.
- Layer-2 mode exercises missing-token completion; Layer-3 mode is the full-token reconstruction path near the first-stage 300-500 kbps target.
- Generated videos are written under `reports/videos/` and intentionally ignored by git.