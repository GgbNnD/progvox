# Tokenizer Analysis

- Generated at: `2026-05-07T17:55:35.448154+00:00`
- Sampled frames: 50
- Visual comparison: `reports/tokenizer_visual_comparison.png`

| max layer | token bits/frame | est. kbps @30fps | PSNR-RGB | SSIM-RGB |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 144 | 4.320 | 13.056 | 0.3427 |
| 1 | 720 | 21.600 | 15.899 | 0.4071 |
| 2 | 3024 | 90.720 | 19.040 | 0.5385 |
| 3 | 12240 | 367.200 | 22.177 | 0.7361 |

## Interpretation

- Layer prefixes monotonically increase token rate and generally improve reconstruction fidelity.
- This tokenizer uses an untrained uniform RGB residual codebook, so the results are a functional baseline rather than the final compression quality target.
- The same API supports learnable codebooks for the later tokenizer training deliverable.