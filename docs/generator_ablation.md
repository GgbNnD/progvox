# Generator Ablation

- Generated at: `2026-05-07T18:00:42.855827+00:00`
- Input clips: `data/processed/xiph_small_clips`
- Resize: 64 px
- Condition: tokenizer reconstruction through layer 2
- TensorBoard logs: `runs/generator_ablation`
- Local checkpoints: `checkpoints/generator`
- Metrics CSV: `reports/generator_ablation_metrics.csv`
- Sample image: `reports/generator_ablation_samples.png`
- LPIPS mode: `lpips-alex-rand`

| model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID-lite ↓ |
| --- | ---: | ---: | ---: | ---: |
| pure_cnn | 24.204 | 0.7336 | 0.0231 | 0.0054 |
| cnn_gan | 24.251 | 0.7310 | 0.0235 | 0.0044 |
| tiny_diffusion | 13.814 | 0.1585 | 0.0881 | 0.0453 |

## Selection

- Current recommended generator: `cnn_gan` for this smoke-scale run.
- This is a bootstrap ablation on a tiny local dataset, not a final convergence run.
- Model weights and TensorBoard event files are intentionally kept out of git.

## Checkpoints

- `pure_cnn`: `checkpoints/generator/pure_cnn.pth`
- `cnn_gan`: `checkpoints/generator/cnn_gan.pth`
- `cnn_gan_discriminator`: `checkpoints/generator/cnn_gan_discriminator.pth`
- `tiny_diffusion`: `checkpoints/generator/tiny_diffusion.pth`

## Metric Notes

- `FID-lite` is a Fréchet distance over color and pooled image features, used here as a deterministic no-download proxy.
- LPIPS uses random AlexNet perceptual weights by default to avoid external downloads in this environment; flip `lpips.pnet_rand` to `false` for publication-grade LPIPS.