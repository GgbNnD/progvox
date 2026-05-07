# Context Model Training

- Generated at: `2026-05-07T19:34:45.418773+00:00`
- Input: `data/processed/xiph_small_clips`
- Frames: 160 total, 128 train
- Token sequence length: 1360
- Token shapes: `[(4, 4), (8, 8), (16, 16), (32, 32)]`
- Checkpoint: `checkpoints/context_model/context_model.pth`
- TensorBoard logs: `runs/context_model`
- Metrics CSV: `reports/context_model_training_metrics.csv`

## Final Metrics

- Final train loss: 4.5788
- Final val loss: 4.2248
- Final val perplexity: 68.360
- Final val next-token accuracy: 0.0410
- Best val loss: 4.2248 at step 80

## Metrics By Evaluation Step

| step | train loss | val loss | val perplexity | val accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.4562 | 6.4308 | 620.674 | 0.0022 |
| 20 | 5.7451 | 5.7129 | 302.742 | 0.0225 |
| 40 | 5.1551 | 4.8830 | 132.031 | 0.0296 |
| 60 | 4.6403 | 4.4069 | 82.011 | 0.0282 |
| 80 | 4.5788 | 4.2248 | 68.360 | 0.0410 |

## Notes

- This is the first formal training run on the local Xiph sample set.
- The model trains next-token prediction over flattened multi-scale tokenizer outputs.
- Checkpoints and TensorBoard event files are intentionally ignored by git.