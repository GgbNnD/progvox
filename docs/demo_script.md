# Demo Script

## 1. 环境检查

```bash
conda run -n alg python scripts/check_env.py
```

展示 CUDA、RTX 4060、CompressAI、ffmpeg 和 WebRTC stack 均可用。

## 2. Baseline 与 ProGVC 核心结果

```bash
conda run -n alg python eval/run_lvc_baseline.py --input data/processed/xiph_small_clips --qualities 1 2 3 4
conda run -n alg python run_progvc_codec.py --frames 20
```

展示 `docs/progvc_integration_test.md` 和 `reports/progvc_integration_samples.png`。

## 3. 传输与 ABR

```bash
conda run -n alg python scripts/simulate_scheduler.py
conda run -n alg python scripts/simulate_abr.py
```

展示滑动窗口调度降低 stall，以及 ABR 用较低码率维持同等 stall。

## 4. DataChannel 协议

```bash
conda run -n alg python scripts/simulate_datachannel_proto.py
conda run -n alg python scripts/run_webrtc_token_loopback.py
```

展示 token binary packet 在内存丢包仿真和真实 aiortc DataChannel 中均可重组。

## 5. 端到端 Loopback

```bash
conda run -n alg python scripts/run_offline_loopback.py --frames 64
```

展示 `docs/offline_loopback_report.md`、`reports/offline_loopback_samples.png` 和 `reports/offline_loopback_metrics.csv`。
