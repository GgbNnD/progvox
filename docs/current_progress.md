# 当前进度整理

## 总览

项目已完成阶段一、阶段二、阶段三、阶段 4.1 本地 WebRTC smoke test、阶段 4.2 DataChannel token 协议和阶段 4.3 离线端到端 loopback，当前进入实时 DataChannel loopback 与最终报告整理。

## 已完成提交

- `032f65e docs: add initial project specifications`
- `66ff0f7 chore: scaffold project workspace`
- `5e5ee3b chore(env): reproduce alg environment`
- `9e0208d chore(env): add environment verification`
- `748cd7b data: add xiph sample dataset pipeline`
- `2b31250 eval: add ssf2020 lvc baseline`
- `f638743 feat(tokenizer): add multiscale residual tokenizer`
- `5eb444a feat(generator): add detail synthesis ablation`
- `284b686 feat(codec): add progvc integration prototype`
- `9327d4f docs: consolidate rules context and progress`
- `8dcb946 feat(context): train token context model`
- `4f99adf feat(transport): add token scheduler simulation`
- `39e7be9 feat(abr): add rule based controller`
- `df9f47e feat(webrtc): add local loopback smoke test`
- `9005322 feat(protocol): add datachannel token packets`

## 阶段一成果

- `alg` 环境可用，CUDA/CompressAI/ffmpeg 检查通过。
- Xiph 小样本数据可下载、统计、预处理。
- SSF2020 baseline 已输出 4 个 quality RD 点。

## 阶段二成果

- Tokenizer：
  - 输出 4 层 token maps。
  - 支持截断重建。
  - full-layer token 码率约 367 kbps @ 30fps。

- Generator：
  - 已实现纯 CNN、CNN+GAN、轻量 Diffusion。
  - 短程消融中 `cnn_gan` 当前领先。
  - 本地权重保存在 `checkpoints/generator/`。

- Codec integration：
  - 已打通 tokenizer -> context fallback -> generator。
  - Layer 2 截断：约 90.6 kbps，PSNR-RGB 19.50，SSIM 0.6417。
  - Layer 3 全 token：约 366.8 kbps，PSNR-RGB 23.70，SSIM 0.8815。

## 上下文模型训练

- `train/train_context_model.py` 已在 Xiph 小样本上完成第一轮正式训练。
- Token 序列长度为 1360，训练 80 step。
- 最终验证 loss 4.2248，验证 perplexity 68.360，next-token accuracy 0.0410。
- 本地 checkpoint 在 `checkpoints/context_model/context_model.pth`，被 git 忽略。
- 报告：`docs/context_model_training.md`。

## 阶段 3.1 成果

- `transport/scheduler.py` 已实现 deadline-based 分层 token 调度仿真。
- 策略包括 `GreedyScheduler` 和 `SlidingWindowScheduler`。
- `scripts/simulate_scheduler.py` 使用固定弱网 trace 输出 CSV/JSON/PNG/Markdown。
- 对比结果：贪婪调度 stall rate 10.0%，滑动窗口 3.3%；PSNR fluctuation 从 0.71 降到 0.47。
- 报告：`docs/scheduler_comparison.md`，弱网复现指南：`docs/network_emu_guide.md`。

## 阶段 3.2 成果

- `models/abr_controller.py` 已实现规则型 ABR 控制器。
- 控制输入包括 throughput、RTT、loss rate、queue delay。
- 策略包括吞吐 EWMA、安全余量、拥塞降级和稳定升档滞后。
- `scripts/simulate_abr.py` 已生成 `analysis_abr.ipynb`、`docs/abr_analysis.md` 和 ABR 曲线图。
- 同一弱网 trace 下，ABR-capped sliding window 将发送码率从 248.0 kbps 降到 189.5 kbps，stall rate 保持 3.3%；代价是 render-PSNR proxy 从 20.86 降到 19.56。

## 阶段 4.1 成果

- `alg` 环境已安装 `aiortc==1.14.0`、`websockets==16.0`、`av==16.1.0`。
- `transport/webrtc_loopback.py` 已实现同进程 offer/answer、DataChannel echo 和合成视频轨道。
- `scripts/run_webrtc_loopback.py` 已输出 `reports/webrtc_loopback.json` 和 `docs/webrtc_loopback_report.md`。
- 最新 smoke test：DataChannel 5/5 ack，视频收帧 12，sender/receiver 均 connected。
- Janus 部署草案：`docs/janus_setup.md`；信令流程：`docs/webrtc_signaling_flow.md`。

## 阶段 4.2 成果

- `transport/datachannel_proto.py` 已实现 ProGVC token 二进制包协议。
- 协议字段包括 magic/version/flags、frame id、layer id、chunk id/count、deadline、payload length 和 CRC32。
- `FrameReassembler` 支持乱序分片重组、重复 chunk 统计、deadline 和 timeout 丢弃。
- 鲁棒性仿真：1% 丢包 decodable frame rate 99.2%，3% 为 96.7%，5% 为 92.5%。
- 报告：`docs/protocol_robustness.md`，指标：`reports/datachannel_protocol_robustness.csv`。

## 阶段 4.3 成果

- `transport/offline_loopback.py` 已串联 tokenizer、ABR、调度仿真、DataChannel protocol、context fallback 和 CNN-GAN generator。
- `scripts/run_offline_loopback.py --frames 64` 已输出 `docs/offline_loopback_report.md`、`reports/offline_loopback_metrics.csv/json` 和小样图。
- 64 帧 foreman loopback：发送码率 251.0 kbps，stall rate 6.25%，PSNR-RGB 17.58，SSIM-RGB 0.5175。
- 性能剖析：encode 1.77 ms/frame，packetize 0.12 ms/frame，reassemble 0.02 ms/frame，decode 1.36 ms/frame，generator 1.84 ms/frame，估计端到端延迟 136.2 ms，峰值显存 284.1 MB。
- `sender_main.py` 和 `receiver_main.py` 已作为当前一进程 loopback 原型入口，后续可拆为 live sender/receiver。

## 当前待办

- 将 `transport/datachannel_proto.py` 的 packet payload 接入 live `aiortc` DataChannel loopback。
- 整理 `docs/final_report.md`，汇总环境、数据、baseline、ProGVC、传输、ABR、端到端性能和失败记录。
- 视时间补一个 3-5 分钟演示脚本/录屏说明。
