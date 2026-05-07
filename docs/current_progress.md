# 当前进度整理

## 总览

项目已完成阶段一、阶段二和阶段三的可运行原型闭环，当前已从“离线压缩/生成链路”推进到“渐进式传输与网络自适应”，下一步进入 WebRTC/DataChannel 联调。

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

## 当前待办

- 阶段 4.1：准备 Janus 或本地 WebRTC loopback 环境，优先验证最小可运行连接。
- 阶段 4.2：实现 `transport/datachannel_proto.py`，把 token 层级、帧号、deadline 和 payload 打包为 DataChannel 二进制包。
- 阶段 4.3：把 tokenizer/context/generator/scheduler/ABR 接入离线 loopback，再推进实时摄像头 demo。
