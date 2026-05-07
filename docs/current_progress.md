# 当前进度整理

## 总览

项目已完成阶段一和阶段二的可运行原型闭环，当前准备从“离线压缩/生成链路”推进到“渐进式传输与网络自适应”。

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

## 当前待办

- 正式训练上下文模型，替代当前未训练 fallback-only 原型。
- 阶段 3.1：实现 `transport/scheduler.py`，完成分层 token 调度与网络 trace 仿真。
- 输出调度器性能报告，至少对比贪婪策略和滑动窗口策略。
