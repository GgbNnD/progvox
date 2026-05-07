# 项目执行规则

## 环境与数据

- 唯一开发环境为 `alg` conda 环境，默认命令形式为 `conda run -n alg ...`。
- 本机 GPU 为 RTX 4060 Laptop GPU，训练与推理优先使用 CUDA；脚本必须能在 CUDA 不可用时退回 CPU。
- 原始视频、处理后视频、模型权重、TensorBoard 日志和大型重建结果不提交进 git。
- 数据优先使用“小样本可复现”策略：先跑通 Xiph 小样本，再扩展 UVG/VoxCeleb2。

## 工程节奏

- 每个小任务按“实现 -> 验证 -> 产物归档 -> git commit”推进。
- 每个阶段性可交付物单独提交，提交信息使用清晰前缀，例如 `feat(tokenizer): ...`、`feat(generator): ...`、`feat(codec): ...`。
- 代码、报告、CSV、JSON、小型 PNG 图可以提交；`.pth`、`.mp4`、`runs/`、`data/raw/`、`data/processed/` 不提交。
- 若上下文接近压缩，需要先把当前状态写入 `docs/dev_context/YYYY-MM-DD-context.md`，再继续开发。

## 技术路线

- 主线为 ProGVC：多尺度残差 tokenizer、上下文 token 模型、生成器解码、渐进式传输和 ABR。
- GFVC 只作为后续风险备用和对比 baseline，不作为当前主线。
- 当前 tokenizer 使用 4/8/16/32 token 网格；未训练码本是功能基线，后续可替换为 learnable VQ。
- 当前生成器短程消融中 `cnn_gan` 暂时领先，端到端原型优先使用该生成器。
- 上下文模型、调度器和后续 ABR 都要保持可复现实验入口、指标产物和失败记录。

## 验证标准

- 环境：`scripts/check_env.py` 必须确认 CUDA、CompressAI、ffmpeg 可用。
- 单测：新增模型/传输模块必须有 pytest，报告写入 `reports/test_*.txt`。
- 实验：每个实验报告必须包含命令入口、指标表、关键假设和已知限制。
- 端到端：先离线 loopback，再做传输仿真，最后才接 WebRTC/DataChannel。
