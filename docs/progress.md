# 项目进度台账

## 已完成

- 2026-05-08：提交初始项目说明文档 `project.md` 与 `plan.md`。
- 2026-05-08：初始化仓库目录、README、忽略规则与进度台账。
- 2026-05-08：补齐 `alg` 环境的阶段一关键依赖：`ffmpeg`、`compressai`、`torchaudio`、评测与 notebook 工具。
- 2026-05-08：新增并运行 `scripts/check_env.py`，环境检查通过，报告写入 `reports/env_check.*`。
- 2026-05-08：新增 Xiph 小样本 manifest、并发断点下载、数据统计和 64 帧 YUV420 预处理脚本；五个样本均已下载和预处理。
- 2026-05-08：新增并运行 SSF2020 baseline 包装器，输出 `reports/ssf2020_rd_points.csv`、`reports/lvc_baseline.md` 和 RD 曲线图。
- 2026-05-08：完成阶段 2.1 tokenizer 原型：4/8/16/32 多尺度残差 token map、RGB 向量码本量化、截断重建、单测和 50 帧分析报告。
- 2026-05-08：完成阶段 2.2 生成器小型消融：纯 CNN、CNN+GAN、轻量 Diffusion 均可训练/推理，`cnn_gan` 在短程实验中暂时领先。
- 2026-05-08：完成阶段 2.3 离线 ProGVC 原型链路：tokenizer、上下文模型接口、缺失 token fallback、CNN+GAN 生成器和集成评测报告。
- 2026-05-08：正式训练上下文模型：基于 Xiph 小样本生成 1360 长度 token 序列，保存本地 checkpoint/TensorBoard 日志，并归档训练指标与报告。
- 2026-05-08：完成阶段 3.1 分层 token 调度器：实现贪婪/滑动窗口策略、固定弱网 trace 仿真、性能对比报告和弱网复现指南。
- 2026-05-08：完成阶段 3.2 规则型 ABR 控制器：基于吞吐 EWMA、RTT、丢包和升档滞后选择最大 token 层级，并输出 trace-driven notebook/报告。

## 当前阶段

- 阶段四：WebRTC 集成与端到端系统联调。
- 当前重点：进入阶段 4.1，规划 Janus/WebRTC 服务部署与基础 loopback。

## 检查点

- [x] `scripts/check_env.py` 输出 CUDA/GPU/CompressAI/ffmpeg 状态。
- [x] `environment.yml` 与 `requirements-lock.txt` 可复现项目关键依赖。
- [x] `data/manifests/xiph_small.yaml` 能下载首批小样本视频。
- [x] `dataset_stats.py` 输出视频数量、分辨率、帧数、时长。
- [x] `prepare_test_videos.py` 生成每个至少 64 帧的测试片段。
- [x] `eval/run_lvc_baseline.py` 输出 SSF2020 RD 点 CSV 与报告。
- [x] `models/tokenizer.py` 输出 K=4 多尺度 token map 并支持截断重建。
- [x] `tests/test_tokenizer.py` 通过并写入 `reports/test_tokenizer.txt`。
- [x] `analysis_tokenizer.ipynb` 与 `docs/tokenizer_analysis.md` 记录 50 帧 token 码率/质量分析。
- [x] `models/generator.py` 包含 CNN、PatchGAN 判别器和轻量 conditional diffusion。
- [x] `train/train_generator.yaml` 与 `train/train_generator.py` 可运行小型生成器消融。
- [x] `docs/generator_ablation.md` 记录 PSNR、SSIM、LPIPS、FID-lite 和当前方案选择。
- [x] `models/context_model.py` 提供 GPT-style 自回归 token Transformer 与 flatten/unflatten 工具。
- [x] `run_progvc_codec.py` 可按最大 token 层级模拟传输并输出解码视频/指标。
- [x] `docs/progvc_integration_test.md` 记录 Layer 2/Layer 3 原型链路结果与 SSF2020 参考。
- [x] `train/train_context_model.py` 完成上下文模型正式训练并输出 `docs/context_model_training.md`。
- [x] `transport/scheduler.py` 实现分层 token 调度器。
- [x] `docs/network_emu_guide.md` 记录弱网环境复现方法。
- [x] `docs/scheduler_comparison.md` 对比贪婪调度和滑动窗口调度。
- [x] `models/abr_controller.py` 实现规则型 ABR 控制器。
- [x] `analysis_abr.ipynb` 完成 trace-driven ABR 决策验证。
- [ ] Janus 或本地 WebRTC loopback 环境部署文档。
- [ ] 基础 WebRTC/DataChannel 连接验证。
