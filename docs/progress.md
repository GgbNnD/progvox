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

## 当前阶段

- 阶段二：生成式压缩核心实现。
- 当前重点：基于 tokenizer 与生成器推进自回归上下文模型和端到端 codec。

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
