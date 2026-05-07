# 项目进度台账

## 已完成

- 2026-05-08：提交初始项目说明文档 `project.md` 与 `plan.md`。
- 2026-05-08：初始化仓库目录、README、忽略规则与进度台账。
- 2026-05-08：补齐 `alg` 环境的阶段一关键依赖：`ffmpeg`、`compressai`、`torchaudio`、评测与 notebook 工具。
- 2026-05-08：新增并运行 `scripts/check_env.py`，环境检查通过，报告写入 `reports/env_check.*`。
- 2026-05-08：新增 Xiph 小样本 manifest、并发断点下载、数据统计和 64 帧 YUV420 预处理脚本；五个样本均已下载和预处理。

## 当前阶段

- 阶段一：环境搭建与学习型编码器基础。
- 当前重点：补齐 `alg` 环境依赖，建立 Xiph 小样本数据管线，跑通 SSF2020 baseline。

## 检查点

- [x] `scripts/check_env.py` 输出 CUDA/GPU/CompressAI/ffmpeg 状态。
- [x] `environment.yml` 与 `requirements-lock.txt` 可复现项目关键依赖。
- [x] `data/manifests/xiph_small.yaml` 能下载首批小样本视频。
- [x] `dataset_stats.py` 输出视频数量、分辨率、帧数、时长。
- [x] `prepare_test_videos.py` 生成每个至少 64 帧的测试片段。
- [ ] `eval/run_lvc_baseline.py` 输出 SSF2020 RD 点 CSV 与报告。
