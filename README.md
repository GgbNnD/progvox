# ProGVC 极低码率视频通话原型

本仓库用于逐步实现 `project.md` 和 `plan.md` 中定义的生成式极低码率视频通话课程项目。当前执行策略是先完成可复现的阶段一基础设施：`alg` conda 环境、Xiph 小样本数据管线、CompressAI SSF2020 baseline 与评测报告。

## 快速开始

```bash
conda activate alg
python scripts/check_env.py
python scripts/download_sample_data.py
python scripts/dataset_stats.py --input data/raw/xiph_small --output reports/dataset_stats.json
python scripts/prepare_test_videos.py --input data/raw/xiph_small --output data/processed/xiph_small_clips
python eval/run_lvc_baseline.py --input data/processed/xiph_small_clips --qualities 1 2 3 4
python scripts/simulate_scheduler.py
python scripts/simulate_abr.py
python scripts/run_webrtc_loopback.py
python scripts/simulate_datachannel_proto.py
```

## 目录

- `project.md`：项目分阶段方案说明。
- `plan.md`：小任务划分与可交付成果。
- `scripts/`：环境检查、数据下载、统计和预处理脚本。
- `eval/`：baseline 和后续 RD/BD-Rate 评测入口。
- `models/`：ProGVC tokenizer、context model、generator 等模型模块。
- `transport/`：渐进式 token 调度、DataChannel 协议和 WebRTC 传输模块。
- `docs/`：进度、上下文压缩记录、实验与部署文档。
- `reports/`：可提交的小型 JSON/CSV/Markdown 指标报告。

## 当前约束

- 使用 `alg` conda 环境，不在仓库内创建虚拟环境。
- 原始视频、处理后视频、模型权重和大型重建结果不提交进 git。
- 每个可交付检查点完成后单独提交，便于回溯实验状态。
