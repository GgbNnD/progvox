# 项目完成度评估与经验总结

## 总体结论

项目已经完成“可运行生成式极低码率视频通话原型”的主体目标：环境、数据、baseline、ProGVC 核心模块、上下文模型训练、渐进式传输、ABR、DataChannel 二进制协议、WebRTC 本地 loopback、离线端到端 loopback 和最终报告均已落地，并有测试和报告产物支撑。

严格对照 `plan.md`，任务不是 100% 完成：Janus Gateway 实机部署、真实摄像头通话、15 秒/3-5 分钟演示录屏尚未完成。当前实现用本地 `aiortc` loopback 和离线端到端 loopback 替代了外部 Janus/摄像头联调，因此更准确的评估是：核心工程与实验目标已完成，真实部署与演示录屏仍是最后的交付增强。

当前验证状态：

- 全量测试：`conda run -n alg pytest tests -q`，33 passed，3 warnings。
- 最新最终报告：`docs/final_report.md`。
- 演示流程脚本：`docs/demo_script.md`。
- 工作区在最后一次检查时保持干净。

## 分阶段完成度

| 阶段 | 完成度 | 说明 |
| --- | ---: | --- |
| 1.1 环境搭建 | 完成 | `alg` 环境可用，CUDA/RTX 4060/CompressAI/ffmpeg/WebRTC stack 已验证，环境文件和锁定文件已更新。 |
| 1.2 数据准备 | 完成 | Xiph 小样本 manifest、下载、统计、预处理完成；按“小样本优先”策略未扩展完整 VoxCeleb2/UVG。 |
| 1.3 SSF2020 baseline | 完成 | CompressAI SSF2020 4 quality RD 点、CSV、报告和曲线图完成。 |
| 2.1 Tokenizer | 完成 | 多尺度 residual tokenizer、测试、notebook 和分析报告完成。 |
| 2.2 Generator | 完成 | pure CNN、CNN+GAN、tiny diffusion 消融完成，短程结果选择 CNN-GAN。 |
| 2.3 Context 与 ProGVC 集成 | 完成 | context model 模块、正式训练、离线 codec 集成报告完成；端到端缺失层仍主要用 zero-residual fallback。 |
| 3.1 调度器 | 完成 | greedy/sliding-window 调度、弱网 trace 仿真、tc/mahimahi 指南、对比报告完成。 |
| 3.2 ABR | 完成 | 规则型 ABR、trace-driven notebook、ABR 报告完成。 |
| 4.1 WebRTC 基础联通 | 部分完成 | 本地 aiortc offer/answer、DataChannel echo、合成视频轨道完成；Janus 实机部署和真实摄像头通话未完成。 |
| 4.2 DataChannel 协议 | 完成 | token 二进制协议、CRC、分片、乱序重组、超时、丢包鲁棒性报告完成。 |
| 4.3 端到端系统 | 大部分完成 | 离线端到端 loopback、live token DataChannel loopback、`sender_main.py`/`receiver_main.py` 原型、最终报告完成；演示视频未录制。 |

## 关键结果

- SSF2020 baseline 聚合结果：quality 1-4 码率约 128.5、190.3、296.4、403.9 kbps，PSNR-RGB 约 28.95、30.20、31.83、33.54。
- ProGVC 离线 codec：Layer 2 约 90.6 kbps，PSNR-RGB 19.50；Layer 3 约 366.8 kbps，PSNR-RGB 23.70。
- 上下文模型训练：80 step，val loss 4.2248，perplexity 68.360。
- 调度器：同一弱网 trace 下 greedy stall 10.0%，sliding window stall 3.3%。
- ABR：发送码率从 248.0 kbps 降到 189.5 kbps，stall rate 保持 3.3%，代价是 render-PSNR proxy 降低。
- DataChannel protocol robustness：1%/3%/5% 丢包下 base-layer decodable frame rate 分别为 99.2%、96.7%、92.5%。
- 离线端到端 loopback：64 帧，约 251.0 kbps，stall 6.25%，估计端到端延迟 136.2 ms，峰值显存 284.1 MB。
- live aiortc token DataChannel：8 帧 x 4 层，56/56 packets received，32/32 layers completed。

## 主要经验

1. 小样本优先是正确路线。先用 Xiph small 建立“能跑通、能复现、能提交”的闭环，比一开始下载和处理大型数据集更稳。

2. 每个阶段都要留下机器可读指标。CSV/JSON 比只写 Markdown 更方便后续比较、画图和写报告。

3. 大型产物必须严格隔离。原始视频、处理后视频、模型权重、TensorBoard 日志和重建视频都应留在 git ignore 范围内，只提交 manifest、脚本、小图和指标。

4. 生成式视频压缩的难点不只是网络层。当前网络/协议链路已经比较完整，但画质瓶颈主要来自 tokenizer codebook、generator 训练规模和 context 补全策略。

5. 先做离线协议仿真，再做 live DataChannel，是降低联调风险的好顺序。`FrameReassembler` 先在内存丢包/乱序场景下验证，再放进 aiortc，问题边界清楚很多。

6. ABR 需要同时看码率、卡顿和画质波动。只降低码率会让画质显著下降；后续应该引入 QoE 目标，而不是只按吞吐阈值选层级。

7. 课程项目报告要和代码同步生成。`docs/current_progress.md`、`docs/dev_context/...`、`docs/final_report.md` 在每个检查点更新，最后汇总时省了大量追溯成本。

8. WebRTC 实机部署应尽早拆分风险。当前本地 aiortc loopback 已证明 packet 和 DataChannel 可行，但 Janus、浏览器、摄像头、NAT/ICE 是另一类系统风险，需要单独预留时间。

## 未完成与后续建议

- 补 Janus Gateway 实机部署，验证 Python/浏览器客户端通过 Janus 建联。
- 接真实摄像头输入，将 `sender_main.py` 和 `receiver_main.py` 拆成两个 live 进程。
- 录制 3-5 分钟演示视频，按 `docs/demo_script.md` 展示完整链路。
- 训练 learned tokenizer codebook，替换 deterministic residual codebook。
- 扩大 generator/context 训练集，接入更真实的缺失 token 补全，而不是主要依赖 zero-residual fallback。
- 将 ABR 从规则型升级为 QoE-aware 控制，显式平衡码率、stall、画质和画质波动。
