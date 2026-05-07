# ProGVC 极低码率视频通话原型最终报告

## 摘要

本项目实现了一个可复现的生成式极低码率视频通话原型。系统主线为 ProGVC：发送端将视频帧编码为多尺度 residual token maps，网络层按 token layer 渐进传输，接收端对缺失层做 context fallback，并使用 CNN-GAN 生成器重建图像。项目已覆盖环境、数据、SSF2020 baseline、tokenizer、generator、context model、弱网调度、ABR、DataChannel 二进制协议、本地 WebRTC loopback 和离线端到端 loopback。

## 环境与数据

- 环境：`alg` conda，Python 3.10.19，PyTorch 2.9.1+cu130，RTX 4060 Laptop GPU。
- 关键依赖：CompressAI 1.2.8，ffmpeg 7.1，aiortc 1.14.0，websockets 16.0，av 16.1.0。
- 数据：Xiph 小样本 5 个 clip，预处理为 64 帧 YUV420 测试片段。
- 环境验证：`reports/env_check.txt`，全量测试：`reports/test_all.txt`。

## Baseline

CompressAI SSF2020 baseline 在 Xiph small 聚合结果：

| quality | bitrate kbps | PSNR-RGB | MS-SSIM-RGB |
| ---: | ---: | ---: | ---: |
| 1 | 128.46 | 28.95 | 0.9524 |
| 2 | 190.31 | 30.20 | 0.9632 |
| 3 | 296.43 | 31.83 | 0.9723 |
| 4 | 403.88 | 33.54 | 0.9798 |

## ProGVC 核心模块

- Tokenizer：4 层 token maps，4x4 / 8x8 / 16x16 / 32x32，512-size codebook，9 bits/token。
- Generator：实现 pure CNN、CNN+GAN、tiny diffusion；短程消融选择 `cnn_gan`。
- Context model：GPT-style token Transformer，已完成 80 step 正式训练；val loss 4.2248，perplexity 68.360。

离线 codec 集成结果：

| variant | bitrate kbps | PSNR-RGB | SSIM-RGB | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Layer 2 + CNN-GAN | 90.63 | 19.50 | 0.6417 | 0.0171 |
| Layer 3 + CNN-GAN | 366.83 | 23.70 | 0.8815 | 0.0061 |

## 渐进式传输与 ABR

同一弱网 trace 下，滑动窗口调度显著降低卡顿：

| policy | stall rate | render PSNR mean | PSNR fluctuation | sent kbps |
| --- | ---: | ---: | ---: | ---: |
| greedy | 10.0% | 19.32 | 0.71 | 251.2 |
| sliding window | 3.3% | 20.86 | 0.47 | 248.0 |

规则型 ABR 使用 throughput EWMA、RTT、loss、queue delay 和稳定升档滞后：

| policy | stall rate | render PSNR mean | sent kbps |
| --- | ---: | ---: | ---: |
| sliding window | 3.3% | 20.86 | 248.0 |
| sliding window + ABR cap | 3.3% | 19.56 | 189.5 |

ABR 以约 23.6% 发送码率下降换取 1.29 dB render-PSNR proxy 下降，卡顿率保持不变。

## WebRTC 与 DataChannel

- 本地 WebRTC smoke test：DataChannel 5/5 ack，合成视频收帧 12，双方 connected。
- Token packet protocol：magic/version/frame/layer/chunk/deadline/payload length/CRC32。
- Protocol robustness：1%、3%、5% 丢包下 base-layer decodable frame rate 分别为 99.2%、96.7%、92.5%。
- Live aiortc token DataChannel：8 帧 x 4 层，56/56 packets received，32/32 layers completed，双方 connected。

## 端到端 Loopback

`scripts/run_offline_loopback.py --frames 64` 将 tokenizer、ABR、scheduler、DataChannel protocol、context fallback 和 CNN-GAN generator 串联成离线端到端路径。

| metric | value |
| --- | ---: |
| frames | 64 |
| sent bitrate | 251.0 kbps |
| stall rate | 6.25% |
| PSNR-RGB | 17.58 |
| SSIM-RGB | 0.5175 |
| encode latency | 1.77 ms/frame |
| packetize latency | 0.12 ms/frame |
| reassemble latency | 0.02 ms/frame |
| decode latency | 1.36 ms/frame |
| generator latency | 1.84 ms/frame |
| estimated E2E latency | 136.2 ms |
| peak GPU memory | 284.1 MB |

## 结论

项目已经达到“可运行原型系统”的目标：所有核心模块具备独立测试、实验入口和报告产物，且 token packet 已能通过真实 aiortc DataChannel 传输。当前方案的工程链路完整，但客观画质仍低于 SSF2020 baseline，主要原因是 tokenizer codebook 和 generator 训练都仍是小样本短程版本，context model 也还没有用于大规模自回归补全。

## 已知限制

- 当前 tokenizer 是 deterministic residual codebook，不是经过大规模训练的 learned VQ tokenizer。
- CNN-GAN generator 只做了短程小样本训练，泛化能力有限。
- Context model 已训练，但端到端缺失层补全仍以 zero-residual fallback 为主，避免密集 token AR 推理过慢。
- Live WebRTC 目前验证了 DataChannel token packet loopback，尚未接入 Janus 和真实摄像头。
- 未提交原始视频、处理后视频、模型权重、TensorBoard 日志和大型重建视频。

## 后续工作

1. 训练 learned tokenizer codebook，并扩大 generator/context 训练集。
2. 将 context model 接入有限预算 AR 补全或并行 masked prediction。
3. 将 `sender_main.py` / `receiver_main.py` 拆成 live sender/receiver，并通过 Janus 或 WebSocket 信令交换 SDP。
4. 接入真实摄像头，记录 3-5 分钟演示视频和真实网络 QoE。
