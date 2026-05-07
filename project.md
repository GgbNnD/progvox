# 生成式极低码率视频通话 —— 课程项目详细方案说明


## 一、项目概述与价值

### 1.1 立意

常规视频通话依赖H.264/HEVC等传统编码器在像素层面对每一帧进行压缩，码率由分辨率、帧率和内容复杂度的乘积决定。5Mbps可以承载1080p，降落到500kbps就只能退守模糊的240p，视觉体验崩溃式下降。

本项目的核心主张是**用生成式重建替代逐帧像素编码**——将视频通话带宽从5Mbps压缩到500kbps，同时不显著牺牲视觉质量。实现方式是在发送端只传输稀疏的“结构令牌”（token maps），接收端则利用生成模型从这些令牌中“脑补”出高质量视频帧。

本项目基于ProGVC范式提出了一套**渐进式生成视频压缩方案**：自底向上构建“基础编码层 → 自适应码率控制 → 渐进式传输 → 生成器解码 → WebRTC集成”，最终在真实网络条件下完成端到端视频通话演示。

### 1.2 核心技术路线

以下三条理念构成了该项目的核心技术引擎：

1. **ProGVC范式——渐进式 + 生成 + 自回归上下文统一压缩**：ProGVC受Visual Auto-Regressive（VAR）模型的next-scale prediction启发，将视频编码为多尺度残差token map，通过coarse-to-fine子集在渐进式传输中实现灵活码率适配。Transformer多尺度自回归上下文模型既用于高效熵编码，也在解码端预测截断的精细尺度token以恢复感知细节。

2. **“压缩即生成”——Decoder-Only重建**：与传统“先压缩像素再原样解回像素”的路径不同，本项目在解码端引入生成先验。发送端只负责压缩粗糙的运动骨架与结构信息，解码端用生成模型合成纹理、光照、面部表情等高频细节。这一思想在ProGVC和GFVC系列工作中均有体现。

3. **渐进式传输 + 自适应码率**：多尺度token map天然支持coarse-to-fine渐进式交付——传输顺序从粗到细，接收端先看到清晰轮廓，随后纹理逐渐被填充。同时，可根据网络带宽动态决定传输到哪一个尺度停止，从而在实时变化的信道条件下灵活控制码率与质量的trade-off。


## 二、整体架构与数据流向

### 2.1 架构总览图

```
┌────────────────────────────────────────────────────────────────────┐
│                         发送端 (Sender)                            │
│ ┌──────────┐     ┌──────────────┐     ┌─────────────────────┐     │
│ │ 摄像头    │────▶│ 基础视频编码器 │────▶│ ProGVC Tokenizer    │     │
│ │ 采集     │     │ (H.264/H.265) │     │ (分层token化 +       │     │
│ └──────────┘     └──────────────┘     │  残差编码)            │     │
│                                        └───────┬─────────────┘     │
│                                                │                  │
│                                    ┌───────────▼─────────────┐     │
│                                    │ 自适应码率控制器 (ABR)    │     │
│                                    │ • 读取网络状态            │     │
│                                    │ • 决定传输token尺度层级   │     │
│                                    └───────────┬─────────────┘     │
└────────────────────────────────────────────────┼───────────────────┘
                                                 │
            ═══════════ 网络 (WebRTC DataChannel) ═══════════
                                                 │
┌────────────────────────────────────────────────┼───────────────────┐
│                         接收端 (Receiver)      │                   │
│                                    ┌───────────▼─────────────┐     │
│                                    │ 渐进式Token缓冲器        │     │
│                                    │ (基础层→增强层逐步到达)   │     │
│                                    └───────────┬─────────────┘     │
│                                                │                  │
│                                    ┌───────────▼─────────────┐     │
│                                    │ 生成器解码 (Decoder)     │     │
│                                    │ • 自回归上下文预测缺失    │     │
│                                    │   token                  │     │
│                                    │ • 轻量Diffusion/VAE      │     │
│                                    │   细节合成               │     │
│                                    └───────────┬─────────────┘     │
│                                                │                  │
│                          ┌──────────┐    ┌─────▼──────┐           │
│                          │ 显示器    │◀───│ 渲染器     │           │
│                          └──────────┘    └────────────┘           │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据传输分层语义

ProGVC的核心创新之一在于视频被编码为**分层多尺度残差token map**，而非传统的像素位流。传输可以在任意token尺度层级停止，即“截断”，从而在不重新编码的情况下动态调整码率。

```
Layer 0（基础运动骨架）: ~50-80 kbps — 轮廓可见，纹理缺失
Layer 1（粗尺度残差）  : +40-60 kbps — 大块纹理出现
Layer 2（中尺度残差）  : +60-80 kbps — 细节开始填充
Layer 3（细尺度残差）  : +80-120 kbps — 几乎无损感知质量

总计（所有层都传输） : ~230-340 kbps
```

网络条件好时，可以传输到Layer 3；网络拥塞时，在Layer 1或Layer 2截断，解码端用生成模型“脑补”缺失的高频细节。这一设计使码率可以随网络波动实现几乎连续的自适应调节，而不需要像传统ABR那样切换编码档位。

### 2.3 数据流时序示意（渐进式交付）

```
时间轴 →
发送端: ─[Layer 0][Layer 1][Layer 2][Layer 3]─
接收端: ─────[Layer 0 到达]──[L1 到达]──[L2 到达]──[L3 到达]
               │              │           │           │
               ▼              ▼           ▼           ▼
渲染:     模糊轮廓       大纹理可见   细节填充中    近乎无损
(用户:   “能认出是谁”  “皮肤质感出来” “在变清晰”   “完全清楚了”)
```


## 三、项目实施步骤总览

本课程项目建议按四个阶段推进，总工期约 **15-17周**：

| 阶段                               | 周数      | 目标                                 | 核心交付物                                 |
| ---------------------------------- | --------- | ------------------------------------ | ------------------------------------------ |
| **一：环境搭建与学习型编码器基础** | 第1-3周   | 搭建开发环境、跑通基础编解码Pipeline | 可训练的LVC模型、BD-Rate对比报告           |
| **二：生成式压缩核心实现**         | 第4-8周   | 实现ProGVC框架的编码-生成-解码核心链 | ProGVC模块完整实现、码率-质量trade-off曲线 |
| **三：渐进式传输与自适应码率**     | 第9-11周  | 分层token调度 + 网络自适应ABR        | 自适应控制器、渐进式交付Demo               |
| **四：WebRTC集成与端到端系统联调** | 第12-15周 | 完整的实时视频通话系统               | 端到端可演示系统、最终报告                 |

> **技术路线选择说明**：本方案以ProGVC为主框架，同时兼容GFVC方案作为对比Baseline（尤其对于视频会议中的人脸内容）。第一阶段的基础编码器可选用CompressAI的SSF2020作为入门模型，后续阶段重写为ProGVC架构。如果某条路线在实现中遇到瓶颈，可退回到只需传输关键点+FOMM生成器的GFVC简化方案，降低架构复杂度。


## 四、阶段一：环境搭建与学习型编码器基础（第1-3周）

### 4.1 目标与产出

本阶段需要完成开发环境的全栈搭建，并使用CompressAI框架跑通一个基础的端到端学习型视频编码器（LVC），获得第一个可工作的编解码Pipeline。同时需理解评估体系和传统编码器Baseline。

### 4.2 步骤1.1：基础环境搭建

**硬件需求**

| 组件 | 最低配置                | 推荐配置                        |
| ---- | ----------------------- | ------------------------------- |
| GPU  | NVIDIA GTX 1080Ti (8GB) | NVIDIA RTX 4090 (24GB) 或 A6000 |
| CPU  | 8核                     | 16核+                           |
| 内存 | 32GB                    | 64GB                            |
| 存储 | 100GB SSD               | 500GB NVMe SSD                  |

**软件安装**

CompressAI是InterDigital维护的端到端压缩研究官方PyTorch平台，提供自定义操作、层、预训练模型以及与传统编解码器的对比评估脚本。

```bash
# 1. 创建虚拟环境
python3 -m venv gvc_env
source gvc_env/bin/activate

# 2. 安装基础依赖
pip install torch torchvision torchaudio
pip install compressai
pip install opencv-python numpy pandas matplotlib
pip install jupyter notebook ipywidgets

# 3. 从源码安装CompressAI（用于修改内部模型）
git clone https://github.com/InterDigitalInc/CompressAI
cd compressai
pip install -e '.[dev]'
```

**数据集准备**

| 数据集         | 用途                     | 规模          | 获取方式                                         |
| -------------- | ------------------------ | ------------- | ------------------------------------------------ |
| **VoxCeleb2**  | 人脸视频训练（GFVC路线） | ~1M条讲话视频 | `http://www.robots.ox.ac.uk/~vgg/data/voxceleb/` |
| **UVG**        | 通用视频编码性能测试     | 7条1080p序列  | `https://ultravideo.fi/dataset.html`             |
| **Xiph-5N**    | 快速原型验证             | 5条HD序列     | `https://media.xiph.org/video/derf/`             |
| **WebRTC自采** | 最终端到端系统验证       | 10条自采      | 项目自行录制                                     |

**验证安装**
```bash
# 测试CompressAI预训练模型
python3 examples/codec.py --help
python3 -m compressai.utils.bench vtm --help
```


### 4.3 步骤1.2：搭建第一个LVC Baseline

**方法一：使用CompressAI预训练的SSF2020模型**

SSF2020（Scale-Space Flow）是CompressAI当前内置的视频压缩模型，适合作为快速Baseline。

```bash
# 使用预训练模型编码/解码视频
python3 -m compressai.utils.video.eval_model pretrained \
  /path/to/video/folder/ -a ssf2020 -q 1
```


**方法二：从零训练（可选，约需3-5天）**

对于希望深入理解学习型编码器训练过程的同学，CompressAI提供了完整的训练脚本：

```bash
# 使用自带训练脚本（需修改模型架构参数）
python3 examples/train.py -d /path/to/image/dataset/ \
  --epochs 100 -lr 1e-4 --batch-size 16 --cuda --save
```


关键超参数与训练注意事项：
- 学习率建议从`1e-4`起步，每50 epoch衰减0.5倍
- Batch size受限于GPU显存，16是1080Ti的安全值
- 需要实现`RateDistortionLoss`（R-D损失函数）: `Loss = λ·D(x, x̂) + R(ŷ)`，其中D使用MSE或感知损失，R由熵模型估计的码率

### 4.4 步骤1.3：基准线评测

运行CompressAI自带的bench工具，建立H.265/VVC的BD-Rate基准：

```bash
# 评估传统编解码器
python3 -m compressai.utils.bench vtm --help
python3 -m compressai.utils.bench x265 --help
```


**BD-Rate计算与VMAF评估**

BD-Rate（Bjøntegaard Delta Rate）是衡量编码效率的标准指标。评测流程如下：

1. **生成RD曲线**：对每个测试视频，在至少4个不同QP/quality level下编码并记录（码率，PSNR）点
2. **计算BD-Rate**：使用三次多项式拟合两条RD曲线，计算曲线间的面积差。负值表示同等质量下码率节省。拟使用`python-bdrate`第三方库，在评估脚本中集成
3. **VMAF评估**：VMAF是Netflix开发的感知质量指标，相比PSNR/SSIM更接近人类主观感受。安装方式：`pip install libvmaf`

**初期Benchmark目标**

| 指标                     | H.265 (x265) | SSF2020 (Baseline) | 本项目目标 |
| ------------------------ | ------------ | ------------------ | ---------- |
| BD-Rate (vs. H.265)      | 0% (锚定)    | +5%~+15%           | -15%~-25%  |
| PSNR (Y分量) @ 500kbps   | ~36dB        | ~34-35dB           | ~36-38dB   |
| VMAF @ 500kbps           | ~75          | ~72                | ~78-85     |
| 编码复杂度 (KMACs/pixel) | <200         | ~300               | 100-200    |

> **参考**：MMSP 2026 PEVC挑战赛将编码器与解码器复杂度均约束在200 KMACs/pixel，并在YUV420色彩空间下使用加权平均PSNR作为质量指标，目标是在不超过ECM在QP=27下的目标码率的前提下，评估编码性能。

### 4.5 本阶段里程碑

- [ ] CompressAI环境搭建完毕，可正常编码/解码
- [ ] H.265/H.264基准RD曲线绘制完成
- [ ] 预训练SSF2020模型的BD-Rate和VMAF数据记录完毕
- [ ] 训练数据准备至少50条视频段落
- [ ] `eval_model`输出结构确认，用于后续自动化评测脚本


## 五、阶段二：生成式压缩核心实现（第4-8周）

### 5.1 目标与产出

本阶段是整个项目最核心的技术攻坚环节——实现ProGVC框架的编码-生成-解码完整链。可在ProGVC主路之外并行搭建一个GFVC简化Baseline用于快速验证和对比。


### 5.2 ProGVC完整技术架构（模块A：主路 ProGVC）

ProGVC将视频压缩重新定义为一个**渐进式多尺度生成任务**，而非传统的逐帧预测任务。其架构可分为三大子系统：

```
                    ProGVC Encoder
┌─────────────────────────────────────────────────────┐
│ 输入帧 x_t                                         │
│   │                                                 │
│   ▼                                                 │
│ ┌────────────────┐                                 │
│ │ 基础编码器      │ ◀── 轻量H.264/HEVC残差编码       │
│ │ (Base Encoder) │    或轻量CNN编码器               │
│ └───────┬────────┘                                 │
│         │ base_frame                                │
│         ▼                                          │
│ ┌────────────────┐                                 │
│ │ 残差提取        │     rₖ = x_t - ↑(↓(x_t), k)     │
│ │ (Residual      │     k级金字塔残差                │
│ │  Extractor)    │                                 │
│ └───────┬────────┘                                 │
│         ▼                                          │
│ ┌────────────────┐                                 │
│ │ 多尺度Token化   │     Token maps: T₀, T₁, T₂, T₃  │
│ │ (Multi-scale   │     scales: 4×4, 8×8, 16×16, 32×32
│ │  Tokenizer)    │                                 │
│ └───────┬────────┘                                 │
│         ▼                                          │
│ ┌────────────────┐                                 │
│ │ 自回归上下文模型 │     为每一个token预测概率分布     │
│ │ (Transformer)  │     P(Tₖ|T₀...Tₖ₋₁)             │
│ └───────┬────────┘                                 │
│         ▼                                          │
│ ┌────────────────┐                                 │
│ │ 熵编码          │    算术编码 → 压缩比特流          │
│ │ (AE/AC)        │                                 │
│ └────────────────┘                                 │
└─────────────────────────────────────────────────────┘

                    ProGVC Decoder
┌─────────────────────────────────────────────────────┐
│ 接收比特流                                         │
│   │                                                 │
│   ▼                                                 │
│ ┌────────────────┐                                 │
│ │ 熵解码          │     从比特流恢复概率估计           │
│ │ (Entropy Dec)  │                                 │
│ └───────┬────────┘                                 │
│   T₀, T₁, ... Tₖ received; Tₖ₊₁...可能截断           │
│         ▼                                          │
│ ┌────────────────┐                                 │
│ │ 上下文预测       │     利用自回归模型预测缺失token    │
│ │ (Context       │     P̂(Tₖ₊₁ | T₀...Tₖ)          │
│ │  Predictor)    │                                 │
│ └───────┬────────┘                                 │
│         ▼                                          │
│ ┌────────────────┐                                 │
│ │ 细节合成生成器  │     从完整token序列重建视频帧      │
│ │ (Detail Synth) │     • 轻量Diffusion model        │
│ │                │     • 或CNN decoder + GAN loss   │
│ └───────┬────────┘                                 │
│         ▼                                          │
│     高质量输出帧 x̂_t                                 │
└─────────────────────────────────────────────────────┘
```

**关键实现模块**

**模块1：多尺度Token化器（Tokenizer）**

- 输入：视频帧经基础编码器后的残差图r
- 通过拉普拉斯金字塔/Lanczos插值生成K级（建议K=4）尺度金字塔（如4×4, 8×8, 16×16, 32×32像素块）
- 每个尺度的一批token通过轻量VQ（Vector Quantization）转换为一组离散token索引
- 技术要点：token map是“残差”——即与上一层上采样重建之间的差值，而非绝对像素值

**模块2：自回归上下文模型（Transformer）**

- 架构：轻量GPT-style因果Transformer，参数量控制在50-100M以内（在实时性约束下）
- 输入：已传输的低尺度token（T₀到Tₖ）
- 输出：下一个尺度token的条件概率分布P(Tₖ₊₁ | T₀…Tₖ)
- 双重用途：①编码端——为熵编码提供精准的概率估计，使算术编码器可以极高地压缩token；②解码端——当传输在K层截断时，预测并生成缺失的高尺度token

**模块3：细节合成生成器（Detail Synthesizer）**

技术路径A：轻量Diffusion model
- 在解码器端，将预测出的完整token序列送入一个轻量conditional diffusion model（扩散步数限制在4-8步，在150ms内完成一帧合成）
- 可用`denoising-diffusion-pytorch`等开源库快速搭建原型

技术路径B：GAN + CNN decoder
- 更轻量但感知质量可能略低，使用条件GAN（条件为token序列）+ 感知损失（LPIPS/VGG loss）
- 整体延迟更低，适合作为第一版原型快速上线

### 5.3 实验项目与验证（第三周-第八周）

**实验2.1：多尺度Token化验证**

- 设计：对50条测试视频，提取K=4级残差token map，统计不同截断尺度下的SSIM下降曲线
- 指标：token编码数据量（kbps）、各尺度重建PSNR
- 目标：确认token化方案能在50kbps内传输“可辨认”的基础结构

**实验2.2：生成器消融实验**

- 对比三种生成器：纯CNN decoder、CNN+GAN loss、轻量Diffusion（4步采样）
- 每个方案训练至收敛后记录：PSNR、SSIM、LPIPS、FID（在解码帧与原始帧之间）
- 主观MOS测试（至少10名测试者，在三个截断尺度下评分）
- 选择PSNR+LPIPS最佳平衡的生成器作为主方案

**实验2.3：完整ProGVC性能对标**

将完整ProGVC系统与以下Baseline进行对比：

| 编码方案              | 码率        | PSNR (dB) | LPIPS↓    | MOS (1-5) |
| --------------------- | ----------- | --------- | --------- | --------- |
| H.265 (x265 veryslow) | 500kbps     | ~35.5     | 0.18      | 3.2       |
| SSF2020 (CompressAI)  | 500kbps     | ~34.0     | 0.22      | 2.8       |
| FOMM (GFVC Baseline)  | 300kbps     | ~30.5     | 0.15      | 3.5       |
| **ProGVC (本项目)**   | **500kbps** | **≥36.0** | **≤0.10** | **≥3.8**  |

### 5.4 GFVC简化基线并行搭建（模块B：辅助对比）

**编码端（发送端）**

- 使用MediaPipe Face Landmarker或HRNet提取人脸468个3D关键点（约7KB/s）
- 每30-60帧传输一张高质量关键帧（I-frame），其余帧仅传输关键点坐标变化

**解码端（接收端）**

- 用关键帧和关键点序列驱动一个FOMM（First Order Motion Model）生成器，动画重建人脸视频
- 参考项目：`https://github.com/AliaksandrSiarohin/first-order-model`

**GFVC验证实验**

- 对测试集，对比不同I帧间隔（30帧、60帧、120帧）下的生成质量
- 记录总体码率：I帧(JPEG+H.264块) + 关键点流的总和
- 训练数据需要针对目标场景（视频会议室环境）做微调


## 六、阶段三：渐进式传输与自适应码率（第9-11周）

### 6.1 目标与产出

利用ProGVC的多尺度分层特性，实现随网络带宽动态调节“精细度”的渐进式传输系统，并利用解码端自回归模型预测和细节合成器生成高质量的最终重建。

**网络带宽 → 截断尺度映射逻辑**

```
网络带宽 (kbps)    →   传输最大尺度   →   依赖生成器恢复
    50-80         →     Layer 0       →    3层（大纹理、细节）
    80-120        →     Layer 1       →    2层（中纹理、细节）
    120-180       →     Layer 2       →    1层（细节）
    180+           →     Layer 3       →    0层（几乎无损）
```

### 6.2 实验项目

**实验3.1：token调度协议**

- 实现一个模拟token调度器：在带宽trace（如Norway 3G/4G trace或`mahimahi`链路模拟器）下推送不同尺度的token
- 关键问题：当带宽波动导致某帧token在特定尺度截断后，下一帧如何处理？
  - 方案A：纯贪婪——每一帧独立决定截断点
  - 方案B：滑动窗口——所有帧最低保证Layer 0，带宽余量向历史帧分配更高层
- 评估指标：不同trace下的平均码率波动性、PSNR稳定性

**实验3.2：基于网络状态的ABR决策**

- 用WebRTC stats API周期性采样以下指标：RTT（往返延迟）、packet loss rate、estimated bandwidth、Buffer occupancy
- ABR控制器实现：
  - Rule-based：基于阈值切换token尺度（如RTT>300ms时降级一个尺度）
  - RL（进阶）：用轻量DQN，状态空间=[吞吐量, 延迟, 丢包率, buffer_occupancy]，动作空间={T₀停止, T₁停止, T₂停止, T₃停止}
- 在`mahimahi`或`tc netem`模拟的弱网环境下对比rule-based vs RL ABR

**实验3.3：渐进式渲染UX**

- 在WebRTC数据通道中为每个token包带上`{frame_id, scale_level, total_scales}`的header
- 接收端每收到一个token包立即更新显示（低延迟渐进式渲染）
- 评估用户主观体验：设置MOS调研，包含“基础可辨度”“纹理逐步填充的自然度”“卡顿主观感知”等维度


## 七、阶段四：WebRTC集成与端到端系统联调（第12-15周）

### 7.1 目标与产出

将ProGVC编码-生成-解码Pipeline嵌入Janus WebRTC Gateway，实现完整的实时视频通话系统，支撑最终Demo演示。

### 7.2 集成架构

Janus WebRTC Gateway提供两种数据传输通道：

- **RTP媒体流**：用于传输传统H.264/H.265视频（本项目中的基础帧）
- **DataChannel**：用于传输ProGVC token数据和自适应控制信令（如当前码率、token尺度请求等）

**路由规则**：
```
I-frame（关键帧）→ RTP media stream（H.264编码，浏览器原生解码）
P-frame的token数据 → DataChannel（ProGVC token flow, 自定义二进制协议）
网络状态反馈 → DataChannel（stats report + ABR决策消息）
```

### 7.3 实施步骤

**步骤4.1：Janus Gateway部署与配置**

- 部署Janus Gateway服务器（Ubuntu 22.04 LTS）
- 配置`janus.transport.http`开启REST API，`janus.plugin.streaming`开启streaming插件支持
- 配置DataChannel支持：`data_channels = true`
- 测试基础WebRTC连通性（浏览器 ↔ Janus ↔ 浏览器）

**步骤4.2：自定义DataChannel协议设计**

在WebRTC DataChannel上实现一个精简的二进制消息协议：

```
| Header (4 bytes)                | Payload (N bytes)              |
| Frame ID (2B) | Scale Lvl (1B) | Total Scales (1B) | Token data |
```

- Frame ID: 16bit，支持最多65535帧（约36分钟@30fps）
- Scale Level: 当前token包的尺度层级
- Total Scales: 该帧的总尺度层级
- 接收端据此判断该帧是否已完整到达，若截断则进入Context Predictor进行缺失token的生成

**步骤4.3：发送端Pipeline**

参考流程：
```
摄像头帧采集 → OpenCV预处理 → 基础H.264编码器（输出I帧）
           ↓
    残差提取 → 多尺度Token化 → 自回归上下文模型 → 熵编码
           ↓
    自适应码率控制器（读取WebRTC stats API，确定截断尺度）
           ↓
    通过DataChannel分包发送token
```

**步骤4.4：接收端Pipeline**

参考流程：
```
DataChannel收包 → 缓冲区重组（按frame_id、scale_level累积）
           ↓
    检测截断 → 若截断，调用自回归上下文模型预测缺失token
           ↓
    完整token序列 → 细节合成生成器（轻量Diffusion/CNN）→ 帧重建
           ↓
    RTP接收基础帧 → 与生成帧融合 → Canvas/Video元素渲染
```

**步骤4.5：端到端延迟优化**

使用WebRTC的`getStats()` API持续测量各环节延迟：
- 编码延迟（tokenization + entropy coding）：目标 < 30ms/帧
- 生成延迟（Context Prediction + Detail Synthesis）：目标 < 120ms/帧
- 总端到端延迟（采集→显示）：目标 < 500ms

常见延迟瓶颈及优化方向：
- Diffusion采样步数>8步时延迟明显增加，可减至4步（质量略微下降延迟显著降低）
- Transformer上下文模型推理可通过`torch.compile()`加速或转为ONNX Runtime部署
- 对于低性能设备，部分生成任务可以降级为更轻量的GAN decoder

### 7.4 本阶段里程碑

- [ ] Janus Gateway部署完成，基础WebRTC通话功能可用
- [ ] 自定义DataChannel协议稳定传输token，丢包率<1%
- [ ] 渐进式交付功能实现，延迟目标<500ms
- [ ] 端到端视频通话Demo完成录制（建议录制两组场景：稳定WiFi + 弱网模拟各一组）
- [ ] 主观MOS测试：至少15名测试者评分，含稳定网络与弱网两组
- [ ] 最终报告撰写完成（含RD曲线、BD-Rate对比、MOS总分、延迟测量数据）


## 八、评估方案

### 8.1 客观指标

| 指标                   | 公式/来源              | 工具                             | 目标             |
| ---------------------- | ---------------------- | -------------------------------- | ---------------- |
| **PSNR (Y分量)**       | `10·log₁₀(255²/MSE_Y)` | CompressAI `eval_model` / FFmpeg | ≥36dB@500kbps    |
| **MS-SSIM**            | 多尺度结构相似度       | `pytorch_msssim`                 | ≥0.95@500kbps    |
| **VMAF**               | Netflix感知质量模型    | `libvmaf` + FFmpeg               | ≥85@500kbps      |
| **LPIPS**              | 感知图像相似度         | `lpips` Python包                 | ≤0.10@500kbps    |
| **BD-Rate (vs H.265)** | 码率节省百分比         | CompressAI plot工具              | -15%~-25%        |
| **FID**                | 生成帧分布距离         | `pytorch-fid`                    | ≤30（越低越好）  |
| **MOS**                | 人类主观评分（1-5）    | 自建问卷                         | ≥4.0（稳定网络） |

> **关于VMAF的局限性提示**：研究表明VMAF在评估AI-based编解码器时可能出现与主观评分的大幅偏差——某测试中Deep Render相比SVT-AV1主观BD-Rate优势为45%，但VMAF仅显示3%的差距。因此本项目将VMAF与LPIPS和MOS共同使用，避免单一指标误导。

**PEVC参考**：MMSP 2026 PEVC挑战赛使用YUV420色彩空间下的加权平均PSNR = (6×PSNR_Y + PSNR_U + PSNR_V)/8作为质量指标，并要求码率不超过ECM在QP=27目标码率。

### 8.2 主观测试方案

- **测试者数量**：不少于15人，男女比例尽量均衡，年龄覆盖18-45岁
- **测试内容**：3-5段含不同内容的15秒视频（纯人脸/人脸+手势/近景+背景）
- **对比条件**：ProGVC@500kbps vs. H.265@500kbps vs. H.265@2Mbps（Golden Reference）
- **评分维度**：整体质量、纹理自然度、面部细节保留度、运动流畅度、渐进式渲染接受度
- **评分标准**（MOS 1-5分制）：
  - 5 = 与原始视频无法区分
  - 4 = 有轻微感知差异但不影响观感
  - 3 = 有可注意的失真但不令人反感
  - 2 = 失真明显且略微影响体验
  - 1 = 失真严重导致无法正常观看

### 8.3 系统性能指标

| 指标        | 测量方式                     | 目标值             |
| ----------- | ---------------------------- | ------------------ |
| 编码延迟    | Python `time.perf_counter()` | < 30ms/帧          |
| 生成延迟    | 同上                         | < 120ms/帧         |
| 端到端延迟  | WebRTC `getStats()`          | < 500ms            |
| GPU显存占用 | `nvidia-smi`                 | < 8GB (1080Ti可跑) |
| 总带宽占用  | 网络流量监控                 | < 500kbps          |
| 卡顿率      | buffer empty事件计数         | < 1%时间           |


## 九、时间规划表

| 周次 | 阶段 | 任务                                  | 检查点            |
| ---- | ---- | ------------------------------------- | ----------------- |
| 1    | 一   | 环境搭建 + CompressAI安装             | 环境就绪          |
| 2    | 一   | 数据集准备 + 预训练模型测试           | 训练数据就绪      |
| 3    | 一   | H.265基准评测 + BD-Rate计算           | 基准曲线完成      |
| 4    | 二   | 多尺度Token化器原型实现               | Token化验证       |
| 5    | 二   | 自回归Transformer上下文模型           | 概率预测正常      |
| 6    | 二   | 细节合成生成器（Diffusion/GAN）       | 生成器收敛        |
| 7    | 二   | ProGVC编码-解码全链路打通             | 全链路可用        |
| 8    | 二   | 评测对比（ProGVC vs. H.265 vs. GFVC） | RD曲线 + 消融报告 |
| 9    | 三   | 分层token调度器实现                   | 调度逻辑可用      |
| 10   | 三   | ABR控制器（rule-based或RL）           | 自适应功能就绪    |
| 11   | 三   | 渐进式交付 + 渲染器                   | 渐进效果可用      |
| 12   | 四   | Janus部署 + WebRTC连通                | WebRTC服务可用    |
| 13   | 四   | DataChannel协议实现 + token传输       | 数据传输OK        |
| 14   | 四   | 端到端系统联调 + 延迟优化             | 系统可演示        |
| 15   | 四   | MOS主观测试 + 最终报告撰写            | 项目完成          |


## 十、关键技术难点与应对策略

| 难点                            | 风险等级 | 应对策略                                                                                                    |
| ------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| **生成器训练不稳定**            | 🔴高      | 先用CNN+GAN方案快速出基线，再迭代到Diffusion；若Diffusion一直不稳则保留GAN方案作为毕业交付                  |
| **ProGVC自回归推理延迟过高**    | 🟡中      | 使用`torch.compile()`或ONNX Runtime加速；限制Transformer层数在6-8层以内，注意力头数在8以内                  |
| **渐进式传输的token包乱序**     | 🟡中      | DataChannel基于SCTP保证有序传输（默认），只需在接收端做按frame_id的缓冲区重组                               |
| **弱网下DataChannel阻塞**       | 🟡中      | 设置token包超时（300ms），超时后放弃该帧的高尺度token，直接用生成器补全                                     |
| **不同人脸/场景下生成质量波动** | 🟡中      | 训练数据包含多人种、多光照、多背景的多样性样本（建议VoxCeleb2 + 自采室内数据混合训练）                      |
| **Janus生产部署稳定性**         | 🟢低      | 开源成熟度高，文档完备；可备选方案：直接用原生WebRTC（浏览器端无服务器模式）                                |
| **GPU资源不足**                 | 🔴高      | 使用云端GPU（AutoDL/Vast.ai按小时租用）训练；推理阶段可尝试CPU+INT8量化的极简版，FP16推理可显著降低显存占用 |


## 十一、成果交付清单

### 11.1 代码仓库结构

```
progvc-videocall/
├── README.md                      # 项目总览、安装与使用说明
├── requirements.txt               # Python依赖
├── train/
│   ├── train_tokenizer.py         # Token化器训练
│   ├── train_context_model.py     # 自回归上下文模型训练
│   ├── train_generator.py         # 生成器训练
│   └── train_config.yaml          # 训练超参数配置文件
├── models/
│   ├── tokenizer.py               # ProGVC多尺度Token化器
│   ├── context_model.py           # Transformer自回归上下文模型
│   ├── generator.py               # 细节合成生成器（Diffusion/GAN）
│   ├── abr_controller.py          # 自适应码率控制器
│   └── gfvc_baseline.py           # GFVC对比Baseline
├── transport/
│   ├── datachannel_proto.py       # DataChannel二进制协议
│   ├── scheduler.py               # Token调度器
│   └── janus_config/              # Janus配置文件
├── sender/
│   ├── capture.py                 # 摄像头采集
│   ├── encoder.py                 # 编码器主流程
│   └── sender_main.py             # 发送端入口
├── receiver/
│   ├── decoder.py                 # 解码器/生成器主流程
│   ├── renderer.py                # 渐进式渲染器
│   └── receiver_main.py           # 接收端入口
├── eval/
│   ├── bd_rate.py                 # BD-Rate计算脚本
│   ├── run_benchmark.py           # 评测主脚本
│   └── mos_survey/                # MOS主观测试问卷模板
├── demo/
│   ├── demo.sh                    # Demo启动脚本
│   └── demo_video.mp4             # 系统演示录制
├── docs/
│   ├── architecture.md            # 架构设计文档
│   ├── implementation_notes.md    # 实现细节笔记
│   └── final_report.md            # 最终课程报告
└── tests/
    ├── test_tokenizer.py
    ├── test_context_model.py
    └── test_generator.py
```

### 11.2 文档交付物

| 文档                | 内容                                                                 | 篇幅建议     |
| ------------------- | -------------------------------------------------------------------- | ------------ |
| **方案报告**        | 问题定义、技术路线与模块选择决策、系统架构、实验结果与分析、Demo展示 | 5000-8000字  |
| **实现文档**        | 每个模块的关键代码说明、API文档、训练配置与超参数                    | 附于代码仓库 |
| **评测报告**        | 完整RD曲线、BD-Rate表、MOS结果、延迟测量                             | 含图表       |
| **演示视频**        | 至少包含3组对比（ProGVC/低码率 vs H.265/低码率 vs H.265/高码率原始） | 3-5分钟      |
| **MOS问卷数据分析** | 统计结果（均值+置信区间）、测试者人口统计、异常值分析                | 附录         |


## 十二、参考资料汇总

### 核心论文

| 论文                                        | 年份 | 与本项目的关联                               | 链接                                                                                          |
| ------------------------------------------- | ---- | -------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **ProGVC** (Li, Dong et al.)                | 2026 | 项目核心架构来源——渐进式生成视频压缩框架     | [arXiv:2603.17546](https://arxiv.org/abs/2603.17546)                                          |
| **GFVC综述** (Chen, Wang, Ye)               | 2024 | GFVC系统框架、特征表示分类、标准化进展       | DCC 2024 / [GitHub项目页](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Coding) |
| **DVC** (Lu et al.)                         | 2019 | 首个端到端视频压缩深度模型，理解LVC基础      | [arXiv:1812.00101](https://arxiv.org/abs/1812.00101)                                          |
| **Scale-Space Flow (SSF2020)**              | 2020 | CompressAI内置的视频压缩基线模型             | ECCV 2020                                                                                     |
| **语义+ VVC混合编码** (Samarathunga et al.) | 2026 | 语义通信视角下autoencoder与VVC融合的方案参考 | [IEEE Access](https://doi.org/10.1109/access.2026.3676702)                                    |
| **PEVC挑战**                                | 2026 | 端到端视频压缩公平评测标准与复杂度约束规范   | MMSP 2026                                                                                     |

### 开源框架与工具

| 框架/工具                       | 用途                           | 链接                                                |
| ------------------------------- | ------------------------------ | --------------------------------------------------- |
| **CompressAI**                  | LVC模型开发、训练、评测        | `github.com/InterDigitalInc/CompressAI`             |
| **Janus WebRTC Gateway**        | WebRTC流媒体服务器             | `github.com/meetecho/janus-gateway`                 |
| **FFmpeg + libvmaf**            | 传统编解码器基准评测与VMAF计算 | `ffmpeg.org`                                        |
| **PyTorch**                     | 所有AI模型的训练与推理框架     | `pytorch.org`                                       |
| **aiortc**                      | Janus的Python WebRTC测试客户端 | `github.com/aiortc/aiortc`                          |
| **LPIPS**                       | 感知图像相似度评估             | `github.com/richzhang/PerceptualSimilarity`         |
| **denoising-diffusion-pytorch** | 轻量Diffusion模型快速原型      | `github.com/lucidrains/denoising-diffusion-pytorch` |
| **mahimahi**                    | 网络链路模拟器                 | `github.com/ravinet/mahimahi`                       |

### 训练数据集

| 数据集             | 场景适用         | 规模            | 说明             |
| ------------------ | ---------------- | --------------- | ---------------- |
| **VoxCeleb2**      | 人脸视频通话     | >1M讲话视频片段 | 多人种、多场景   |
| **UVG**            | 通用视频编码评测 | 7条1080p序列    | 丰富场景多样性   |
| **Xiph-5N**        | 快速原型验证     | 5条HD序列       | 公开下载方便     |
| **WebRTC自采数据** | 端到端系统验证   | 10条以上自采    | 面向真实部署场景 |


## 十三、下一步行动建议

1. **即刻行动**：按照阶段一步骤1.1搭建CompressAI环境，确保一天内能跑出第一个编解码结果。

2. **并行预热**：在环境搭建的同时，可以仔细阅读ProGVC论文正文的前4页（引言+Methods概览），把握多尺度token化、自回归建模、渐进式传输三者之间的耦合逻辑。

3. **快速原型**：第一个功能原型建议选择GFVC路线（MediaPipe + FOMM），它的集成复杂度低、可视效果好，可以在3-5天内产出一个可演示的最小系统，为团队提供信心和中期展示素材。

4. **每周同步**：建议每周固定一次项目同步会议（60分钟），内容为：上周任务完成情况check（对照里程碑清单）+ Code Review + 下周任务分配。每两周输出一份简短的进度报告（一页纸）。

5. **风险缓冲**：第9周结束时评估核心ProGVC路线进度。如果token化+生成器全链路尚未跑通，则退回到GFVC方案作为毕业交付主体，ProGVC部分以设计文档+部分实验结果作为“探索性”内容呈现在最终报告中。

6. **资源建议**：如果只有一块GPU且显存≤8GB，训练阶段优先使用云端GPU（AutoDL A6000约¥3-5/小时）；推理Demo阶段可用本地GPU+FP16半精度推理。