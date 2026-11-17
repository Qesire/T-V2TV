# 05 · 文献关系、创新性边界与真正待解的问题

本节按**运算机制**而不是论文题目相似度判断先例；重要源码引用见 [03_code_references](03_code_references.md)。检索记录截至 2026-11-18，新增论文或版本必须再次核对。

## 直接先例

| 论文 | 已证明／已实现的内容 | 与本项目的关系 |
|---|---|---|
| [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)（2025） | umT5 固定条件 + 视频 flow matching；公开 1.3B 基座 | 目标系统与不可替代的原生基线 |
| [Uni-ViGU: Towards Unified Video Generation and Understanding via A Diffusion-Based Video Generator](https://arxiv.org/html/2604.08121v1)（2026，Preview） | 联合文字／视频去噪、原文 Knowledge Recall、共享注意力和模态 FFN | **exact overlap**：共同去噪和恢复原 prompt；**待验证差异**：文字恢复如何改善视频动作、时序和生成目标 |
| [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://proceedings.mlr.press/v202/bao23a.html)（UniDiffuser，ICML 2023） | 不同模态各自扰动时间，统一联合／条件分布 | **shared ingredient**：独立噪声时钟；其数据与主要对象是图文，不是 Wan 视频运动 |
| [CoDi: Any-to-Any Generation via Composable Diffusion](https://arxiv.org/abs/2305.11846)（2023） | 多模态扩散互相耦合，包含文字与视频生成设置 | **conditional overlap**：共同生成本身已非新颖机制 |
| [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)（MDLM，NeurIPS 2024） | 吸收态 mask 文字扩散及可靠采样实现 | **shared ingredient**：可迁移文字去噪器；不能证明视频受益 |
| [FullFlow: Upgrading Text-to-Image Flow Matching Models for Bidirectional Vision--Language Generation](https://arxiv.org/abs/2605.20316)（2026） | 图像模型经 LoRA + 文字头后训练支持双向图文生成 | **new application / conditional overlap**：低成本后训练思路直接相关，但图像结果不等于视频结果 |

## 时序、运动与语言动态性直接相关的对照

| 工作 | 有效问题 | 与本课题的区别 |
|---|---|---|
| [Temporal Specification Guidance for Text-to-Video Generation](https://arxiv.org/abs/2610.20822)（2026） | 用显式时间规范在推理中操控视频的顺序、时机与持续条件 | 证明条件接口可改变视频时序；其时间规范不是文字重建轨迹自动学到的 |
| [Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation](https://arxiv.org/abs/2604.10030)（2026） | 各视频时间段读取对应的 prompt 并平滑切换 | 预先提供事件时间路由；不依赖当前生成视频的文字恢复 |
| [TALC: Time-Aligned Captions for Multi-Scene Text-to-Video Generation](https://arxiv.org/abs/2405.04682)（2024） | 场景 caption 的显式时序对齐 | 说明“分段条件提高时间对应”本身已有先例 |
| [Why Do Video Diffusion Models Violate Physics? Unveiling the Flaws in Attention Mechanisms](https://arxiv.org/abs/2609.23658)（2026） | Wan1.3B 注意力头的可视化与因果消融、早期轨迹和位置锁定 | 指出“能读出运动”和“能控制运动”不同，为本项目的 R/W 分离提供机制依据 |
| [TV2TV: A Unified Framework for Interleaved Language and Video Generation](https://arxiv.org/abs/2512.05103)（2025/2026） | 沿片段时间交替自回归生成文本与视频 | **distinct mechanism**：片段级规划，不等于同一去噪过程内的原文恢复反馈 |

## 不能以此作为单独创新的表述

- “首次让文本和视频一起扩散”——已有 Uni-ViGU、CoDi 等。
- “首次恢复已知原 prompt 作为辅助任务”——Uni-ViGU Knowledge Recall 是直接先例；文字重建辅助视觉训练也有更早工作。
- “首次通过注意力实现语义—视频时间对齐”——TALC、Prompt Relay、TSG 等已研究。
- “使用一个更强的文本语言模型必定改善视频”——语言知识与视频执行能力不同；需要具体映射和生成端验证。
- “恢复原文为模型增加了新的条件信息”——在完整原文始终可见、文字噪声独立的理想设置下不成立；可能的收益是有限模型的重表示、优化与反馈。

## 更可信的待解空间（目前是研究假说）

**假说 A：视频条件下的原文恢复包含有效的视觉语义诊断。** 它须能区分同一 prompt 下的正确与错误视频，而非仅依赖语言概率或反复复制目标。

**假说 B：这项诊断可以转换为目标相关的修正算子。** 文字去噪器可能只是描述当前生成状态；新增写入模块需要同时感知原始目标和当前语义状态，才能修正而非强化错误。

**假说 C：事件—视频时间轴对齐比全局语义增强更关键。** 对“先抓握再抬起”一类错误，增强全局 token Value 可能不足；显示事件—帧关系的接口或许更有效，但应与已有时序控制强基线比较。

**假说 D：联合文字去噪相对静态文字提取有独立收益。** 与一次性 masked 表示、相同参数的普通 adapter、公用动态条件日程匹配比较，才能给“动态文字恢复”本身归因。

四条假说可以分别失败。真正的论文贡献最终取决于哪条获得可重复的原生视频证据，而不是预先锁定“联合文本视频扩散”的叙事。

## 研究主张对应的最小证据

| 想写出的主张 | 最少需要的真实证据 |
|---|---|
| 文本状态理解了视频动作 | 固定被遮蔽文字，变更视觉证据得到正确区分；检查复制泄漏 |
| 重建语义能控制视频 | 干预文字恢复状态或时间路由，最终视频动作按预期变化 |
| 文本重建**带来**视频改善 | 同一接口、有/无原文重建损失的等预算配对生成比较 |
| 持续的文字去噪不可替代 | 动态轨迹优于一次计算后固定的文字表示，计入额外 compute |
| 方法在后训练预算内实用 | 冻结/LoRA 的真实 GPU·小时、显存、推理延时和视频质量 Pareto |

**当前结论**：方法构造可行的文献基础足够；能够建立新的贡献的关键环节尚未得到实验验证。详细算法候选与停止规则见 [02](02_operator_and_alignment.md) 和 [04](04_experiment_logic.md)。
