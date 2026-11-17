# 03 · 固定版本的代码引述与论文—实现对应

**核对日期**：2026-11-18。以下引述都锚定**不可变 commit**，而不是 `main`／`master` 浮动分支。已通过 GitHub 连接器读取相关文件区间，属于 **STATIC_ONLY**，不代表本地运行、训练复现或实验验收。

## 上游版本锁定

| ID | 上游仓库 | 检查的 commit | 用途 |
|---|---|---|---|
| W | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | `9737cba9c1c3c4d04b33fcad41c111989865d315` | 原生 T2V-1.3B、原生条件和调度入口 |
| U | [Fr0zenCrane/Uni-ViGU](https://github.com/Fr0zenCrane/Uni-ViGU) | `b73244f3a71143a3960ba9814f3686f50a778620` | 已有的双模态去噪、文字/视频联合网络与推理 |
| M | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | `c112c526d193436838c98d81455ee51f90309470` | 吸收式 mask 文字扩散、参数化和采样 |
| F | [ericbill21/fullflow](https://github.com/ericbill21/fullflow) | 仓库默认 `gh-pages`，不绑定可用训练实现 | 目前只作为图文后训练**论文参考**，不要假设训练脚本已发布 |

## W · Wan2.1 T2V-1.3B 的不可变源码入口

| Claim / 要验证的算子 | 文件与固定行号 | 源码观察 |
|---|---|---|
| 1.3B 配置 | [`wan/configs/wan_t2v_1_3B.py` L19–28](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/wan_t2v_1_3B.py#L19-L28) | patch=(1,2,2)、D=1536、FFN=8960、heads=12、layers=30 |
| prompt 在采样前编码 | [`wan/text2video.py` L173–184](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L173-L184) | 正向／负向 prompt 通过 T5 编码，在去噪循环外得到 `context` |
| 采样循环与 CFG | [`wan/text2video.py` L202–254](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L202-L254) | 每一步同一 `context` 输入视频模型，正负分支的速度由 CFG 合并 |
| Cross-Attention Q/K/V 来源 | [`wan/modules/model.py` L162–181](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L162-L181) | `q` 来自视频 `x`；`k,v` 来自 `context` |
| 原生 Block 次序 | [`wan/modules/model.py` L278–317](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L278-L317) | self-attn → cross-attn → FFN；包含 AdaLN 调制与残差 |
| 文字 context 组织／复用 | [`wan/modules/model.py` L552–574](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L552-L574) | `text_embedding` 后传递相同 `context` 给各个 block |

**直接结论**：Wan 的文字条件静态输入，但不同帧的 Query 可以产生不同注意力。新增机制要提供的是**状态回读、文字状态更新及反馈**，不是声称“原模型完全不能动态读取文字”。

## U · Uni-ViGU 已实现部分与关键差异

| Claim / 需要引述的地方 | 固定链接 | 实际内容 |
|---|---|---|
| 联合网络的输入输出 | [`model/transformer.py` L398–516](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/model/transformer.py#L398-L516) | 视频 latent `x_v`、文本 token ID `x_t`、独立 `time_v/time_t`，固定 umT5 `cond_t`；返回视频速度和文字 logits |
| 文字 token embedding、两套时间嵌入 | [`model/transformer.py` L413–438](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/model/transformer.py#L413-L438) | `nn.Embedding`、两套时间条件、文字 head；不是把 video latent 直接当文字 |
| 视频与文字 Query／Key 拼接 | [`diffusers/models/transformers/transformer_wan.py` L127–182](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/diffusers/models/transformers/transformer_wan.py#L127-L182) | 共享 self-attention 的 Q/K/V 由视频、文本拼接后计算；各自使用 3D／1D RoPE |
| 独立的文字 FFN | [`diffusers/models/transformers/transformer_wan.py` L595–612](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/diffusers/models/transformers/transformer_wan.py#L595-L612) | `ffn_t=copy.deepcopy(block.ffn)`，并复制 norm；需与论文的初始化描述区分 |
| 联合 Block 的实际执行序 | [`diffusers/models/transformers/transformer_wan.py` L660–697](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/diffusers/models/transformers/transformer_wan.py#L660-L697) | 共享 self-attn → 对拼接模态做原有 T5 cross-attn → 各自 FFN；文字与视频状态分别输出 |
| 原生 T2V 与 T2VT 的文字输入差异 | [`inference_unified_wan.py` L746–788](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/inference_unified_wan.py#L746-L788) | T2V 固定 prompt token；T2VT 从 uniform/mask 的 token ID 源分布开始 |
| 联合去噪两种输出与 CFG | [`inference_unified_wan.py` L790–825](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/inference_unified_wan.py#L790-L825) | 视频速度、文字 logits 在一次 forward 得到；CFG 用于视频速度 |
| **公开离散文字更新** | [`inference_unified_wan.py` L827–845](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/inference_unified_wan.py#L827-L845) | softmax logits→multinomial→按步数进度随机覆盖 token；不是文字 embedding Euler 积分 |
| 最终文字解码 | [`inference_unified_wan.py` L853–872](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/inference_unified_wan.py#L853-L872) | 最终再 forward 文字 logits，以最终预测得到文本 |

### 论文与代码不可强行视为同一实现

**论文 v1 `3.1**：[arXiv:2604.08121v1](https://arxiv.org/html/2604.08121v1) 的公式 (7)–(11) 在文字 token **连续 embedding** 空间对高斯噪声线性插值，使用速度预测及最终词表匹配，并独立采样两种模态噪声水平；**公开推理代码**接收文本 token ID，输出 logits，并在 T2VT 路径作离散随机更新、使两个模态的时间戳同步。两个版本不能混用同一训练和采样描述。

**论文 `3.2** 将文本 FFN 描述为新引入并初始化，代码将视频 FFN 拷贝到文本分支；“新增模块”与“随机初始化”不是同义词。

**论文 `3.3** 的 Stage 1 确实以原始 prompt 为文字恢复目标，Stage 2 以详细 caption 为目标；公开根目录的源码快照中没有完整 `train_unified_wan.py` 训练入口。因此，Stage 1 的实际 corrupt distribution、loss 系数、condition dropout 的批次细节标记为 **OPEN_BINDING / UNVERIFIED**。不能凭推理脚本推断训练完全一致。

### 许可核对

Uni-ViGU 根目录 [`LICENSE`](https://github.com/Fr0zenCrane/Uni-ViGU/blob/b73244f3a71143a3960ba9814f3686f50a778620/LICENSE#L1-L20) 是 Apache-2.0；但部分源文件头部包含 `CC-by-NC` 注释，与根许可证存在待澄清差别。暂时只建立**外链源码索引**，不要直接复制或 vendoring 源文件进入此仓库。Wan 和 MDLM 根目录许可文本均为 Apache-2.0，若实际改造并分发仍需遵守原有版权和 NOTICE 义务。

## M · 离散 masked diffusion 的可用实现基线

| Claim | 固定链接 | 重要数学语义 |
|---|---|---|
| 对已揭示 token 的 SUBS 参数化 | [`diffusion.py` L259–277](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L259-L277) | 不能预测 mask 自身；未 mask 位置强制复制 |
| 随机 mask 前向腐化 | [`diffusion.py` L575–590](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L575-L590) | `q_xt` 根据 mask 机会 `move_chance` 替换 token |
| 文字采样中的 DDPM 转移 | [`diffusion.py` L611–639](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L611-L639) | 使用噪声率与预测的 `p(x_0)` 构造下一 token 转移，未 mask 的位置复制 |
| 主采样入口 | [`diffusion.py` L658–705](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L658-L705) | 步数、采样方法与模型前向条件，需要适配 Wan 的视频状态输入 |

这是**纯文本** MDLM 的源码；接入视频需要增设视觉读取和耦合接口，不能把它已有的重建准确率视为视频语义对齐能力。训练英文模型与 tokenizer 的适用分布应与首批测试语言相符。

## 论文级相近工作（不等同于已核实可运行源码）

| 论文 | 原始出处 | 对本课题的可用结论 |
|---|---|---|
| Wan 技术报告 | [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) | 原始模型架构、训练和评测背景 |
| Uni-ViGU Preview | [arXiv:2604.08121v1](https://arxiv.org/html/2604.08121v1) | 联合去噪与原 prompt 知识恢复的直接先例 |
| MDLM, NeurIPS 2024 | [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) | 吸收式 mask 文字扩散、可用 checkpoint |
| UniDiffuser, ICML 2023 | [PMLR](https://proceedings.mlr.press/v202/bao23a.html) | 两模态独立噪声时间、条件／联合扩散 |
| FullFlow (图像) | [arXiv:2605.20316](https://arxiv.org/abs/2605.20316) | LoRA 与文字头的后训练思路，不可直接外推视频性能 |
| Temporal Specification Guidance | [arXiv:2610.20822](https://arxiv.org/abs/2610.20822) | 对生成时间关系进行干预和评价；作为时序接口强对照 |
| Why Do Video Diffusion Models Violate Physics? | [arXiv:2609.23658](https://arxiv.org/abs/2609.23658) | Wan1.3B 运动相关头的可视化不等于因果控制；用真实消融识别 |

源码引用用于明确“真正算过什么”，论文用于明确作者的科学主张；两者发生偏差时分别记录，不自动替论文或代码作有利解释。

## 待核项目

- OPEN_BINDING：Uni-ViGU 完整训练脚本、Stage 1 的实际 mask/噪声核、梯度流与数据处理。
- OPEN_BINDING：选择 MDLM checkpoint 后的 tokenizer、mask token、最大长度与语言适用范围。
- METHOD_SPEC_GAP：T-V2TV 自己的 `TextCorrupt`、`TextRestore`、`TemporalAlign` 实现版本尚未冻结。
- UNVERIFIED：上游联合推理是否能在当前研究硬件重现、是否有可比较的视频侧收益。
