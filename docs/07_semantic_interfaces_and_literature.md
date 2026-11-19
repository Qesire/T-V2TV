# 07 · 语义如何真正进入生成器：从 Kling、Bernini、X-Omni 到 FLUX.3

**阅读日期：2026-11-19；状态：论文与公开代码的研究性核对，STATIC_ONLY。** 目标仍是 **Wan2.1 T2V-1.3B 的视频生成提升**，允许有限后训练，不进行视频骨干预训练。此文扩展 [05 文献边界](05_literature_boundary.md)。不应预设「文字同时去噪」是最有效接口。

按完整计算图重新归类并补入 GRAN-TED、Semantic Routing 的综合整理，见 [09 文本条件的计算图与文献分类](09_text_condition_computational_graph.md)。

## 1. 用功能而非模型名称组织文献

| 来源 | 语义中间对象 / 发生在哪个阶段 | 已有的作用证据 | 对 T-V2TV 的实际启发 | 机制重合 |
|---|---|---|---|---|
| [Kling, Behind Video Generation, 2026-11](https://arxiv.org/abs/2611.04916) | dense caption -> 多层 text encoder 的逐 token hidden states -> DiT；并研究 caption、prompt enhancement、感知训练、条件稳定性 | 报告第 6 节的内部人类对照：相对 Qwen TE 基线，多模态适配/多粒度/稳定融合，指令遵循比例依次 1.04 / 1.11 / 1.14；基座与训练步匹配；不等于冻结 Wan 的收益 | **语义传输精度**可能是瓶颈。应先检查逐 token 的动作/关系信息是否仍能通过文字编码及 DiT 注意力作用到视频；避免只用 pooled 向量替代条件 | shared target |
| [FLUX.3 Image Klein / Diffusers 分析, 2026-10](https://huggingface.co/blog/flux-3) | Qwen3-VL 的 36 层取第 4,8,...,32 层；每 token 拼接 8×2560 -> 投影到 3072；每次去噪文字 token 均用 clean timestep 0；文本与图像先各自 5 block，再联合 28 block | 公布图像生成结构与代码，非视频任务；不是文字本身扩散 | **静态多层逐 token 语义仍可强有力**。动态文字分支必须与多层静态 TE 强对照，而非只与原生 T5 比 | distinct mechanism / strong static baseline |
| [Bernini, 2026](https://arxiv.org/abs/2605.22344) | MLLM 将目标视觉的连续 ViT embedding 作为 masked tokens 逐轮补全；MLP connector 将目标语义隐状态写入 DiT（另保留 T5 条件） | 编辑/参考生成与 ablation；T2V VBench 84.64 对底座 84.79，说明保留能力；公开 [代码](https://github.com/bytedance/Bernini)；1.3B 检查点是 **renderer-only** | **复原的对象可以是视觉语义而不是原文字**。有真实视频语义 teacher。先 planner 后 renderer：此处不是每个 DiT 步都再进行语义补全 | direct competitor for semantic-latent planning; not dynamic text reconstruction |
| [X-Omni, 2025](https://arxiv.org/abs/2507.22058) | SigLIP2 视觉语义 -> VQ 图像 token -> LLM 自回归生成语义 token -> FLUX.1-dev diffusion decoder 输出像素，RL 让生成 token 适应解码器 | **图像**生成与语义图像词表的实证，不是 T2V；解码器与语义先验对齐是关键 | **“LLM 生成 coarse image”** 是先生成语义可读的视觉代理，再让视觉模型补细节；可用连续目标视觉 embedding 替代重做 VQ 词表 | shared semantic prior; different generation process |
| [TokenFlow, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_TokenFlow_Unified_Image_Tokenizer_for_Multimodal_Understanding_and_Generation_CVPR_2025_paper.html) | 语义/像素双 codebook，通过共享 indices 关联高层语义与低层细节 | 验证了图像理解与重建需要不同粒度的 tokenizer | 如果离散视觉 token 表达过粗，可把语义—外观两条路径分开；本项目不需先训练新视觉词表 | relevant representation prior |
| [SemHiTok, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/49ffa271264808cf500ea528ed8ec9b3-Abstract-Conference.html) | 在语义 codebook 上建立细节层级 codebook | 图像理解/生成研究中的语义与细节解耦 | 需要匹配视频训练预算，**先用已有冻结 ViT 表示**，后续再考虑离散码本 | relevant representation prior |
| [A Dual Stream Visual Tokenizer, IJCAI 2025](https://www.ijcai.org/proceedings/2025/167) | 高层语义 tokens + 低层像素 tokens；粗分辨率引导与扩散高分辨率重建 | 图像场景的粗→细分工 | 与用户提到的“coarse image”路线同族；区别“语义中间变量”和“画面细节编码” | shared coarse-to-fine mechanism |
| [Transfusion, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/12678c3948153f4bc391f51e2082bd6e-Abstract-Conference.html) | 文字按 next-token loss 自回归；图片连续 patch 用 diffusion；图像内部双向注意力，跨序列元素因果 | 从头预训练的图文模型；实证支持不同模态保留各自合适的预测目标 | 联合模态**不要求文字也扩散**。语言可 AR、masked diffusion 或预编码，需由视频侧收益判别 | joint architecture shared ingredient |
| [MoT (Mixture-of-Transformers), TMLR 2025](https://arxiv.org/abs/2411.04996) | 共享多模态输入空间下，FFN/attention/norm 使用模态感知稀疏专家 | 作者在 Transfusion 模型设定报告相近质量约 1/3 FLOPs（特定规模与条件） | 如最终要联合文字／视觉状态，可让小的文本支路与视频支路保留独立参数和归一化；节省计算依赖具体实现，不能将 1/3 FLOPs 直接外推到 Wan | parameter architecture reference |
| [Uni-ViGU, 2026](https://arxiv.org/html/2604.08121v1) | 视频与文字 token 共同更新；Stage 1 已以原始 prompt 为文字恢复目标 | 已实现双模态与原文重建；公开推理 token sampler 与论文连续 embedding flow 存在实现差异 | **direct overlap**：原文重建与同时更新已经有先例。新贡献必须证明视频侧的语义传输/反馈收益 | exact overlap for basic premise |

**需要强调的证据差别**：Kling 的文本条件消融是在其内部训练与人类评测设置下进行的；FLUX.3 是图像且文本不去噪；Bernini 完整 planner 使用 Wan2.2-T2V-A14B，而公开 Wan2.1-1.3B 模型是**仅渲染器 Bernini-R**，不能将 Bernini-R-1.3B 误记成完整 MLLM planner 的开源 checkpoint。

## 2. 从这些工作抽象出共同瓶颈：Semantic Transport

设用户原文 `c`、视频 latent `x_t`。Wan 的标准路径是：

\[
c\xrightarrow{E_{\rm T5}} C_j
\xrightarrow{Q(x_t),K(C),V(C)}
v_\theta(x_t,t,C).
\]

真正由 DiT 使用的**不是语言模型能输出什么句子**，而是 `C_j`（逐 token）及其与视频位置的交互。

给定只改变一个原子语义的最小对 `c^+,c^-`（例如拿起/放下），对第 `\ell` 层文字中间表征定义 `\delta C^{(\ell)}=C^{(\ell)}(c^+)-C^{(\ell)}(c^-)`。若最后送入视频的接口无法保留这个差异，那么下游视频模型就无法用这一接口区分两种条件。形式上，如 `\Phi(c^+)=\Phi(c^-)` 且其余输入与随机性相同，生成器输出条件分布必相同。**这一等价结论仅在条件接口真的相同的前提下成立**。

若差异尚存，视频局部响应的一阶近似为：

\[
\delta v\approx J_{v,C}\delta C,\qquad J_{v,C}=\frac{\partial v_\theta}{\partial C}.
\]

从这里产生两种不同的失败模式：

1. **Encoding bottleneck**：`\delta C` 对任务语义不敏感，或者只有 pooled 语义、丢失主客体/动作方向/先后信息。
2. **Consumption bottleneck**：`\delta C` 明确区分语义，但 `J_{v,C}\delta C` 的有效视频响应很小、集中在不相关帧、或方向错误。

这解释了为什么单纯堆叠更强 LLM、提高文字恢复 token accuracy、画注意力热图，都不一定提升视频。Kling 还强调融合层权重在扩散训练阶段可能漂移，使条件接口不稳定；动态条件也应检查相同 prompt 在相邻采样步的归一化和响应连续性。

### 文字去噪的真实自由度

本项目候选 `U_{t,s}=D_\phi(z_s,R(x_t),t,s)` 与纯静态编码 `C=\Phi(c)` 的不同在于：`U` 既要从**残缺原文**恢复语义，也可读取**正在形成的视频**并把误差反馈。它提供的价值要同时满足：

- 残缺文本 + 视频能对动作/关系形成更正确的证据（不能从已融合完整 prompt 的隐藏层偷看答案）；
- 该信息被有效写回到 DiT，且指向**原始目标**而不是只描述当前错误；
- 与无文字恢复损失的同结构 adapter、固定多层 TE、静态视觉语义计划相比，真正改善最终视频。

## 3. 第二种语义对象：图像/视频的可视化语义计划

Bernini、X-Omni 和语义 tokenizer 启示：文字的同一句最终答案 `c` 已经确定，但**目标视觉的语义实现**通常存在多个合乎指令的实现。

令 `S` 为与视频视觉状态相关的语义潜变量，例如目标视频的冻结 ViT patch/time 表示。训练数据中 `S^\star=E_{\rm ViT}(V^\star)` 是可从真实视频获得的 teacher。一个规划器学习：

\[
p_\psi(S\mid c),\qquad
v_\theta(x_t,t; C,S).
\]

这里的 `S` 不是要复原同一原文，而是表达**从语言语义到视觉实例**的中间决策。`S` 可以连续，也可以离散量化；不需要先训练新的视觉码本。真实视频的 `S^\star` 是**只在训练时合法的 oracle teacher**，推理时必须由 `p_\psi(S\mid c)` 或已有生成状态生成，不可作为公平基线的额外真值输入。

区别两类后训练任务：

- **text restoration**：从残缺 `c` 及 `x_t` 学到 “现有动作/关系能否支持原文”；
- **visual semantic prediction**：从 `c` 生成合理的视觉语义实现 `S`，类似 Bernini 的 masked ViT plan；

两者可以独立比较，也可能组合为“目标视觉计划 + 当前视频状态的重建误差反馈”。不应直接把 VQ 图像词表学习、完整 MLLM planner 和完整视频重训同时列为第一轮工作。

## 4. 安全的模块接口（不先扩建巨型统一网络）

对 Wan2.1-1.3B，最小语义接口应当保留：

\[
C_{\rm native}\in\mathbb R^{B\times L_{\rm T5}\times1536},
\]
\[
H_{\rm aux}\in\mathbb R^{B\times L_{\rm aux}\times d},
\quad
S_{\rm visual}\in\mathbb R^{B\times N_s\times d_s}\ (\text{可选}),
\]

以及 `A_{f,e}`（事件—latent 时间软对应）。原始 `C_{\rm native}` 不替换；附加状态先经过有界可训练 adapter 和零初始化末端残差进入 selected Wan blocks。不同 tokenizer 与 umT5 token index 必须对齐或通过事件单元作桥接。

值得优先比较的条件输入依次为：

1. **Native**：原生 umT5 条件；
2. **Static multilayer**：冻结 umT5/现成 VLM 多层逐 token hidden，以轻量 adapter 投入相同写入位置；
3. **Static visual semantic**：使用视觉语义 teacher 训练轻量视觉计划预测／写入，记录 oracle vs 推理预测差距；
4. **Dynamic text restoration**：原文恢复中间态经过同一容量的写入接口；
5. **Dynamic visual/semantic state**：只有前几个阶段证明可用时才探索与视频去噪步同步的更新。

按 **video semantic quality / GPU-hours / latency / VRAM** 判别，不把语言 perplexity、检索 cosine 或图文一致性单独作为选择标准。

## 5. 固定源码引用与公开资产（不是训练复现）

| 来源 | 版本与固定代码位置 | 本项目用途 |
|---|---|---|
| Wan2.1 official | [model.py L162–181](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L162-L181) | 原生 Q 来自视频、K/V 来自文本的已实现语义 |
| Bernini planner | [connector L51–99](https://github.com/bytedance/Bernini/blob/e6c2cf11843470b08491745947abcba9c3c6b079/bernini/models/bernini.py#L51-L99) | MLLM hidden 到渲染器/ViT 语义两个分支 |
| Bernini masked semantic | [L437–520](https://github.com/bytedance/Bernini/blob/e6c2cf11843470b08491745947abcba9c3c6b079/bernini/models/bernini.py#L437-L520) | 目标 ViT token mask、分离 decoder/renderer 输入 |
| Bernini-R 1.3B | [docs/bernini_r.md L1–19](https://github.com/bytedance/Bernini/blob/e6c2cf11843470b08491745947abcba9c3c6b079/docs/bernini_r.md#L1-L19) | Wan2.1-1.3B 的 **renderer-only** 权重，不含完整 planner |
| Bernini renderer training | [docs/bernini_r_train.md L45–55](https://github.com/bytedance/Bernini/blob/e6c2cf11843470b08491745947abcba9c3c6b079/docs/bernini_r_train.md#L45-L55) | 预提取冻结 ViT 与 VAE 特征，减少训练时重复计算 |
| MoT | [README L1–100](https://github.com/facebookresearch/Mixture-of-Transformers/blob/2ba51c05c9ebfdf0febb44c447a89ab24b9a36fb/README.md#L1-L100) | 模态专用 FFN/attention/norm 的稀疏分工 |
| X-Omni | [README L1–44](https://github.com/x-omni-team/X-Omni/blob/2b8237bb3789638c290eeda3e83ed81bd3652c3b/README.md#L1-L44) | 公开离散语义 prior + FLUX.1-dev decoder 的推理入口 |
| FLUX.3 | [HF Diffusers Technical Walkthrough](https://huggingface.co/blog/flux-3) | 具体 Qwen3-VL 8 层 hidden 及 modality-specific→joint blocks；**图像** |
| Kling | [Technical Report Section 3/6](https://arxiv.org/abs/2611.04916) | token-level precision、密集语义契约、融合稳定性及人评 |
| TokenFlow / SemHiTok | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_TokenFlow_Unified_Image_Tokenizer_for_Multimodal_Understanding_and_Generation_CVPR_2025_paper.html) / [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/49ffa271264808cf500ea528ed8ec9b3-Abstract-Conference.html) | 语义码本与细节码本的分层，不直接实施新 tokenizer |

注意：Bernini/Transfusion/X-Omni 等大多数包含预训练阶段；它们提供**设计原则与可冻结的 teacher／权重**，不能直接套用训练资源数字作为 Wan 的后训练预算。代码资产许可证需要另行按用途检查。

## 6. 研究结论与待回答问题

- **更准确的抽象**：原构想真正想引入的是**条件语义的计算与逐步修正能力**，不一定要由“输出字词 token 的扩散”唯一实现。
- **必要基础**：先检查原生 umT5 对原子动作/关系的可区分性与 DiT 的消费位置；Kling/FLUX.3 提示 **静态的多层逐 token 特征** 可能就是强基线。
- **强竞争路线**：Bernini/X-Omni 提示 **先合成可视化语义，再由扩散渲染细节**；目标不是某个离散 token ID 本身，而是预测视觉可使用的条件。
- **方法空间**：本项目最值得检验的是原文恢复是否能让视觉语义证据更精确地反馈到 Wan，及其相对于静态多层条件与静态视觉计划是否有独立收益。
- **新颖性边界**：双模态扩散、token 视觉词表、MLLM→DiT connector、多层静态编码均有直接先例。需要用视频侧因果实验确定真正贡献。

实验顺序见 [08_condition_interface_experiments.md](08_condition_interface_experiments.md)；原计划 [06](06_operator_experiments.md) 保留，作为具体对照实施资产。
