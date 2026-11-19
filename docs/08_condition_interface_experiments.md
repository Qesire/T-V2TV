# 08 · 更新后的实验顺序：先测语义传输，再评价共同去噪

**修订理由：** [Kling / FLUX.3 / Bernini / X-Omni 文献对照](07_semantic_interfaces_and_literature.md) 表明，文本到视频的核心瓶颈不只来自文字是否会更新，还可能是**原子语义在 text encoder → Wan 条件 → attention → 视频帧轴中丢失、混淆或不被有效消费**。本计划作为 [06](06_operator_experiments.md) 的前置决策层；不废弃已定义的 E0–E3，也不启动全模型预训练。所有实验仍 **UNVERIFIED**。

## A · 几个具有不同预测的竞争机制

| 机制 | 若成立，最早应观察到什么 | 经济的下一步 |
|---|---|---|
| H_enc：最后一层文本编码丢失动作细节 | 正反事实 c± 在部分中间层明显可区分，到原生最终条件后动作/主体/时序差异弱 | 缓存 umT5 多层逐 token hidden，训练**小型**多层融合/残差映射；不改动 umT5 预训练权重 |
| H_consume：文本信息尚在，但 DiT 未正确消费 | 条件序列可区分 c±，但相同噪声视频的速度差与最终动作无对应，或集中在错误帧/位置 | 控制 cross-attention 的有效写入和事件—latent 时间路由 |
| H_visual：需要从文字先推断视觉语义实现 | 静态文字条件已经清楚，额外 ViT 目标语义（训练时 oracle）能让**训练好的小型接口**明显改善视频；预测的视觉语义可部分继承收益 | 研究静态视觉语义计划，参照 Bernini；暂不训练新 VQ codebook |
| H_reconstruct：恢复原文能从视频中学出纠错状态 | 同一遮蔽文本配不同动作视频，原文恢复分数变化方向正确；后训练恢复损失使视频目标更好 | 文字恢复 + 视频写入的同结构消融 |
| H_dynamic：逐采样步联合文字演化不可替代 | 同 checkpoint 的动态更新，超过输入/计算匹配的静态或重复计算文本状态 | 训练两个噪声时钟并做文字更新频率/修订机制研究 |

每条假说都可能独立成立、相互补充或失败，不预先宣称哪一个是“原因”。

## B · 新增 E−1：原生文字语义传输审计（不训练）

### B1 输入与对照

复用 [operator_pairs_v0.json](../experiments/prompt_sets/operator_pairs_v0.json) 的动作／主体／时序反事实 prompt 对，优先取不改变实体与场景的 c±。同一对共享视频初始噪声、原生 Wan 权重、调度器时间、负面 prompt 和 CFG。

在 umT5 的可访问层记录**保留 token 索引及注意力 mask 的逐 token**隐藏状态 `C_\ell(c)`；必须记录子词对齐、固定 prompt 与反事实之间受影响 token 范围。与 pooled 表示对比，但不能以 pooled 结果代替 token-level 条件。

### B2 两道“信息是否传达”关卡

**第一道：编码差分**

\[
\delta C_\ell=C_\ell(c^+)-C_\ell(c^-).
\]

计算各层相关 token 的标准化差异、跨层分布及一个冻结 probe 的动作方向辨识。`\|\delta C_\ell\|` 大**不等于**语义可被读取；必须比较最小对上任务正确的可分性，而不是任意向量范数。tokenizer 对齐不相同时用事件 span，而非逐位置作未经确认的减法。

**第二道：视频消费差分**

固定相同 `x_t`，计算 `\delta v_t=v_\theta(x_t,t,c^+)-v_\theta(x_t,t,c^-)`，再按 latent 视频时间 `f` 和空间区域 `p` 分组，记录：

\[
D_{t,f}=\|\delta v_{t,f,:,:}\|_2,\qquad
\Delta_{t,f}=\frac{\langle \delta v_{t,f},d^{\rm target}_{t,f}\rangle}{\|\delta v_{t,f}\|\|d^{\rm target}_{t,f}\|+\varepsilon}
\]

其中 `d^{\rm target}` 仅在有真实视频、清楚速度目标且坐标相容时定义；对普通 prompt-only 生成不伪造 `d^{\rm target}`。另评估最终视频动作和顺序，**不要把 D>0 当作性能证据**。

如需估计 token 贡献，可作选定 token 的干预（替换其 K/V、对应层屏蔽、或小扰动 JVP），但每种干预都要区分输入分布外效应与原始 prompt 反事实。第一轮仅选择少量 block/step，不显式存全注意力图。

### B3 结果分支

- **编码能力不足** → 优先验证**静态多层逐 token**映射；Kling 和 FLUX.3 给出直接参考。
- **编码能力已有但消费不足** → 优先验证**时间/对象路由 writer**，接现有 [E1](06_operator_experiments.md)。
- **编码与消费都能区分但最终视频动作错误** → 重点检查训练目标与时序实现；可能需要视觉语义计划或生成端后训练。
- **本来就能完成动作** → 选择更有区分力的长动作与对象交互，不在容易案例上拟合局部指标。

这个前置 probe 的 GPU 成本可共享原生 Wan 的条件缓存和去噪轨迹；不必先从头运行完整联合模型。

## C · 最小表示实验：在同一个视频写入接口上对比语义来源

令 `H^\ell_v` 是目标 Wan block 的视频隐藏状态；统一低秩写入器

\[
\tilde H^\ell_v=H^\ell_v+\gamma_\ell(t)\,W_\omega(S_{\rm cond},f,p)
\]

必须**初始等于原生 Wan**，并在每一组维持相近可训练参数和视频监督预算。不同条件输入通过各自的投影统一到 `d_a` 宽度：

| 组 | 条件来源 | 是否跨视频去噪步更新 | 额外训练或预提取 |
|---|---|---|---|
| N | 原生 Wan + 原始 umT5 | 否 | 无 |
| M | 冻结多层 umT5 / 可用 VLM 的逐 token hidden | **否**（虽然网络层间也可变） | 小型映射；冻结 text backbone |
| S_oracle | 训练视频的冻结 ViT 语义 embedding | 否 | **仅训练时上界**；预提取真实视频特征并训练适配器 |
| S_pred | 从 prompt 预测的视觉语义状态 | 规划阶段可迭代，但视频去噪时先固定 | 与 S_oracle 相同 renderer，训练小型预测头/adapter |
| T_static | 原文 mask 恢复中间特征，仅计算一次 | 否 | 与 T_dynamic 同样恢复任务与参数 |
| T_dynamic | 原文 mask 恢复中间特征 + 当前视频输入 | **是** | 两个噪声时钟、文字恢复 loss、视频写入 loss |

**重要比较边界**：

- `S_{\rm oracle}` 额外使用真实目标视频的信息，只是**可学接口的 ceiling/probe**，不能宣称它是公平 T2V 生成对照；`S_{\rm pred}` 才是生成时可用的方案。
- `M` 需要训练小型映射，**不能直接把 Qwen-VL hidden 塞进 Wan 原先 T5 K/V**。
- `T_{\rm static}` 与 `T_{\rm dynamic}` 的公平性还依赖算力：除了固定一次，还要有重复执行但不利用新视频信息的计算匹配对照。
- 对于 S_pred，若目标 ViT teacher 使用视频帧特征，必须保存真实视频时间坐标；pooling 成单个全局 vector 后无法直接监督事件时段。
- 先比较 N、M、T_static 与 T_dynamic 是否存在可读/可控空间；若较明显地需要视觉语义先验，再训练 S_oracle/S_pred。**不同时开六个训练分支**。

## D · 图像/视频语义词表是否现在就应建设？

**不作为当前第一轮里程碑。** TokenFlow、SemHiTok 和 X-Omni 已经说明如何学习语义与像素的视觉 token。重新训练 tokenizer 将扩大数据、训练成本，也无法直接判别“原文恢复是否改善视频”。

优先用冻结 ViT / SigLIP2 的**连续** patch/time 特征，建立 `S^\star=E_{\rm vision}(V^\star)`；通过轻量 connector 测试**有意义的视觉语义表示是否能到达 Wan**。只有连续表示的瓶颈确实来自序列长度、存储、语义/细节纠缠，才开启 VQ/codebook 对照。X-Omni 报告的 RL 是图像生成离散语义 token→diffusion decoder 的对齐训练，不能直接照搬到 Wan 视频。

## E · 多层条件与动态条件的共同稳定性

Kling 报告指出：多层 text hidden 的可训练融合，在视频 DiT 训练目标跨分辨率/噪声阶段变化时可能漂移；FLUX.3 证明保留多层 token 语义并不要求让文本本身随视频去噪。由此新增两个追踪量：

\[
\Delta_{\rm cond}(t_i,t_{i+1})=
\frac{\|S_{i+1}-S_i\|_F}{\|S_i\|_F+\varepsilon},
\qquad
\Delta_{\rm out}(i)=\|\delta v_{i+1}-\delta v_i\|.
\]

`\Delta_{\rm out}` 在不同 `x_t` 位置比较时不是严格相同状态 Jacobian，必须同时保留对应的 `x_t` 和 step；更严格的条件改变实验在**固定同一个 `x_t`** 下替换 `S_i`/`S_{i+1}`。未证明动态条件增益之前，避免依赖随步数任意变化的巨幅残差。

另必须区分三类“状态变化”：(1) Transformer **层深**变化；(2) 视频 **采样去噪步**变化；(3) 视频 **帧时间轴**变化。FLUX.3 在联合 blocks 内更新文本隐藏状态，但文字条件在不同去噪调用之间从同一个输入重新计算，不等于「跨去噪步携带文本采样状态」。

## F · 优先级、资产与停止条件

1. **P0（冻结模型）**：B1/B2 的编码和视频消费诊断；保留同 prompt/seed 的完整视频，并与现有 E1 语义差分探针共享缓存。无模型训练。
2. **P1（小型后训练）**：如果多层文本含有更清楚的动作语义，先做 N vs M；如果消费接口主导，先做 N vs 相同静态条件的轻量 writer。不得只用 text-only retrieval 得分选方案。
3. **P2（文字重建的增量）**：在同一 writer 上做 T_dynamic（有恢复 loss）vs 同结构无恢复 loss，并对照 T_static。这才回答起始研究假说。
4. **P3（语义视觉 prior）**：需要时把 Bernini 的连续 ViT teacher 作为训练先验，比较 S_oracle→S_pred 的落差；仅在 P2 有明确不足或潜在收益时安排。

### 实验产物应含

```text
experiments/runs/<id>/
  token_semantic_diffs.jsonl     # prompt pair, token spans, encoder layer, c± gap
  velocity_semantic_diffs.npz    # same x_t/step, c± velocities, latent time indexing
  condition_adapters/           # adapter cfg/weights, version, trainability
  visual_semantic_targets/      # frozen ViT features + source/time mapping (if used)
  recon_traces.jsonl            # masked target, text states, visual dependence
  videos/                       # paired mp4 for manual judgement
  manifest.json                 # source sha/checkpoint/solver full state/seed/precision
```

报告 video-side 事件完成、顺序、主体/动作绑定、局部物体连续性和画质；独立统计单位是 prompt 或原始视频来源，多 seed 为嵌套重复。正式确认集须在观察 pilot 后重新冻结，不以先验未冻结阈值宣称已通过。评估、训练、运行参数尚未验收，状态 **UNVERIFIED**。

## G · 最重要的决策变化

以前的方案默认“文字扩散是主方法，语义对齐服务于它”。经过这一批论文，现在应该允许**强静态条件与视觉语义先验胜过文字扩散**。若 T_dynamic 赢得真实视频收益，就继续研究其独特的**跨采样步语义纠错功能**；若 M 或 S_pred 更有效，就把工作转成更准确的语义传输/视觉语义计划，而不是为了原先的想法强行加入文本扩散。
