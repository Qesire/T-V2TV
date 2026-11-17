# 06 · 基于算子判别的实验方案

状态：**EXPERIMENT DESIGN / STATIC_ONLY**。目标模型是 Wan2.1 T2V-1.3B；只考虑参数高效后训练。此文细化 [04](04_experiment_logic.md)，并将研究目标拆成三个独立的可测机制：视觉读取 R、视频控制 W、联合重建的额外收益 G。没有已完成的视频运行或训练结果。

## 1. 科学对象与复合雅可比

原文 `c` 始终提供给原生 Wan：`v_0(x_t,t,c)`。另将 `c` 腐化成 `z_s`，通过当前视频状态获得恢复中间表示：

\[
U_{t,s}=D_\phi(z_s,R_\psi(x_t),t,s),\qquad
\widehat v_t=v_0(x_t,t,c)+\Delta v_\omega(x_t,c,U_{t,s}).
\]

其中 `t` 是视频去噪进度，`s` 是文字去噪进度，`f` 是视频时间位置，三者严格区分。局部闭环灵敏度是

\[
J_R=\frac{\partial U}{\partial x_t},\quad
J_W=\frac{\partial \widehat v}{\partial U},\quad
J_{\mathrm{loop}}=J_W J_R.
\]

`J_R\ne0` 只证明文字表示响应视觉，`J_W\ne0` 只证明视频预测响应文字；二者非零既不保证纠错方向，也不保证最终视频动作改变。真实视频生成仍是终端验证。 **这里的雅可比只在固定离散文字状态、对连续隐藏表示求导时成立；离散 token 重采样有跳跃，不能用常规可微链式法则跨越采样步骤。** 对完整文字采样链只能做有限差分、受控干预或指定的松弛估计。

在训练视频加噪 `x_t` 上可另记录目标残差 `d_t=v^\star-v_0` 与新增修正 `\Delta v` 的对齐：

\[
a_t=\frac{\langle d_t,\Delta v_t\rangle}{\|d_t\|\|\Delta v_t\|+\varepsilon}.
\]

它是局部速度目标代理，不是视频质量。原文 `c` 已知，`z_s` 的独立噪声不会给理想条件模型增加数据；必须以相同 adapter / 计算预算的非重建基线确认增益。

## 2. 正确的物理和算子索引

官方 [Wan2.1 T2V-1.3B 配置](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/wan_t2v_1_3B.py) 规定 VAE stride=(4,8,8)、patch=(1,2,2)、隐藏维度 1536、30 层。视频 tensor `x_t\in[B,C_z,F_z,H_z,W_z]`，patch 后 `H_v\in[B,F_z H'W',1536]`。

建议探索性配方：832×480、49 像素帧、50 步 UniPC、shift=5.0、CFG=5.0（后四项来自 [原生 T2V 推理](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L114-L164) 的实际默认）。49 帧满足 `4n+1`，对应 `F_z=(49-1)/4+1=13`；`H'=480/(8*2)=30`、`W'=832/(8*2)=52`，总视频 tokens 为 `13*30*52=20,280`。确认实验再使用官方默认 81 帧；此方案还未实际验证 GPU 显存或速度。

全部对照固定初始视频噪声、原文、negative prompt、原生 checkpoint、solver、调度器、CFG。Wan UniPC 是**多步求解器**：从中间步分叉必须恢复包括历史输出在内的完整 scheduler 状态，而不仅复制 `x_t`；否则反事实比较不成立。

## 3. E0：CPU 算子验收先于 Wan 集成

通过 `python -m unittest discover -s tests -p "test_*.py"` 先验证：

1. 零初始化低秩 writer 在第 0 步严格保持 `H`（同 shape、dtype）；真实 Wan hook 另测浮点容差。
2. 事件—帧分配 `A_{f,e}` 在交换时间行时产生对应时间位置的输出交换；全局无序平均单独作为失序对照。
3. 只将低秩输出矩阵 `B` 置零、输入矩阵 `A` 保持非零：初始 `\partial L/\partial B` 可非零，首次更新 `B` 后 `\partial L/\partial U` 才可能非零。**不要同时把 `A/B` 和门控都零初始化**。
4. 冻结 Wan 参数使用 `requires_grad=False`；新增写入模块之后仍需梯度图。不能照搬 Wan 推理脚本里的 `torch.no_grad()`。
5. `E=0`（无事件）明确回退原始 `H`；事件 span 根据文字模型 tokenizer 而非 umT5 token 位置构造。

这些单元测试不等于已运行的真实模型或能力实验。

## 4. E1：先用不训练的方法验证视频可控性

使用 [`experiments/prompt_sets/operator_pairs_v0.json`](../experiments/prompt_sets/operator_pairs_v0.json) 的动作对 `(c^+,c^-)`。两个 prompt 只在动作/方向/时序上不同。在**同一个视频状态和去噪步**分别运行 Wan，得到：

\[
v^+=v_\theta(x_t,t,c^+),\quad
v^-=v_\theta(x_t,t,c^-),\quad
v_{\rm cfg}=v_\varnothing+g(v^+-v_\varnothing).
\]

在指定的**latent 时间位置**注入条件速度差分：

\[
\boxed{\widetilde v_{k,f}
=v_{{\rm cfg},k,f}+\alpha\,m_{k,f}\,g\,(v^-_{k,f}-v^+_{k,f})}.
\]

`m_{k,f}` 是视频去噪步 `k` × latent 帧位置 `f` 的路由掩码；`g` 为当前 CFG scale。这个训练自由探针只检验**采样器是否能在选定时间位置响应已知语义方向**；它既不是文字重建算法，也不证明 block 内的轻量写入已可控。

必须比较：
- 原生无注入；
- 全帧均匀差分；
- 时间位置受控差分；
- 颠倒/错配时间位置的差分。

报告实际注入 `L_2` 范数，尽可能将受控与均匀方案范数匹配，排除只是幅度更大的解释。

**时间传播算子**：在某个 latent 时间位置 `r` 注入后，继续运行一个未受干预的后续采样步，记录 `M_{f\leftarrow r}=\|v^{\mathrm{perturbed}}_{k+1,f}-v^{\mathrm{base}}_{k+1,f}\|_2`。这比注入当步的时间局部性更有信息：注入步在掩码外人为为零，但后续 self-attention/solver 可能把影响传播到其他时间位置。记录 `M` 的时间分布，并对照最终视频中的事件变化，不能把 `M` 本身当作语义正确性。首先取 4 对 prompt ×1 seed，扫描两个早/中期窗口，保留完整 mp4；必要时才扩大到晚期。这里的“早/中/晚”按**采样索引**，不直接使用 scheduler 下降的整数时间戳。

**结果判别**：只有人眼在最终视频中识别到预期动作或顺序变化，才算获得可控性证据。`\|\Delta v\|>0` 和局部注意力变化都不够。若未改变，先查采样器重放、注入坐标、幅度及 prompt 差分是否真实具有所需语义，而不是先训练文字支路。

## 5. E2：冻结视频骨干，只训练文字视觉读取器

优先用英文 masked diffusion 参考模型 [MDLM](https://github.com/kuleshov-group/mdlm) 的吸收态 mask 恢复，保留非扩散静态文本恢复作匹配对照。利用 [Something-Something V2 官方研究数据](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) 中“左→右/右→左、上/下、靠近/远离、真实完成/尝试但未完成”等可反事实动作。获取数据须符合原有 Research Use 许可；不上传原视频。小范围探索性建议先从真实数据中选择约 200–400 train、80–120 val，按原视频来源与场景隔离；这一规模不支持顶会级确认性结论。

对同一已遮蔽文本 `z` 和两种真实视频 `V^A,V^B`，训练模型产生 logits。两个未遮蔽上下文必须**完全一致**，且动作/关系差异中的全部 subword token 一起 mask。定义：

\[
D(V,z)=\log p(c^A_{\mathcal M}\mid z,R(V))
      -\log p(c^B_{\mathcal M}\mid z,R(V)),
\]

\[
\boxed{\Delta_R=D(V^A,z)-D(V^B,z)}.
\]

`\Delta_R>0` 是动作反事实的预期方向。实际 masked-token CE 应在相同 mask 与长度归一后计算。实验包含**text-only、视频交换、帧序打乱、错误可见 prompt 的故意泄漏对照**。文字 head 不能直接看到完整目标 `c`，也不要从已经注入正向干净 `c` 的深层状态冒充纯视觉观测；`detach` 不会消除信息泄漏。

把真实视频 VAE latent 加噪后的教师状态与原生 Wan 实际生成的 `\hat x_t` **分别**测试；扫 3 个视频进度 ×3 个文字 mask 率，记录 `\Delta_R(t,s)`，不要先假设 `t=s`。预先恢复原文的任务如果只凭语言就猜对，则这个机制不通过视觉读取测试。

**推进规则**：文字 head 在不泄漏条件下不能区分相反动作时，先改读取位置、视频时序表示与 mask 单元；不训练反馈 writer。

## 6. E2.5：语义纠错方向的低成本检查

文字重建器可构造目标能量：

\[
E_c(x_t,z_s)=-\log p_\phi(c_{\mathcal M}\mid z_s,R(x_t),t,s).
\]

`-\nabla_{x_t}E_c` 给出了一个**候选**视频状态修正方向。对少量 Wan 采样状态比较：目标原文的重建梯度、反事实文字的重建梯度、等范数随机梯度与不加梯度。先核对 solver 的更新符号、步长和梯度所在变量，不能把 `-\nabla E` 无条件当作视频 flow 速度。该探针有助于区分“恢复头可以读懂当前错误”与“它提供了正确纠错方向”，但昂贵且只用于小样本机制验证。

训练数据加噪状态还监控 `\cos(\Delta v,v^\star-v_0)`，以及针对**同一个可训练文字表示 `U`** 的 `\cos(\nabla_U L_{\rm text},\nabla_U L_{\rm video})`。这些是局部代理，**完整视频结果为准**。

## 7. E3：最少后训练因子对照

E1/E2 取得足够局部证据后，才开始有成本的视频侧后训练。共享相同 adapter 容量、数据、训练步数和采样设置：

| 组 | 文字表示／更新 | 视觉读取 | 文字重建目标 | 测量含义 |
|---|---|---|---|---|
| B0 | 无 | 无 | 无 | 原 Wan |
| A | 普通辅助表示 | 与实验接口匹配 | 无 | 额外参数及条件接口本身 |
| B | 迭代 mask 恢复 | 关 | 有 | 文字先验和迭代计算是否已经够用 |
| C | 迭代 mask 恢复 | 开 | **有** | 候选方法 |
| E | 与 C 同结构 | 开 | **无**（另训） | **原文重建监督**相对仅视频目标的贡献 |
| D | C 的权重，推理时固定一次文本状态 | 开/与 C 匹配 | 沿用 C 训练 | **持续更新**相对固定表示的增量 |

**先训 E 与 C，直接判别中心假说**；有增益再补 B、A 与 D。D 可能受到训练/推理状态分布改变的干扰，确认时应训练覆盖静态状态的对应对照。对于 C 与 D，分别报告视频质量与文字更新计算成本，不以更多 forward 自动证明“扩散有益”。

相对于已有时序控制 [TSG](https://arxiv.org/abs/2610.20822)，在确认集上保留可复现的强基线。TSG 输入显式时间约束时拥有不同的标注信息，比较须注明额外信息预算；不要与完全自动从原文恢复的方案混作公平条件。

## 8. 可读窗口 × 可控窗口：实验是否值得继续的核心判据

E2 估计 `R(t,s)`：当前状态能否提供动作/时序差异；E1 估计 `W(t)`：该时刻的语义干预能否改变最终动作。研究的有效联合区间为

\[
\Omega=\{(t,s):R(t,s)\text{ 可区分动作且 }W(t)\text{ 可改变目标事件}\}.
\]

在 `\Omega` 有可观测交集后，再决定 `s=g(t)`、文字更新频率和训练时的噪声水平联合分布。若高噪声时可控但不可读、低噪声时可读但不可控，可考虑“语义先验提前形成、视频证据后续修正”或只保留训练期重建监督；不能据此宣称闭环已成立。

## 9. 完整视频判读和统计

目标评价：事件完成／缺失、A→B 顺序、对象与主体绑定、接触状态、背景与视觉稳定性、额外 GPU·小时／显存／每视频前向次数。自动帧级 VLM 仅作辅助，完整 mp4 必须保留供人工盲评。

- 首轮 4 对×1 seed 和 3×3 噪声窗口只做**探索性筛查**。
- 正式确认前冻结独立 prompt 集、checkpoint、adapter 配方、目标事件标准及统计方式，再用多个配对 seed。24–32 条独立 prompt ×2–3 seed 是一个预算起点，实际样本量需依据 pilot 变异与最低有意义效应确定。
- 独立单位是 prompt／视频来源；同一 prompt 多 seed 和逐帧标签是相关观测。报告配对差与按来源聚类的置信区间，记录质量退化和失败案例。

**停止规则**：R 失败→先修视觉读取；W 失败→先修视频控制通路；C≈E→原文重建未显示独立收益；C≈D 且更贵→无证据表明持续文字去噪必要；训练状态成功而生成状态失败→优先修复 exposure gap。所有成功与失败均按真正执行的范围报告。

## 10. 实验资产与源绑定

- 源码绑定：[Wan](03_code_references.md)、[Uni-ViGU / MDLM](03_code_references.md)，所有固定上游 commit 见 [机器索引](../references/code_sources.json)。
- Prompt 对：[operator_pairs_v0.json](../experiments/prompt_sets/operator_pairs_v0.json)，均为 **prompt-only**；尚未配真实视频。
- 基座轨迹 manifest：记录视频 latent、完整 solver 历史、prompt/negative prompt、noise seed、模型 checkpoint SHA 和 runtime。
- 文字诊断：按样本保存 mask、动作候选 token log probabilities、视频时间索引、文字 mask 率及读取特征来源。
- 干预视频：每个基线／差分／时间路由条件保存独立 mp4 与人眼评语。
- 后训练：保存相同结构消融组的训练配置、梯度冲突统计、运行时开销及模型差分权重。

**尚未运行**：Wan 实际视频生成、训练／验证数据下载、文字模型视觉适配和联合视频后训练。下一步只需完成 E0 与 E1 的最小代码绑定与 GPU preflight，不进入完整训练。
