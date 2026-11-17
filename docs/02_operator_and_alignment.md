# 02 · 文字去噪、视频语义对齐与可后训练算子

本文件是候选算子空间与第一版推荐契约；**不是 Uni-ViGU 源码的逐行重写**。视频骨干使用 Wan2.1 T2V-1.3B；条件 `c` 是固定的用户原文，文字扩散只恢复 `c` 的被破坏版本。设计优先从轻量接口识别机制，再考虑共享整层注意力。

## 1. 文字去噪：三种必须区分的实现

| 路线 | 随时间变化的状态 | 训练目标 | 优点 | 关键限制 |
|---|---|---|---|---|
| **D1 离散 masked diffusion** | token ID `z_s\in\{1,\dots,V,\mathrm{MASK}\}^L` | 预测被 mask 的原始 token，含 `p_\phi(c_j\mid z_s,R(x_t),s,t)` | 词面恢复可直接验收；MDLM 有成熟实现 | 自然语言可由上下文猜出；单调解除 mask 不支持已确定词的修订；需要视频条件适配 |
| **D2 连续 embedding flow** | `z_s\in\mathbb R^{L\times d}` | `\|v_\phi-(E(c)-\epsilon)\|^2`，另验收词表投影还原 | 可端到端连续反传，视频/文字形式一致 | embedding 回归小不等于语义正确；解码与表示空间必须对应；预训练知识的迁移另需解决 |
| **D0 非扩散辅助表示** | 静态 masked encoder 或单步语义表示 | 与 D1 匹配的 masked-token CE | 成本更低、是必须有的对照 | 不能支持“迭代文字去噪有独立收益”的主张 |

**建议先 D1（mask 重建），同时保留 D0 作公平对照。** D2 可作为针对连续梯度及算子耦合的第二候选；不要把二者混称“离散文字流”。已有 Uni-ViGU **论文**写了高斯插值的 token embedding flow；**公开推理脚本**却是 token ID 的随机采样与按进度覆盖。二者不是同一个采样转移核，见 [源码引述](03_code_references.md)。

D1 的基本腐化：

\[
z_{s,j}=\begin{cases}
c_j,& B_j(s)=0,\\
\mathrm{MASK},&B_j(s)=1,
\end{cases}
\qquad P(B_j(s)=1)=m(s).
\]

`m(s)` 随文字恢复进度减小；具体权重与采样核必须与选定模型保持一致。MDLM 的 `q_xt` 和 `_ddpm_update` 是可核对的参考，不应把 Uni-ViGU 的“概率随步数增大而覆盖”直接当成 MDLM 的正确反向转移。

**遮蔽的统计单元**优先是完整事件短语和动作／方向／次序短语，另包含随机 token mask 作基线。保留 token 级目标 `c_j`；完整动作短语内的若干 subword token 一起遮蔽。英文 MDLM 权重能否迁移到中文 prompt 未验证，最初的机制测试优先让文字模型语料与测试语言一致。

## 2. 端到端表示与张量边界

Wan 原始视频输入是 VAE latent：

\[
x_t\in\mathbb R^{B\times C_z\times F_z\times H_z\times W_z}.
\]

Wan `patch_size=(1,2,2)`，其视频 hidden state 扁平化后可写：

\[
H_v^\ell\in\mathbb R^{B\times N_v\times D},\quad
N_v=F_z\,H'\,W',\quad D=1536.
\]

这里 `F_z` 是**压缩后视频 latent 的时间长度**，不是输出像素帧数；`H',W'` 是 patchify 后空间尺寸。原始模型的 `grid_sizes` 含 `(F_z,H',W')`，可用于恢复时间索引；不要把全部 `N_v` token 当成同一帧。

设文字模型 tokenizer 长度为 `L`，动作／事件单元数为 `E`：

\[
Z_s\in\mathbb N^{B\times L},\quad
U_{t,s}\in\mathbb R^{B\times L\times d_u},\quad
M\in\{0,1\}^{B\times E\times L}.
\]

`M_{e,j}` 表示 tokenizer 位置 `j` 是否属于事件 `e`；**它与 Wan 的 umT5 token 长度通常不相等**。先在本身 tokenizer 下建立事件 span，再由可训练映射将事件表示送入视频通道，不进行错误的 token 索引复用。

事件读出可定义为：

\[
U^{\rm evt}_{b,e,:}=
\frac{\sum_jM_{b,e,j}U_{b,j,:}}
{\max(1,\sum_jM_{b,e,j})}
\in\mathbb R^{B\times E\times d_u}.
\]

语义事件至少包括动作谓词、参与主体、客体、方向、显式先后／持续关系；识别不到明确动作时允许 `E=0`，回退原 Wan，而非伪造事件。

## 3. 视觉读取：必须携带时间、避免原文复制

建议先从 **Wan 第一次正向文字 cross-attention 之前**的视频 patch 表示中提取，或使用没有注入该正向 prompt 的专门 probe 路径。前者比已融合完整 prompt 的深层表示更容易判断视觉证据来自哪里，代价是早期特征可能缺少语义。

一种低成本读取器：

\[
R_\psi(H_v)_{f,:}=
\operatorname{Pool}_{p}\big(P_\psi H_{v,f,p,:}\big)
+P_{\rm time}(f),
\]

\[
U_{t,s}=G_\phi(Z_s,\ R_\psi(H_v),\ s,t).
\]

`Pool` 可以是带空间位置的可学习聚合；即使池化后只有 `F_z` 个时间 token，位置也必须保留。只对独立逐帧表征做无序平均 `F^{-1}\sum_f r(x_f)`，会与帧排列交换，**从构造上无法区别 A→B 和 B→A**。

对所有视觉读取配置必须记录：`hook block`、`pre/post CA`、是否带干净 `C`、时间与位置编码、张量 shape／dtype。`detach` 不会删除已编码的原文。

## 4. 时间对齐：把语言去噪状态映射到视频的帧轴

令 `Q_{f,:}` 是从视频第 `f` 个 latent 时间位置读出的状态，`K_{e,:}` 是事件 `e` 的文字恢复中间表示，定义：

\[
S_{f,e}=\frac{(Q_fW_Q)(U_e^{\rm evt}W_K)^\top}{\sqrt{d_a}},
\qquad
A_{f,e}=\operatorname{softmax}_{e}S_{f,e}.
\]

`A\in[0,1]^{B\times F_z\times E}` 是每个视频 latent 时间对应哪个事件的软对齐，不是视频去噪时间。允许同时发生的事件时可改用多个 sigmoid 门；`softmax` 仅适合竞争的互斥事件。未来若要支持帧内多个主体，扩展为 `A_{f,p,e}` 或事件—主体联合索引。

若 prompt 明确规定事件先后 `e_i\prec e_j`，可用弱排序监督：

\[
\mu_e=\frac{\sum_f f A_{f,e}}{\sum_f A_{f,e}+\varepsilon},
\qquad
\mathcal L_{\rm order}
=\max(0,\delta+\mu_{e_i}-\mu_{e_j}).
\]

它只对明确有先后关系的成对事件使用；**不凭空固定每段具体起止时刻**。同一对象的连续身份、接触关系、同时发生的事件，需要额外观测或专门标注，不能用单一排序损失覆盖。

在扩散步上演化的文字状态 `U_{t,s}` 和在帧轴上的 `A_{f,e}` 是两个操作：即使先恢复动作词 A，也不意味着 A 必须出现在早期视频帧。

## 5. 优先推荐的轻量时序写入算子

最早版本保留 Wan 所有原始权重和原始文字条件 `C`，在选定层的 cross-attention **之后**加一个时间受控残差：

\[
\Delta H^\ell_{b,f,p,:}
=
\gamma_\ell(t)\;
\sum_{e=1}^{E}
A_{b,f,e}\,
W^\ell_OU^{\rm evt}_{b,e,:}.
\]

\[
\widetilde H^\ell=H^\ell+\Delta H^\ell,\qquad
\hat v=F_{\theta_0,\omega}(x_t,t;C,U_{t,s},A).
\]

`W^\ell_O` 可以低秩分解，`\gamma_\ell` 由标量／小型门控提供，残差投影的最后一层零初始化，使第一步等于原 Wan。**写入点和后续原生 Block 仍需计算梯度传递，但冻结的底座权重不更新。**

这个接口的工程优点是不必构造 `N_v\times L` 的显式注意力 bias，也不必立即修改 FlashAttention 内核；它直接给不同帧不同的写入。第一批写入层的选择要由功能探针确定，不默认全 30 层。

**必须设置对照**：共享 Value-only `\Delta H_{f,p}=\sum_j\Pi_{f,p,j}\Delta V_j`。如果两个帧位置的 `\Pi` 相等，则当前层的写入相等；时间门控版本则可以通过不同 `A_{f,e}` 显式区分帧。后续非线性层可以转换共享 Value 的作用，因此这不是对 Value-only 的全网络不可能性证明。

候选扩展是直接给原有 cross-attention logits 加 `\beta_{f,j}`；它能改变 attention allocation，却可能要求改动底层注意力核。仅在残差路由的因果探针显示必须控制原生路由时再推进。

## 6. 训练目标及梯度路径

视频使用与当前官方实现一致的流匹配速度目标 `u_v^\star`（先校验 scheduler 的方向和尺度），文字使用原文 masked-token CE：

\[
\mathcal L_v=\mathbb E\|\hat v-u_v^\star\|^2,
\quad
\mathcal L_{\rm rec}
=\mathbb E\sum_{j\in\mathcal M}-\log p_\phi(c_j\mid z_s,R(x_t),s,t).
\]

第一版总损失：

\[
\mathcal L=\mathcal L_v+\lambda_r\mathcal L_{\rm rec}
+\lambda_o\mathcal L_{\rm order}
\]

其中 `\lambda_o=0` 作为最初单因素测试，随后仅在确认时序路由有效时激活排序项。具体数值不预设为最终方案；需记录每项梯度范数与夹角，判断文字重建是否与视频优化冲突。

训练采用噪声配对 `(t,s)` 的单步监督，先不反传整条采样链。**保证** `\mathcal L_v\rightarrow\Delta H\rightarrow U\rightarrow G_\phi` 的梯度可达；不要将写入后所有视频块包在 `torch.no_grad()` 中。全骨干冻结仅指 `requires_grad=False`。

## 7. 三个实现边界

1. **条件可见性**：原文 `C` 给视频骨干，文字预测头只能通过 `z_s` 和受控视觉读取路径求解；如果需要写入端比较目标与现状，`C` 可在写入端提供，但不要让完整目标复制到文字预测 head。
2. **训练—推理一致性**：训练从真实 VAE 视频加噪的 `x_t` 读取；采样时从模型自己的 `\hat x_t` 读取。必须单独测试推理轨迹中 R/W 是否仍然有效，避免在 teacher-forced 状态下验证成功却推理失败。
3. **计算成本**：全局联合注意力在 `N_v\gg L` 时仍需视频 token 的二次注意力及与文字的交叉项，文本专家每层复制完整 FFN 成本高。低秩写入参数少，也不代表反向激活或每步推理成本自动少。

## 8. 最小模块契约（开发前冻结）

| 模块 | 输入 | 输出 | 验收 oracle |
|---|---|---|---|
| `TextCorrupt` | tokenized `c`、`s`、seed、事件 spans | `z_s`、mask 位置、target IDs | `s=1` 时目标保留；mask 率、跨度和 token 对齐正确 |
| `VisualRead` | `x_t` 或指定 pre-CA hidden、`grid_sizes` | `R\in\mathbb R^{B\times F_z\times d_r}` | 改变帧序会改变时间化输出；不读正向 prompt |
| `TextRestore` | `z_s,R,t,s` | logits、`U^{\rm evt}` | 已知位置、mask 位置和最终恢复率可独立检验 |
| `TemporalAlign` | `U^{\rm evt},R` | `A\in\mathbb R^{B\times F_z\times E}` | shape、归一性、帧索引和事件顺序反例通过 |
| `VideoWrite` | 选层 `H_v,U^{\rm evt},A,t` | `H_v+\Delta H` | 零初始化逐 token 等于原模型；目标视频损失能反传至新增权重 |
| `Sampler` | 视频 scheduler、文字转移核、`c` | 视频、文字轨迹、时间表 | 相同 seed 可复现；两个噪声时钟有明确单位和方向 |

未完成的代码绑定：UNVERIFIED（本仓库尚未写实现），见 [实验判别](04_experiment_logic.md)。
