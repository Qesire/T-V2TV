# 文字条件对最终视频分布与语义几何的数学作用

日期：2026-11-21。研究对象：公开 Wan2.1 T2V-1.3B，固定视频骨干与采样配置，优先使用无需预训练的分析与后训练接口。

**状态：数学推导与解析小例子已核查；尚未进行 Wan 采样测量、模型训练或性能实验。** 下文把标准理论、针对当前问题的推导和待验证假说分开。

## 1. 核心对象：生成映射、条件测度与语义约束

设原始文字为 $c$，固定形状的上下文化条件张量为 $C=E(c)\in\mathbb R^{L\times d_c}$；需要求导时按 token 与通道轴向量化。文本投影、CFG 和 solver 均计入采样映射。

$$
V=G_C(\xi)=D_{\mathrm{VAE}}(T_C(\xi)),
\qquad \xi\sim\nu,
\qquad \mu_C=(G_C)_\#\nu.
$$

$V$ 表示量化与视频压缩前的连续视频张量；$\xi$ 收入采样使用的全部随机源。固定 $\xi$ 后，生成映射是确定的。$\mu_C$ 是完整随机种子总体上的条件输出分布。

选择任务读出 $\Psi$，例如对象相对背景的轨迹、事件发生时间、相对关系、身份或相机运动：

$$
y_C(\xi)=\Psi(G_C(\xi)),\qquad \bar\mu_C=(\Psi)_\#\mu_C.
$$

$\Psi$ 决定分析保留什么信息。忽略帧顺序的池化特征不能承担动作先后分析；硬检测或人工评分可用于有限干预，但不应默认可微。

| 对象 | 数学描述 | 回答的问题 |
|---|---|---|
| 单种子响应 | $D_CG_C(\xi)[\delta C]$ | 同一次生成怎样改变？ |
| 任务响应 | $D_Cy_C(\xi)[\delta C]$ | 动作、顺序、角色或相机怎样改变？ |
| 分布响应 | $\partial_a\mu_{C(a)}$ | 全部种子中的结果概率怎样改变？ |
| 参数化几何 | $(D_\xi y_C)^\top W(D_\xi y_C)$ | 噪声局部变化映射为怎样的视频差异？ |
| 语义约束集合 | $\mathcal M_c=\{V\in\mathcal M_{\mathrm{valid}}:F(V)=b(c)\}$ | 哪些视频符合指定语义？ |

这里 $\tau$ 表示噪声到视频的采样进度，$f$ 表示视频帧时间，$a$ 表示条件干预参数。三个轴不能互换。

## 2. 为什么不把“文本影响”直接等同于“支持流形变化”

在一个明确的理想条件下，latent 采样流 $T_C:\mathbb R^n\to\mathbb R^n$ 是全局微分同胚，$\nu$ 满支持，decoder 固定且连续，则

$$
\operatorname{supp}\mu_C=\overline{D_{\mathrm{VAE}}(\mathbb R^n)}
$$

不依赖 $C$。条件仍然能大幅改变共同支持上的概率质量与高概率语义区域。该命题不声称实际有限步 UniPC 必然可逆。

若 decoder 的像在局部为光滑流形，还可得到

$$
\operatorname{Im}D_\xi G_C
=\operatorname{Im}D D_{\mathrm{VAE}},
\qquad
D_CG_C[\delta C]\in\operatorname{Im}D_\xi G_C.
$$

因此，一个改变运动语义的方向，可以是共同 decoder 流形的切向，却是旧语义约束面的法向。这两种几何不能混同。

从理想联合数据分布出发，若离散文本事件 $c$ 具有正概率，条件化还可写成不要求环境密度的测度重加权：

$$
\mu_c(dV)
=\frac{P(c\mid V)}{P(c)}\,\mu_{\mathrm{base}}(dV).
$$

这里的 $\mu_{\mathrm{base}}$ 是同一联合分布的边缘。任意负 prompt 的生成分布不能直接替代它。学习到的 Wan 条件分布也需要和理想数据条件分布区分。

## 3. 两帧运动例子：支持相同，动作区域不同

用 $V=(q_0,q_1)$ 表示物体在两帧的水平位置，条件 $a$ 表示期望位移。令 $z,\eta$ 为独立标准 Gaussian，$\sigma>0$：

$$
q_0=z-\frac{a+\sigma\eta}{2},
\qquad
q_1=z+\frac{a+\sigma\eta}{2}.
$$

于是

$$
q_1-q_0=a+\sigma\eta,\qquad
\frac{q_0+q_1}{2}=z.
$$

所有 $\mu_a$ 都在 $\mathbb R^2$ 满支持，但概率集中在不同直线 $q_1-q_0=a$ 附近。相同 seed 的条件响应为

$$
\partial_aV=(-1/2,1/2)^\top.
$$

这保持平均位置，改变位移。直线的切向 $(1,1)$ 改变共同位置但保持位移；法向 $(-1,1)$ 改变位移。

$$
\mathbb E[q_1-q_0]=a,\qquad
\operatorname{Var}(q_1-q_0)=\sigma^2,\qquad
\|\partial_aV\|^2=1/2.
$$

由于协方差相同，两个条件的 Wasserstein 距离为

$$
W_2^2(\mu_a,\mu_b)=(a-b)^2/2.
$$

取 $\sigma=0.4$ 的解析值已用 NumPy 核对：位移方差 $0.16$，协方差最小特征值 $0.08$，条件响应平方范数 $0.5$。这是说明模型，不是 Wan 实测。

## 4. 文本如何改变流匹配速度场

这一节是对理想条件 flow matching 的推导。采用 $\tau:0\to1$ 表示从噪声到数据：

$$
X_\tau=\tau X^\star+(1-\tau)\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\qquad \epsilon\perp(X^\star,c).
$$

平方误差训练的理想条件速度为

$$
b_\tau^c(x)
=\mathbb E[X^\star-\epsilon\mid X_\tau=x,c].
$$

记 $m_\tau^c(x)=\mathbb E[X^\star\mid X_\tau=x,c]$，则

$$
\boxed{
b_\tau^c(x)=\frac{m_\tau^c(x)-x}{1-\tau}
},\qquad 0<\tau<1.
$$

文字改变的是：当前加噪状态可能来自哪些干净视频，以及这些干净视频的后验均值。这不是说最终视频就是该均值；实际样本由整个速度场的积分得到。

对 Gaussian 卷积求导，又有

$$
b_\tau^c(x)
=\frac{x}{\tau}
+\frac{1-\tau}{\tau}\nabla_x\log p_\tau(x\mid c).
$$

因此相对于参考条件 $c_0$：

$$
\boxed{
b_\tau^c-b_\tau^{c_0}
=\frac{m_\tau^c-m_\tau^{c_0}}{1-\tau}
=\frac{1-\tau}{\tau}
\nabla_x\log\frac{p_\tau(x\mid c)}{p_\tau(x\mid c_0)}
}.
$$

在 Bayes 分解成立处，最后的对数比可替换为 $p(c\mid X_\tau=x)/p(c_0\mid X_\tau=x)$，因为先验项对 $x$ 的梯度为零。

- $\tau=0$ 不直接代入带 $1/\tau$ 的式子；单独有 $b_0^c(x)=\mathbb E[X^\star\mid c]-x$。
- 不应仅凭系数 $(1-\tau)/\tau$ 判断晚期语义作用弱，因为 score 本身也随噪声尺度变化。
- Wan 的学习误差、CFG 和数值求解器均会改变真实生成映射。上述是理想路径解释，下一节给出真实计算图的作用模型。
- 固定时刻的 CFG score 线性组合不自动保证终点密度等于条件密度与参考密度的幂乘积。

## 5. 从局部条件注入到最终视频：终点响应算子

### 5.1 连续采样形式

固定 seed，写出包含实际指导策略的速度场：

$$
\frac{dX_\tau}{d\tau}=b_\theta(X_\tau,\tau,C),\qquad X_0=\xi.
$$

沿连续条件路径 $C(a)$，令 $\delta C=C'(a)$，在一条基准采样轨迹上定义

$$
S_\tau=\partial_aX_\tau,\quad
A_\tau=D_Xb_\theta,\quad
B_\tau=D_Cb_\theta.
$$

在可微条件下，直接微分 ODE：

$$
\dot S_\tau=A_\tau S_\tau+B_\tau\delta C,\qquad S_0=0.
$$

令 $\Phi(\tau,s)$ 满足

$$
\partial_\tau\Phi(\tau,s)=A_\tau\Phi(\tau,s),\qquad \Phi(s,s)=I.
$$

一般不同时刻的 $A_\tau$ 不交换，不能把传播算子无条件写成普通矩阵指数。终点响应为

$$
S_1=\int_0^1\Phi(1,\tau)B_\tau\delta C\,d\tau.
$$

故在任务空间：

$$
\boxed{
D_Cy_C(\xi)[\delta C]
=
\underbrace{D\Psi\,D D_{\mathrm{VAE}}}_{\text{终点读出}}
\int_0^1
\underbrace{\Phi(1,\tau)}_{\text{后续传播}}
\underbrace{B_\tau}_{\text{条件注入}}
\delta C\,d\tau
}.
$$

定义

$$
K_\tau=D\Psi\,D D_{\mathrm{VAE}}\,\Phi(1,\tau)B_\tau,
\qquad
J_C=\int_0^1K_\tau\,d\tau.
$$

这是条件经完整生成过程影响最终任务结果的算子。它不是某个 attention map。

Wan 的 cross-attention 在局部包含

$$
\delta(\Pi V)=(\delta\Pi)V+\Pi\,\delta V.
$$

两项分别改变语义分配与所传内容；$B_\tau$ 包含这些路径，$\Phi$ 再计入后续层、视频 self-attention、FFN 与采样动力学的传播。

### 5.2 对真实多步求解器的精确离散形式

对 UniPC，完整状态必须包括历史预测：

$$
s_k=(X_k,h_k),\qquad s_{k+1}=\mathcal F_k(s_k,C).
$$

固定网格、CFG、负提示及其他配置，令

$$
S_{k+1}=A_kS_k+B_k\delta C,\qquad
A_k=D_s\mathcal F_k,\quad B_k=D_C\mathcal F_k.
$$

则

$$
\boxed{
J_C=
D(\Psi\circ D_{\mathrm{VAE}}\circ P_X)
\sum_{k=0}^{K-1}
A_{K-1}\cdots A_{k+1}B_k
}.
$$

空乘积取单位算子。$P_X$ 取最终 latent。中途分叉的干预必须连同 solver history 一起复制；只复制 latent 不代表相同采样状态。

若将每步条件形式上解绑定成 $C^{(k)}$，则 $J_C=\sum_kJ_k$。因此

$$
J_C^\top WJ_C=\sum_{k,l}J_k^\top WJ_l.
$$

逐步敏感性平方和会遗漏跨步加强或抵消的项。

### 5.3 离散文字的作用如何定义

词语本身没有普通欧氏导数。可选择语义最小对照 prompt $c_0,c_1$，编码后定义

$$
C(a)=C_0+a(C_1-C_0),\qquad a\in[0,1].
$$

该路径中间点不必对应合法句子；它是一条明确的模型接口干预。端点有限变化满足

$$
y_{C_1}(\xi)-y_{C_0}(\xi)
=\int_0^1J_{C(a)}(\xi)(C_1-C_0)\,da.
$$

上下文化编码器使一个词的修改通常影响全部输出 token。修改原始动词，与把一个末层 token slot 置零，不是同一个干预。按 token 或采样步细分的路径归因也未必唯一。

## 6. 两种拉回几何：条件可控性与生成族几何

### 6.1 条件方向的有效几何

选择固定任务度量 $W\succeq0$：

$$
M_C=\mathbb E_\xi[J_C(\xi)^\top WJ_C(\xi)].
$$

小扰动满足

$$
\mathbb E_\xi\|y_{C+\delta C}-y_C\|_W^2
=\delta C^\top M_C\delta C+o(\|\delta C\|^2).
$$

$M_C$ 在有效满秩子空间上可作局部度量；一般只是半正定形式。特征值大表示该条件方向更能改变选定任务表示；零空间表示所选读出在一阶上不敏感。

若用少量语义坐标 $a\in\mathbb R^m$ 参数化条件，令 $Q=D_aC$，实际只需估计 $Q^\top M_CQ$，不必显式构造数百万维矩阵。

作用强度与正确性不同。对于固定目标文本 $c_\star$ 和任务损失 $\ell$，真正改善要求

$$
\frac{d}{da}\mathbb E_\xi[\ell(G_{C(a)}(\xi);c_\star)]<0.
$$

### 6.2 给定文字后的生成族几何

若关心文本是否改变“种子局部方向代表怎样的视频变化”，定义

$$
J_\xi=D_\xi y_C,\qquad
g_{\xi\xi}(C,\xi)=J_\xi^\top WJ_\xi.
$$

在二阶可微处，沿条件路径：

$$
\partial_ag_{\xi\xi}
=(\partial_aJ_\xi)^\top WJ_\xi
+J_\xi^\top W(\partial_aJ_\xi).
$$

这是条件对生成映射局部度量的影响。它可表达局部放大、压缩和方向变化，但属于指定生成参数化与特征读出的几何；不能直接宣称真实自然视频流形的内禀几何改变。

两帧例子中条件只平移均值，$g_{\xi\xi}$ 不变，条件分布仍发生变化。因此度量形变不是语义控制的必要条件。

## 7. 把语义“流形”写成可核对的约束面

取光滑局部语义读出 $F$，目标为 $b(a)$：

$$
\mathcal M_a=\{V:F(V)=b(a)\}.
$$

在 $A=DF(V)$ 满行秩处，正则水平集定理给出

$$
T_V\mathcal M_a=\ker A.
$$

对于运动方向、精确事件时间等连续约束，这一模型直接可用。事件先后一般写成不等式，例如 $t_A<t_B$；对应的是可行区域及其边界，可用平滑 margin 表示，而不强行假定全局光滑等式流形。角色切换等离散结构也宜分局部模式分析。

若生成结果精确满足语义，令 $U_a=\partial_aG_{C(a)}(\xi)$，则

$$
\boxed{A\,U_a=b'(a)}.
$$

在欧氏局部坐标中：

$$
U_a=A^\top(AA^\top)^{-1}b'(a)+U_T,\qquad A U_T=0.
$$

第一项是满足目标语义变化所需的最小范数法向响应；第二项保持当前约束值。对于真实视频有效性约束，应在有效视频流形的局部坐标中使用同一公式，或先限制到其切空间，不能把任意像素法向扰动视为有效视频。

实际模型定义语义误差

$$
e_a(\xi)=F(G_{C(a)}(\xi))-b(a).
$$

则

$$
\boxed{
e_a'=DF(G_{C(a)}(\xi))\,D_CG_{C(a)}(\xi)\,C'(a)-b'(a)
}.
$$

这个“语义响应偏差”回答：文字要求的改变与视频实际执行的改变是否一致。它比 Jacobian 范数大不大更接近动作遵循。

若目标 prompt 固定而改变辅助文字 writer，应令 $b$ 固定；此时考察 $\ell$ 下降或现有残差减小，而不是把辅助条件变化重新解释成新目标。

## 8. 从同 seed 响应到分布输运

沿条件路径记 $G_a=G_{C(a)}$，$U_a=\partial_aG_a$。若 $G_a$ 在 $L^2(\nu)$ 中绝对连续且导数平方可积，定义

$$
u_a(V)=\mathbb E[U_a(\xi)\mid G_a(\xi)=V].
$$

对光滑测试函数 $\varphi$：

$$
\frac{d}{da}\int\varphi\,d\mu_a
=\mathbb E[\nabla\varphi(G_a(\xi))^\top U_a(\xi)]
=\int\nabla\varphi(V)^\top u_a(V)\,d\mu_a.
$$

因此以弱意义有

$$
\boxed{\partial_a\mu_a+\nabla\cdot(\mu_a u_a)=0}.
$$

这不要求 $\mu_a$ 有环境 Lebesgue 密度，适用于 decoder 像上的奇异分布。若在选定坐标中有光滑正密度 $\rho_a$，则

$$
\partial_a\log\rho_a
=-\nabla\cdot u_a-u_a\cdot\nabla\log\rho_a.
$$

固定种子的配对只是一个合法 coupling，因此

$$
W_2^2(\mu_{C_0},\mu_{C_1})
\le
\mathbb E_\xi\|G_{C_0}(\xi)-G_{C_1}(\xi)\|^2.
$$

同 seed 响应大不保证分布变化大。例如 $G_a(z)=R_az$、$z\sim\mathcal N(0,I_2)$、$R_a$ 为旋转：每个样本移动，分布却完全不变。

### 更精细的分解

在 $L^2(\mu_a)$ 中，令 $\mathsf T_{\mu_a}$ 为光滑梯度场的闭包，$v_a=P_{\mathsf T_{\mu_a}}u_a$。对几乎处处 $a$：

$$
|\mu'|_{W_2}^2
=\|v_a\|_{L^2(\mu_a)}^2
\le \|u_a\|_{L^2(\mu_a)}^2
\le\mathbb E\|U_a\|^2.
$$

令 $w_a=u_a-v_a$，条件期望与正交投影给出

$$
\mathbb E\|U_a\|^2
=
|\mu'|_{W_2}^2
+\|w_a\|_{L^2(\mu_a)}^2
+\mathbb E\|U_a-u_a(G_a(\xi))\|^2.
$$

三项依次是总体分布变化、保持分布的一阶重排、同一输出纤维上不同种子的速度差异。这里的 Wasserstein 切空间不是支持流形的经典切空间。

实测不应固定有限样本数后令差分步长无限趋零，再把经验 OT 速度当总体速度；匹配锁定可能把种子重排误报为分布变化。可直接报告语义统计量的期望与分布，明确有限样本误差。

## 9. 动态文字为什么可能有额外作用空间

在同一基准轨迹上线性化，若允许每个采样进度具有不同条件扰动 $u_\tau=\delta C_\tau$：

$$
\delta y=\mathcal Ku:=\int_0^1K_\tau u_\tau\,d\tau.
$$

静态条件是特例 $u_\tau=\delta C$：

$$
\delta y_{\mathrm{static}}=
\left(\int_0^1K_\tau\,d\tau\right)\delta C.
$$

因此在允许任意平方可积条件扰动的放宽模型中：

$$
\operatorname{Ran}J_{\mathrm{static}}
\subseteq\operatorname{Ran}\mathcal K.
$$

包含未必严格；收益也可能体现为达到同一输出变化时需要更小扰动。

对时间区间长度为 1、相同 $L^2$ 输入能量，定义

$$
W_{\mathrm{dyn}}=\int_0^1K_\tau K_\tau^\top d\tau,\qquad
W_{\mathrm{static}}=JJ^\top,\quad J=\int_0^1K_\tau d\tau.
$$

则直接展开可得

$$
W_{\mathrm{dyn}}-W_{\mathrm{static}}
=\int_0^1(K_\tau-J)(K_\tau-J)^\top d\tau\succeq0.
$$

若任务残差修正 $r\in\operatorname{Ran}W_{\mathrm{dyn}}$，无约束线性系统的最小能量输入为

$$
u_\tau^\star=K_\tau^\top W_{\mathrm{dyn}}^\dagger r.
$$

若 $r$ 不在可达空间，只能达到其投影 $W_{\mathrm{dyn}}W_{\mathrm{dyn}}^\dagger r$。小奇异值会放大所需输入；线性修正需控制在局部范围内。

这说明原始设想有清楚的数学成立空间：文字恢复中间态若产生有效的阶段相关条件方向，就可能避免静态条件跨步抵消，或更高效地实现语义修正。

但任意 embedding 控制空间比可实现的文字去噪轨迹宽。若输入限定在子空间 $\mathcal U$，应使用

$$
W_{\mathcal U}=\mathcal K P_{\mathcal U}\mathcal K^*.
$$

是否包含静态控制空间也需检查。若文本恢复依赖当前视频，必须把文字状态 $Z_\tau$ 与 solver history 一并加入系统重新线性化；新增反馈项类似 $F_C R_X$，不能仍套用冻结反馈的开环动力学。

固定输出 prompt 的恢复准确率不决定这些中间控制方向是否有用。需要判断它们是否降低同一目标的语义残差，且保留身份、外观等要求。

## 10. 面向 Wan 的最小可计算模型

先估计有限个有意义的条件方向和任务输出，不拟合全局视频流形。

1. 选语义明确的 prompt 对，如运动方向、速度、角色对调或动作先后；保留完整上下文化条件变化。
2. 固定 seed、CFG、负提示、时间网格和全部 solver 状态，对完整采样器做方向导数或有限差分。
3. 分开记录相对物体运动、相机运动、动作顺序、身份与外观。保留完整视频供人工核对读出是否正确。
4. 估计低维 $J_a=D_Cy\,D_aC$、条件度量 $M_a=\mathbb E[J_a^\top WJ_a]$ 及语义响应偏差。
5. 需要判断动态条件空间时，再将干预限定到早、中、晚采样窗口估计 $K_\tau$ 的粗分块；这不是按视频帧分段。
6. 只有在确定有可用写入方向后，才比较静态语义 writer、时间调度 writer 和文字恢复反馈。目标均保持为原 prompt 的视频语义改善。

中心差分的可计算形式为

$$
J_C\delta C\approx
\frac{y_{C+h\delta C}(\xi)-y_{C-h\delta C}(\xi)}{2h}.
$$

它需要完整采样，但无需保留全程反向计算图。自动微分 JVP/VJP 也可用，是否高效取决于具体实现；不能把单步 latent 差分当完整采样端点导数。应使用多个合理步长检查局部近似，避免数值噪声或过大扰动改变结论。

## 11. 与已有理论与文献的关系

| 来源 | 已有对象 | 当前问题的延伸 |
|---|---|---|
| Park et al., NeurIPS 2023, Diffusion-Pullback | 图像扩散 noisy latent 到 U-Net bottleneck 的 Jacobian 拉回几何；分析 prompt 对局部子空间的影响 | 完整 Wan 采样到视频任务特征的终点算子；区分条件度量与 seed 度量 |
| Lipman et al., Flow Matching, ICLR 2023 | 向量场、流映射、概率路径与连续性方程 | 将文字作为外部条件，计算条件对流与终点的微分 |
| Song et al., Score SDE, ICLR 2021 | 条件 score 与可控生成 | 解释条件如何偏转当前噪声尺度的生成动力学 |
| Ambrosio–Gigli–Savaré, Gradient Flows | 测度曲线、连续性方程、最小范数 Wasserstein 速度 | 区分同 seed 变化、分布输运与保持分布的重排 |
| Wan2.1 固定版本代码 | 条件注入、CFG、VAE、真实 solver | 把上述抽象接回可测的多步计算图 |

Diffusion-Pullback 第 4.3 节以 Stable Diffusion 和 50 条 MS-COCO captions 分析局部子空间；它不是 Wan 视频验证，也没有测量这里定义的最终动作响应。其几何依赖选定的 U-Net 特征空间，不能直接等同真实自然图像流形。

### 原始来源

- Park et al. [Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry](https://proceedings.neurips.cc/paper_files/paper/2023/file/4bfcebedf7a2967c410b64670f27f904-Paper-Conference.pdf), NeurIPS 2023；[作者代码](https://github.com/enkeejunior1/Diffusion-Pullback)。
- Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2), ICLR 2023，特别是第 2 节与附录 B。
- Song et al. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/html/2011.13456v2), ICLR 2021，第 5 节。
- Ho & Salimans. [Classifier-Free Diffusion Guidance](https://arxiv.org/html/2207.12598v1), 2022，第 3.2 节。
- Ambrosio, Gigli & Savaré. [Gradient Flows](https://www2.stat.duke.edu/~sayan/ambrosio.pdf), 2005，第 8 章，Theorem 8.3.1、Lemma 8.4.2、Proposition 8.4.5。
- Wan2.1 [model.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py) 与 [text2video.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py)，固定 commit 9737cba9c1c3c4d04b33fcad41c111989865d315。

相关前置整理：[文本条件的计算图与文献分类](09_text_condition_computational_graph.md)。
