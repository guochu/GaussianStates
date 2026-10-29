# 理论与算法

本页给出 v7 API 的约定与算法：二次型 block 与 `c₀`、协方差表示、
`purify`（费米/玻色）、`GaussianState`（Gibbs）、TFD、熵与偏迹。
所有关键公式都用稠密 Fock 构造逐元素仲裁验证（`test/runtests.jl`，
误差 1e-10 ~ 1e-16）。

## 1. 二次型算子与 block 约定

### 费米

Nambu 自旋量 `Ψ = (c₁..c_n, c₁†..c_n†)`。一般（非厄米）算子只存
3 个独立 block 与常数：

```text
H = c†hc + ½(c†Ac† + cBc) + c₀ ,    A、B 任意输入
                                    （构造时自动反对称化：对称部分在
                                    c†ᵢc†ⱼ = −c†ⱼc†ᵢ 下自动抵消，对角恒为零）
```

规范 BdG 矩阵（`bdg(op)`）

```text
X = ⎡ h   A ⎤ ,    ½Ψ†XΨ = c†hc + ½c†Ac† + ½cBc − ½tr(h)
    ⎣ B  −hᵀ⎦
```

从 2n×2n 矩阵 `X` 构造时提取 `h = (X₁₁−X₂₂ᵀ)/2`、`A = antisym(X₁₂)`、
`B = antisym(X₂₁)`，并计入常数差 `c₀ = c_in + ½tr(X₂₂) + ½tr(h)`。
厄米特例 `h = h†, B = A†` 只存 `(h, A)`；`c₀` 必为实。

**重要**：伴随作用——`e^{−H}Ψ_ae^{H} = Σ_k [e^X]_{ak}Ψ_k`，即伴随作用
矩阵就是 BdG 矩阵的指数（这是费米 `purify` 的核心；`M = [[−h,B],[A,hᵀ]]`
之类的变体都是错的）。

### 玻色

```text
H = a†ha + ½(a†Aa† + aBa) + c₀ ,    A、B 任意输入
                                    （构造时自动对称化：反对称部分在
                                    a†ᵢa†ⱼ = a†ⱼa†ᵢ 下自动抵消）
X = ⎡ h   A ⎤ ,    ½Ψ†XΨ = a†ha + ½a†Aa† + ½aBa + ½tr(h)
    ⎣ B   hᵀ⎦
```

矩阵输入时常数 `c₀ = c_in + ½tr(X₂₂) − ½tr(h)`。厄米特例同费米。

**指数约定**：`H` 是指数——热态传 `H = −β·H_sys`（`h ≈ −βω < 0`）。
玻色 Gibbs 态存在要求 `H_sys` 有下界；`e^H` 无界 / 动力学失稳时
`GaussianState`/`purify` 抛 `ArgumentError`（判据是 Bogoliubov 生成元的
σ₃ 符号结构，谱全实本身不保证稳定）。

## 2. 态的协方差表示（内部）

`GaussianState` 只存实协方差，不存 bdg/M 矩阵：

```text
费米：Γ_ab = i⟨γ_aγ_b⟩      2N×2N 实反对称，γ_{2m−1} = c_m + c_m†，γ_{2m} = i(c_m† − c_m)
玻色：Σ_ab = ½⟨{R_a,R_b}⟩   2N×2N 实对称正定，R = (x₁,p₁,x₂,p₂,…)，x = (a+a†)/√2
```

用哪种基是内部约定（`to_majorana` / `to_xp` 不导出）。纯态判据：
费米 `Γ² = −I`；玻色辛本征值全 `½`（`ispure`）。物理模式 `p` 对应
协方差指标 `[2p−1, 2p]`；`partialtr` 取子矩阵，`entropy` 在子矩阵上算。

`GaussianState` 不开放从协方差直接构造：纯态用 `purify(exp(H))`，
混合态用 `partialtr`，厄米 Gibbs 态用 `GaussianState(exp(H_herm))`。

## 3. purify：一般非厄米 e^H 的 Choi 矢量化

目标：`|ρ⟩⟩ = (ρ⊗I)|I⟩_G`，其中
`|I⟩_G = exp(Σᵢ cᵢ†dᵢ†)|0⟩`（费米）或 `exp(Σᵢ aᵢ†dᵢ†)|0⟩`（玻色），
ket 模 `1..n`、bra 模 `n+1..2n`。这是 Gaussian 算符相干态矩阵元的
矢量化，也是 GMPS/IF 系数态的约定。

### 3.1 费米（O(n³)）

`|I⟩_G` 上的镜像恒等式（算符在态上的恒等式，无基约定歧义）：

```text
cᵢ|I⟩ = dᵢ†|I⟩ ,    cᵢ†|I⟩ = −dᵢ|I⟩
```

对任意 ket 扇区线性算符 `X`，利用 `Xe^H = e^H·(e^{−H}Xe^H)` 与镜像：

```text
(X − M(X'))|ψ⟩ = 0 ,    X' = e^{−H}Xe^H ,   M = 镜像（cᵢ↦+dᵢ†, cᵢ†↦−dᵢ）
```

于是湮灭子矩阵（2n×4n，`(c,c†,d,d†)` 序）

```text
V = [I₂ₙ | −exp(X)·Pmᵀ]
```

行空间张成湮灭子；协方差 = 湮灭子行空间投影子（用 QR 正交基构造，
避免 `(VV†)⁻¹` 的平方条件数），重排到 `(c,d,c†,d†)` 后转 Majorana。
n=1..4 与稠密零空间逐元素一致（1e-15）。

### 3.2 玻色（O(n³)）

镜像恒等式无符号：`aᵢ|I⟩ = dᵢ†|I⟩`、`aᵢ†|I⟩ = dᵢ|I⟩`；伴随作用矩阵
`W = exp(𝒜)`，`𝒜 = [[h,Aᵀ],[−Bᵀ,−hᵀ]]`（`[H,Ψ] = 𝒜Ψ`）。湮灭子

```text
V = [I₂ₙ | −exp(𝒜)·Pmᵀ]
```

**A、B 复对称 ⟹ 𝒜 ∈ sp(2n) ⟹ V 辛迷向 ⟹ Choi 态恒为高斯**
（对任意 n、任意（非）厄米配对成立；n=1 与稠密 Fock Choi 对拍至机器
精度）。协方差由核条件补全：纯态满足
`W̄_q(Σ + (i/2)Ω) = 0`（`W_q` 为四极协变式），即线性系统

```text
[W_q; W̄_q]·Σ = [∓(i/2)W_qΩ; ±(i/2)W̄_qΩ]
```

解出的 Σ 实对称；不可归一化（如 `h` 含增长方向、`|tanh r| → 1`）时
结果不正定，抛 `ArgumentError`。

## 4. Gibbs 态与 TFD（厄米 e^H）

### 4.1 费米 Gibbs

`C = (I + e^X)^{−1}`（Nambu，特征分解实现），`Γ = to_majorana(C)`。

### 4.2 玻色 Gibbs

不能用 `½coth(−bdg/2)`（只在 `A = 0` 正确；压缩热态会错 ~20%）。
正确构造：Bogoliubov 生成元 `𝒜 = [[h,A],[−Ā,−hᵀ]]` 本征值 ± 配对，
负支 `α` 的模热权重 `ν = ½coth(−α/2)`，按本征值分组做 σ₃-Gram 组装

```text
C = Σ_groups ν (T₋T₋† + T₊T₊†)
```

（T± 为 σ₃-正交基，简并谱按子空间 Gram 求逆，基不变）。失稳/无界
由 σ₃ 符号检查与 `α < 0` 要求拒绝。

### 4.3 TFD

`purify(gs)`：`√ρ = e^{H/2}` 的矢量化（内部把 `gs.op/2` 经
`.B = A†` 合成转成一般 exp 再走 Choi 路径），2n 模纯态，两种统计
同一路径。检验
`partialtr(psi, 1:n) == GaussianState(gs)`（测试固化）。

两种语义别混：`purify(gs)`（TFD）的纠缠 = Gibbs **态**的纠缠熵；
Gibbs **密度矩阵**的算符空间 OSEE 要用完整指数：
`purify(exp(FermionicQuadraticOperator(gs.op.h, gs.op.A, gs.op.B)))`
（`.B` 合成对齐接口，无需 `asoperator`）。

## 5. 熵与偏迹

物理模式 `p` ↔ 协方差指标 `[2p−1, 2p]`；`partialtr(gs, modes)` 取
`Γ[2modes−1,2modes]` / `Σ[...]` 子矩阵。

```text
费米：ν_k = |eig(iΓ_A)|（成对 ±ν）；S = ½ Σ_k h((1+ν_k)/2)
玻色：ν_k = |eig(Ω_A Σ_A)|（成对 ±ν）；S = ½ Σ_k f(ν_k)
      h = 二元熵，f(ν) = (n̄+1)log(n̄+1) − n̄ log n̄，n̄ = ν − ½
```

Rényi：`α = 1` 为 von Neumann；一般 `α` 按同一谱的 Rényi 公式。
OSEE 的写法只有一种：`psi = purify(exp(H))` 算一次，然后
`entropy(psi, modes)`。费米子系统按代数（模式）约化；连续块/单模
与朴素偏迹一致，非连续割不要拿朴素偏迹当参照。

**站点割**：物理站点 `p` 的割在 doubled 空间是 `modes = [p, n+p]`；
包不引入 site 抽象，`(branch,time,band)` 标签由下游映射成 `modes`。

## 6. 复杂度

| 操作 | 复杂度 |
|---|---|
| 算子构造/提 block | O(n²) |
| `purify`（费米/玻色） | O(n³)（矩阵指数 + 投影子/线性解） |
| `GaussianState(Gibbs)` | O(n³) |
| `partialtr` | O(1)（子矩阵拷贝） |
| `entropy` | O(模式数³) |

实测：费米 `purify` n=64 约 72 ms、n=128 约 112 ms；玻色 n=8/16 的
Gibbs/TFD/purify 均毫秒级。`purify` 不缓存——调用一次、复用返回值。

## 7. 数值验证摘要

全部见 `test/runtests.jl`（95 项）：

- 费米 `purify`：n=1..4 与稠密零空间/`|ρ⟩⟩` Majorana 协方差逐元素
  （1e-15）；TFD 的 `partialtr == Gibbs`。
- 费米 Gibbs：协方差 vs 稠密 `e^H/Z`。
- 玻色 Gibbs：单模解析 `ν = ½coth(βω/2)`；压缩热态（A≠0）vs 稠密
  Fock；失稳/无界/零模抛 `ArgumentError`。
- 玻色 `purify`：数守恒复 `t = e^h` 的 TMS 解析协方差；配对/复 `h`
  vs 稠密 Fock Choi；不可归一化抛错。
- 玻色 TFD：n=1 解析（含 x-p 交叉）；n=1..8 随机稳定算符
  `ispure` + `partialtr == Gibbs`。
- 性质：纯态整态熵 0、`S(∅)=0`、纯态互补 `S(A)=S(Ā)`、`c₀` 无关、
  Rényi `α→1` 连续、费米 `α=2` vs 稠密偏迹、玻色 `α=2` 解析。
- 错误路径：尺寸/对称性/厄米性校验、非物理协方差、越界/重复
  `modes`、非法 `α`。

稠密 Fock 仲裁的注意点：配对矩阵元 `∝ √(mn)` 随能级增长，强压缩/
热态需要大 `d`（`fockmatrix` 仅调试用）；判别"构造错还是仲裁
截断错"要看仲裁随 `d` 的收敛。
