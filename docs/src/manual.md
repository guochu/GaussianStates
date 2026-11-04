# 理论与算法

本页给出 v7 API 的约定与算法：二次型 block 与 `c₀`、表象对照
（算子 Nambu vs 协方差 Majorana/(x,p)）、`purify`（费米/玻色）、
`GaussianState`（Gibbs）、TFD、熵与偏迹。
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

## 2. 两种表象：算子 block（Nambu）与态协方差（Majorana / (x,p)）

本包的**算子矩阵与协方差矩阵定义在不同的表象下**——基矢与排列顺序
都不同，二者不能直接相乘或逐元素比较：

| 矩阵 | 表象与基矢顺序 | 基矢定义 | 矩阵性质 |
|---|---|---|---|
| 费米 `X = bdg(op)`（block `h,A,B`） | Nambu `Ψ = (c₁..cₙ, c₁†..cₙ†)`：湮灭全体在前、产生全体在后 | `cₘ`：Fock 湮灭算符 | `[[h,A],[B,−hᵀ]]`，2n×2n 复；厄米算子 `X = X†` |
| 费米 `Γ = covariance(s)` | Majorana `(γ₁..γ₂ₙ)`：模式 `m` 占相邻指标 `[2m−1, 2m]` | `γ₂ₘ₋₁ = cₘ + cₘ†`，`γ₂ₘ = i(cₘ† − cₘ)` | `Γ_ab = i⟨γ_aγ_b⟩`，2n×2n 实反对称，`|eig(iΓ)| ≤ 1` |
| 玻色 `X = bdg(op)`（block `h,A,B`） | Nambu `Ψ = (a₁..aₙ, a₁†..aₙ†)`：湮灭全体在前、产生全体在后 | `aₘ`：Fock 湮灭算符 | `[[h,A],[B,hᵀ]]`，2n×2n 复；厄米算子 `X = X†` |
| 玻色 `Σ = covariance(s)` | 四极 `R = (x₁,p₁,x₂,p₂,…)`：模式 `m` 占相邻指标 `[2m−1, 2m]` | `xₘ = (aₘ+aₘ†)/√2`，`pₘ = i(aₘ†−aₘ)/√2` | `Σ_ab = ½⟨{R_a,R_b}⟩`，2n×2n 实对称正定，辛本征值 `ν ≥ ½` |

`GaussianState` 只存右列两种实协方差，不存 bdg/M 矩阵。注意**排序
不同**：Nambu 基把湮灭/产生各 `n` 个分居前后；Majorana / (x,p) 基按
模式交错——这也是 `partialtr` / `entropy` 的 `modes` 约定（物理模式
`p` ↔ 协方差指标 `[2p−1, 2p]`）。纯态判据：费米 `Γ² = −I`；玻色
辛本征值全 `½`（`ispure`）。

**中间量与换算**：`covariance(gs)` / `purify` 内部先在 Nambu 表象组装
Nambu 协方差 `C_ab = ⟨Ψ_aΨ_b†⟩`（费米）/ `C_ab = ½⟨{Ψ_a,Ψ_b†}⟩`
（玻色），再经 `to_majorana` / `to_xp` 转出（内部函数，不导出；逆
`to_nambu` / `to_bogoliubov`，反解用恒等式 `⟨γ_aγ_b⟩ = I − iΓ`——
对称部分被 CAR `{γ_a,γ_b}=2δ` 钉死——与 `W` 的幺正性）。用户侧的
跨表象入口只有两个，内部已自动换算：`covariance(gs)`（`bdg` 的
Nambu → Majorana/(x,p)）与 `quadraticoperator(s)`（协方差 → Nambu
`(h,A)`）。

`GaussianState` 不开放从协方差直接构造：纯态用 `purify(exp(H))`，
混合态用 `partialtr`，厄米 Gibbs 态用 `GaussianState(exp(H_herm))`。

## 3. purify：一般非厄米 e^H 的 Choi 矢量化

### 3.0 矢量化与 EPR 参考态

在倍增 Fock 空间（ket 模 c/a，bra 模 d）上定义 Choi/矢量化态

```text
|ρ⟩⟩ = (ρ⊗I)|I⟩ ,
|I⟩ = exp(Σᵢ cᵢ†dᵢ†)|0⟩ （费米） ,   exp(Σᵢ aᵢ†dᵢ†)|0⟩ （玻色）
```

ket 模 `1..n`、bra 模 `n+1..2n`。对 bra 模偏迹还原 ρ（差一个标量；
协方差与归一化无关，故 ρ 不必归一化）。这是算符相干态矩阵元表示，
也是 GMPS/IF 系数态与算符空间 OSEE 的约定。

费米 `|I⟩_G = ⊗ᵢ(|00⟩+|11⟩)`：求和有限、已归一。玻色 `|I⟩_B`
（单模即 `Σₖ|k,k⟩`）**不可归一化**；但下述镜像恒等式都是它承受的
算符作用恒等式，逐字成立，可归一化由 `(ρ⊗I)` 的衰减补上，并在玻色侧
体现为协方差补全的正定性（§3.2）。

**镜像恒等式**（`M` 是线性镜像，只定义在 ket 算符的线性张成上，
非代数同态）：

```text
费米：cᵢ|I⟩ = dᵢ†|I⟩ ,  cᵢ†|I⟩ = −dᵢ|I⟩     M: cᵢ↦+dᵢ† , cᵢ†↦−dᵢ
玻色：aᵢ|I⟩ = dᵢ†|I⟩ ,  aᵢ†|I⟩ = dᵢ|I⟩     M: aᵢ↦+dᵢ† , aᵢ†↦+dᵢ
```

（费米符号来自 `|I⟩_G` 中 `c†d†` 的排序；玻色无符号。）

### 3.1 费米（O(n³)）

**湮灭子。** 伴随作用矩阵就是 BdG 指数：`e^{−H}Ψe^H = WΨ`，
`W = exp(bdg(H))`（§1）。记 `Ψ = (c,c†)`，`W` 第 r 行给
`O_r = ΣₖW_{rk}Ψₖ`，链式

```text
Ψ_r|ρ⟩⟩ = e^H·(O_r|I⟩_G) = e^H·(M(O_r)|I⟩_G) = M(O_r)|ρ⟩⟩
```

（第一步 `Xe^H = e^H·e^{−H}Xe^H`，第二步镜像恒等式，第三步 ancilla
算符与 `e^H` 对易）给 2n 个独立湮灭子，按 `(c,c†,d,d†)` 序

```text
V = E − W·Pmᵀ = [I₂ₙ | −W·Pmᵀ]
```

（`E = [I₂ₙ|0]` 取系统 Nambu 基；`Pm` 第 k 列 = `M(Ψₖ)` 的 4n 坐标，
只落在 ancilla 列上。）纯高斯态的湮灭子空间恰 2n 维，故 `V` 行满秩。

**定理（协方差 = 湮灭子行空间投影子）。** 设 `A = QΦ` 湮灭态，`Φ` 为
`(c,c†,d,d†)` 的 4n 维 Nambu 向量，`C_ab = ⟨Φ_aΦ_b†⟩`。由
`{Φ_a, Φ_b†} = δ_ab` 即 `Φ_a†Φ_b = δ_ab − Φ_bΦ_a†` 与
`⟨φ|A_k†A_l|φ⟩ = 0`：

```text
0 = Σ Q̄_ka·Q_lb·⟨Φ_a†Φ_b⟩ = [Q̄(I − C)Qᵀ]_kl  ⟹  (I − C)Qᵀ = 0
```

（`Q` 行满秩）。于是 `C` 在 rowspace(`Q`) 上恒等于 `I`；`C` 厄米、
纯态秩 2n，故 `C` 是湮灭子系数行空间上的**正交投影子**
`C = Q†(QQ†)⁻¹Q`。数值实现取 `V†` 的 QR 正交基 `B`、`C = BB†`
（避免 `(VV†)⁻¹` 的平方条件数），经 `(c,c†,d,d†) → (c,d,c†,d†)`
重排后按 `Ψ = (c,d)` 经 `to_majorana` 转 Majorana。
n=1..4 与稠密零空间逐元素一致（1e-15）。

### 3.2 玻色（O(n³)）

**湮灭子。** 玻色生成元带符号约定 `[H,Ψ] = −𝒜Ψ`：

```text
𝒜 = [[h, Aᵀ], [−Bᵀ, −hᵀ]] ,    e^{−H}Ψe^H = exp(𝒜)Ψ ≡ WΨ
```

（可直接验证：`[a†ₖaₗ, aᵢ] = −δᵢₖaₗ` 等。）与费米完全同构的链式给
2n 个湮灭子 `V = E − exp(𝒜)·Pmᵀ`（`(a,a†,d,d†)` 序）。

**V 迷向 ⟹ 恒高斯。** A、B 复对称（构造器强制）⟹ `𝒜 ∈ sp(2n)`
（块形 `[[K,L],[M,−Kᵀ]]`，L、M 对称）⟹ `W` 辛。镜像把 Nambu 交换
形式 `J_ab = [Ψ_a,Ψ_b]` 变号（`(a,a†)` 换成 `(d†,d)`：`[d†,d] = −[a,a†]`），
于是

```text
[V_r, V_s] = J_rs − (WJWᵀ)_rs = 0
```

湮灭子两两对易。两两对易的 Nambu 线性算符组湮灭的态是（可能奇异的）
高斯态；可归一化时即正规高斯纯态。此结论对任意 n、任意（非）厄米
配对成立；n=1 与稠密 Fock Choi 对拍至机器精度。

**协方差补全。** 湮灭子写作 `A = w·R`：`R` 为倍增 (x,p) 四极算符
（`R = (x₁,p₁,…,xₙ,pₙ | x'₁,p'₁,…)`），`w` 为 `V` 的四极协变式行。
由算符代数 `½{A,R_a} − ½[A,R_a] = R_aA`、对易关系 `[R_a,R_b] = iΩ_ab`
（`Ω` 取倍增 4n×4n）及 `⟨φ|R_aA|φ⟩ = 0`，纯态约束化为左核形式

```text
w(Σ − (i/2)Ω) = 0 ,    w̄(Σ + (i/2)Ω) = 0
```

`w` 行与其复共轭合计 4n 个独立约束，解线性系统

```text
[w; w̄]·Σt = [ +(i/2)wΩ ; −(i/2)w̄Ω ]
```

唯一确定 `Σ`（约束奇异 ⟺ 湮灭子协变式线性相关，抛 `ArgumentError`）。
解为实对称；**实对称正定 ⟺ `(ρ⊗I)|I⟩_B` 可归一化**。不正定（`e^H`
含增长方向、压缩参量 `|tanh r| → 1` 等失稳/无界情形）时抛
`ArgumentError`。

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

纯度（整态；子系统经 `purity(partialtr(s, modes))`）：

```text
purity = Tr ρ² = Πₖ (1+νₖ²)/2 （费米） ,   Πₖ 1/(2νₖ) （玻色）
```

纯态为 1；费米最大混合单模 ½，玻色随 ν→∞ 到 0。

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

全部见 `test/runtests.jl`（146 项）：

- 费米 `purify`：n=1..4 与稠密零空间/`|ρ⟩⟩` Majorana 协方差逐元素
  （1e-15）；TFD 的 `partialtr == Gibbs`。
- 变换：Nambu↔Majorana、Bogoliubov↔(x,p) 与稠密 Fock 直算的双向
  仲裁 + 往返（1e-10 / 1e-12）。
- 费米 Gibbs：协方差 vs 稠密 `e^H/Z`。
- 玻色 Gibbs：单模解析 `ν = ½coth(βω/2)`；压缩热态（A≠0）vs 稠密
  Fock；失稳/无界/零模抛 `ArgumentError`。
- 玻色 `purify`：数守恒复 `t = e^h` 的 TMS 解析协方差；配对/复 `h`
  vs 稠密 Fock Choi；不可归一化抛错。
- 玻色 TFD：n=1 解析（含 x-p 交叉）；n=1..8 随机稳定算符
  `ispure` + `partialtr == Gibbs`。
- 性质：纯态整态熵 0、`S(∅)=0`、纯态互补 `S(A)=S(Ā)`、`c₀` 无关、
  Rényi `α→1` 连续、费米 `α=2` vs 稠密偏迹、玻色 `α=2` 解析；
  purity 纯态 1、单模解析、稠密 `Tr ρ²`（整态 + 子系统）。
- 错误路径：尺寸/对称性/厄米性校验、非物理协方差、越界/重复
  `modes`、非法 `α`。

稠密 Fock 仲裁的注意点：配对矩阵元 `∝ √(mn)` 随能级增长，强压缩/
热态需要大 `d`（`fockmatrix` 仅调试用）；判别"构造错还是仲裁
截断错"要看仲裁随 `d` 的收敛。
