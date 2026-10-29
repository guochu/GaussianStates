# GaussianStates.jl v2：非厄米高斯算子 / 纯化 / OSEE —— API 计划

状态：**设计草案**（大改前）
日期：2026-10-29
当前实现快照：`backup/GaussianStates`（含 `.git` 与未提交改动；commit `4aa5fc5` + 工作区）

---

## 0. 需求、对象与设计原则

### 0.1 核心需求

我们的核心物理对象是**离散化后的影响泛函（IF）**，它是一个 Bogoliubov 形式的二次型
算符

```text
ρ = exp(H) ,      H = ½ ψ† X ψ  (+ 常数)      X 一般复矩阵，H 一般非厄米
```

- `ρ` **不是密度矩阵**：`H` 非厄米 ⟹ `ρ` 非厄米、非正定（甚至不迹类）。
- 核心观测量是 **OSEE**：把 `ρ` 矢量化/纯化为 doubled 空间中的纯态
  `|ρ⟩⟩ = (ρ ⊗ I)|I⟩`，OSEE 就是 `|ρ⟩⟩` 的**二分纠缠熵**（子系统的纠缠熵）。
- `|ρ⟩⟩` 是**纯高斯态**；它的子系统约化态是**正定、厄米、归一的高斯 Gibbs 态**。
  因此包内必须把 "Gaussian Gibbs state" 作为一等公民定义出来。

### 0.2 对象链（唯一主链）

```text
H (二次型，非厄米)
   │  GaussianOperator
   ▼
 ρ = e^H                         （算子，未归一化，非厄米）
   │  purify  / vectorize
   ▼
 |ρ⟩⟩ = (ρ ⊗ I)|I⟩               （doubled 空间中的纯高斯态，归一化后）
   │  reduce（对补集作约化）
   ▼
 ρ_A                             （子系统高斯 Gibbs 态：厄米、正定、归一）
   │  entropy
   ▼
 OSEE(A) = S(ρ_A)
```

**Gibbs 态在这条链里是"约化态"，不是起点。** 起点永远是 `GaussianOperator`。
这正是与当前包最大的语义差别：当前包把 "Gibbs 密度矩阵" 和 "纯配对向量"
硬塞进同一个 `FermionicGaussianState`，没有"非厄米算子"这一层。

### 0.3 设计原则

1. **类型即语义**：算子、纯态、Gibbs 态分成不同类型；不再用 `isbdg::Bool` 区分。
2. **一条主链、两个便利入口**：
   - 主链：`GaussianOperator → purify → PureState → reduce → GibbsState → entropy`；
   - 便利：`osee(op, sites) = entropy(reduce(purify(op), modes))`。
3. **协方差是唯一数值表示**：纯态用 Majorana/正则协方差 `Γ` 表示；Gibbs 态用
   协方差 `Γ` + 厄米生成元（费米 `M`、玻色 `K`）表示；算子用 BdG 矩阵 `X`。
   所有熵都从协方差算，O(模式数³)。
4. **约定显式化**：费米的 Z₂（graded）约化、doubled 空间的模式/站点布局、
   共轭位置、归一化常数，全部写进类型文档，不靠"实矩阵时恰好对"。
5. **稠密仲裁**：小系统一律用 `densitymatrix`/显式 Fock 构造做独立对照；
   特殊极限（厄米、纯配对）必须退化为现有公式。
6. **向后兼容层**：现有 `FermionicGaussianState`/`BosonicGaussianState`/`osee`
   的调用方式通过适配器（带 deprecation）保留到 v2.x，QuAPI/GTEMPO 不必立刻改。

### 0.4 非目标（本计划不做）

- 不做非二次型哈密顿量/非高斯态。
- 不做张量网络、MPS 截断、时间演化本身（这些在 QuAPI/GTEMPO）。
- 不把 `osee` 做成"任意算符"的通用 SVD（只对**高斯**算子走协方差快速算法；
  非高斯输入直接拒绝）。
- 不在 v2.0 支持玻色一般非厄米算子的**所有**可归一化边界情形（见 §9 工作包）。

---

## 1. 数学设定与约定

### 1.1 费米子

**模式与 Nambu 自旋量**：`m` 个费米模，

```text
ψ = (c₁ … c_m , c†₁ … c†_m) ,      {ψ_a, ψ_b†} = δ_ab ,   {ψ_a, ψ_b} = (S₀)_{ab}
S₀ = [[0, I], [I, 0]] ,              ψ† = S₀ ψ
```

**BdG 生成元与规范形式**：`H = ½ ψ† X ψ`，`X ∈ C^{2m×2m}`。由于 Nambu 冗余，
`X` 只有两个独立块；规范形式取

```text
X = ⎡ h   A ⎤ ,     h ∈ C^{m×m} 一般复,   A, B ∈ C^{m×m} 反对称
    ⎣ B  −hᵀ⎦

H = c† h c + ½ (c† A c† + c B c) + const
```

- 厄米 `H`：`B = A†`，`h = h†`（当前包的 Gibbs 分支）。
- 纯配对 `H`：`h = 0`、`B = 0`、`A = W`（当前包的纯配对态 `e^{½c†Wc†}|0⟩` 的
  生成元；注意那是**态** `e^H|0⟩`，不是**算符** `e^H` 本身，见 §1.4）。
- 常数：`H → H + c` 只给 `ρ` 乘 `e^c`，不影响归一化态/熵；但 `logZ`/`weight`
  需要它。

**算子**：

```text
ρ = e^{H} = e^{c} · exp(½ ψ† X ψ)
```

`ρ` 是有限维矩阵（Fock 空间 `2^m` 维），对任意复 `X` 都有定义；`ρ` 可能奇异
（`e^H` 的零本征值），但 `|ρ⟩⟩` 仍是良定义向量（除非 `ρ = 0`，即 `c = −∞`）。

**线性项（位移）**：若 `H` 还含 `η†ψ + h.c.`，`ρ` 仍是高斯算子，但 `|ρ⟩⟩` 是
**有位移**的高斯纯态。位移只改变一阶矩，不改变协方差，因此**不影响 OSEE**。
v2.0 的熵计算忽略线性项；`densitymatrix`/期望值若需要，再扩展 first moments
（v2.x）。

**矢量化的 EPR 态**（与当前包一致）：

```text
|I⟩_G = exp( Σ_i c†_i d†_i ) |0⟩_c|0⟩_d
|ρ⟩⟩  = (ρ ⊗ I_d) |I⟩_G = Σ_{s} (ρ|s⟩_c) ⊗ |s⟩_d
```

- `c` 为 ket 拷贝（算子本身的模），`d` 为 bra 拷贝；
- doubled 空间有 `2m` 个模：`c₁..c_m`、`d₁..d_m`；
- **物理站点** `p` 对应模对 `[p, m+p]`（`c_p` 与 `d_p`）；这是 OSEE 的割单位。
- 费米子的 Z₂/graded 约化：OSEE 用的子系统熵是**代数（模式）约化**熵，不是
  占据数基朴素偏迹；连续站点块下二者一致（当前包 `entropy` 的约定，沿用）。

**纯化态是纯高斯态**：`H` 二次 ⟹ `H⊗I` 在 doubled 空间二次 ⟹
`e^{H⊗I}|I⟩_G` 是高斯态；`|ρ⟩⟩` 作为向量是纯态（`|ρ⟩⟩⟨⟨ρ|` 纯）。

**子系统约化**：对 doubled 空间的补集模式作（代数）偏迹，得到 `2|A|` 模的
Gibbs 态

```text
ρ_A = Tr_{Ā} ( |ρ⟩⟩⟨⟨ρ| / ⟨⟨ρ|ρ⟩⟩ ) ,     ρ_A = e^{−H_A}/Z_A ,  H_A 厄米
```

`Γ_A` 是 `|ρ⟩⟩` 协方差 `Γ_d` 的子矩阵；`ρ_A` 由 `Γ_A` 唯一决定（高斯）。

**OSEE**：

```text
OSEE(A) = S(ρ_A) = Σ_k h((1+ν_k)/2) ,    {±ν_k} = eig(i Γ_A) ,  h(p)=−p ln p−(1−p)ln(1−p)
```

Rényi 同理（每个 ±ν 对换两能级 `p±=(1∓ν)/2`）。

### 1.2 玻色子

**正则坐标**：`R = (x₁,p₁,…,x_n,p_n)`，`[x_a,p_b] = iδ_ab`，辛形式 `Ω`。

**BdG 生成元**：`H = ½ Rᵀ K R`，`K ∈ C^{2n×2n}` **复对称**（`R` 是实算符，
二次型的反对称部分恒为零）。`K` 一般非厄米（复对称）⟹ `ρ = e^H` 非厄米。

- 厄米 `H`：`K` 实对称正定（当前包的 Gibbs 分支）。
- 纯配对 `H`：`H = ½(a† W a† + h.c.)` 的配对项，对应特定的复对称 `K`；
  当前包的玻色纯配对分支是**态** `e^{½c†Wc†}|0⟩`。

**矢量化**：`|ρ⟩⟩ = (ρ⊗I)|I⟩`，`|I⟩ = Σ_s|s⟩|s⟩`（无符号 EPR）。
copy-B 是**转置表示**：`R_B = Z R`，`Z = diag(1,−1,1,−1,…)`（`pᵀ=−p`）。
与当前包的玻色 doubled 构造一致。

**可归一化性**：玻色 Fock 空间无穷维；`|ρ⟩⟩` 必须在 Fock 空间内（有限范数）。
当前纯配对分支要求最大奇异值 `< 1`。一般复对称 `K` 的判据要在 §4.2/§9 推导；
构造时校验，失败抛 `ArgumentError`（带诊断量）。

**站点**：玻色的"站点"沿用模式对 `[p, n+p]`（与当前包 `osee` 的约定一致）。

### 1.3 与现有实现的关系（重要）

| 现有 | 新对象 | 说明 |
|---|---|---|
| `FermionicGaussianState(M)`，`M` 厄米 | `FermionicGibbsState(M)` | 归一化密度矩阵；`entropy`/`osee`（算符空间）保留 |
| `FermionicGaussianState(W)`，`W` 反对称 | `FermionicPureState(W; kind=:pairing)` | **态** `e^{½c†Wc†}|0⟩`，不是算子 `e^H` |
| `osee(s::Gibbs, modes)` | `operator_osee(gs, sites)`（或 `osee(asoperator(gs), sites)`） | 密度矩阵的算符空间纠缠；**不是** Gibbs 态的子系统熵 |
| `osee(s::PurePairing, modes)` | `osee(ps, sites)` = `entropy(ps, sites)` | 纯态自身的二分纠缠熵 |
| `entropy(s::Gibbs, sites)` | `entropy(gs, modes)` | Gibbs 态的子系统熵（模式指标） |
| `FermionicGaussianOperator`（新） | — | 一般非厄米 `e^H`；**新增核心** |

> **关键区分**：`e^H|0⟩`（态）与 `e^H`（算符）是两个不同对象。
> 当前纯配对分支是前者；新核心要后者。二者在 `H` 纯配对时**不相等**，
> 其 OSEE 一般也不同。API 必须用不同类型/构造器把二者分开。

### 1.4 特殊极限（必须退化到今天的结果）

1. `X` 厄米：`ρ = e^H = e^{−H_phys}`（差归一化）；`purify(ρ)` 的协方差 = 当前
   Gibbs doubled 构造的 `Γd`；OSEE = v1 的 `osee(gs, sites)` = v2 的
   `operator_osee(gs, sites)`。
2. `X = [[0,W],[0,0]]`（纯配对生成元）：`ρ = e^{½c†Wc†}`；`purify(ρ)` 是
   **算符**的矢量化，一般 **不等于** 态 `e^{½c†Wc†}|0⟩`。两者都要支持，
   测试里显式对比。
3. 纯态 `e^{½c†Wc†}|0⟩`：`FermionicPureState(W; kind=:pairing)`，其 `Γ` 与
   当前纯配对分支一致。
4. 玻色：`K` 实对称正定 → 当前 Gibbs；纯配对 `W` → 当前纯配对态。

---

## 2. 类型层次

```text
AbstractGaussian
├── AbstractGaussianOperator        # 非厄米二次型算子 ρ = e^H（未归一化）
│   ├── FermionicGaussianOperator
│   └── BosonicGaussianOperator
└── AbstractGaussianState           # 归一化态
    ├── AbstractPureState           # 纯态 |ψ⟩（norm 1）
    │   ├── FermionicPureState
    │   └── BosonicPureState
    └── AbstractGibbsState          # 混合态 ρ = e^{−H}/Z（Hermitian, positive, trace 1）
        ├── FermionicGibbsState
        └── BosonicGibbsState
```

### 2.1 `FermionicGaussianOperator`

```julia
struct FermionicGaussianOperator <: AbstractGaussianOperator
    X   :: Matrix{ComplexF64}   # 2m×2m，规范形式 [[h,A],[B,−hᵀ]]
    c   :: ComplexF64           # ρ = e^c · exp(½ψ†Xψ)
end
```

- 不变量：`size(X,1) == size(X,2)`、偶数维；`X` 已规范化（`X[2,2] = −X[1,1]ᵀ`）。
- 惰性缓存：`purification`（`FermionicPureState`，`4m` 维协方差）、`W = e^{−X}`、
  `logZ`、`isphysical`。
- `nmodes(op) = size(X,1) ÷ 2`。

### 2.2 `BosonicGaussianOperator`

```julia
struct BosonicGaussianOperator <: AbstractGaussianOperator
    K   :: Matrix{ComplexF64}   # 2n×2n，复对称 K = Kᵀ
    c   :: ComplexF64           # ρ = e^c · exp(½RᵀKR)
end
```

### 2.3 `FermionicPureState`

```julia
struct FermionicPureState <: AbstractPureState
    Γ   :: Matrix{Float64}       # 2N×2N Majorana 协方差，实反对称，Γ² = −I
end
```

- `N = size(Γ,1) ÷ 2` 个模；纯态由 `Γ` 唯一决定（up to 相位）。
- 站点：纯态有 `N` 个模；`N` 为偶数时有 `N÷2` 个站点，站点 `p` 对应模对
  `[p, N÷2 + p]`（见 §3.3）。
- 也允许由"态生成元"构造：`FermionicPureState(W; kind=:pairing)`。

### 2.4 `FermionicGibbsState`

```julia
struct FermionicGibbsState <: AbstractGibbsState
    M    :: Matrix{ComplexF64}    # 2n×2n 厄米规范 BdG，ρ = e^{−½ψ†Mψ}/Z
    Γ    :: Matrix{Float64}       # 2n×2n Majorana 协方差
    logZ :: Float64
end
```

- 不变量：`M` 厄米、规范；`Γ` 实反对称、`|ν|≤1`；`tr ρ = 1`。
- `densitymatrix(gs)` 用 `M`/`logZ` 构造；`entropy(gs, modes)` 用 `Γ`。
- 可以由 `M` 或 `Γ` 构造，二者惰性互算并校验一致。

### 2.5 `BosonicPureState` / `BosonicGibbsState`

同上，协方差为实对称 `2N×2N`；Gibbs 生成元 `K` 实对称正定；纯态辛本征值全 `½`。

### 2.6 为什么不用 `isbdg::Bool`

当前包用一个 `FermionicGaussianState` + `isbdg` 同时表示 "Hermitian Gibbs" 和
"纯配对算符"，导致：
- 分支歧义（厄米 ∧ 反对称）；
- `entropy`/`osee` 的语义依赖隐藏字段；
- 无法表示一般非厄米 `e^H`。

v2 用类型区分，构造时**不做歧义猜测**：算子只接受 `X`（任意复），态/Gibbs 只接受
对应生成元/协方差；需要哪种语义就构造哪种类型。

---

## 3. 公开 API

> 下面所有函数都在 `GaussianStates` 下导出（除特别标注 `internal` 的）。
> `α=1` 为 von Neumann，`α≠1` 为 Rényi。

### 3.1 构造

```julia
# ---- 算子：核心 ----
FermionicGaussianOperator(X::AbstractMatrix{<:Number}; const=0)
FermionicGaussianOperator(h::AbstractMatrix, A::AbstractMatrix,
                          B::AbstractMatrix; const=0)
BosonicGaussianOperator(K::AbstractMatrix{<:Number}; const=0)

# ---- Gibbs 态 ----
FermionicGibbsState(M::AbstractMatrix{<:Number})              # M 厄米（规范或非规范均可）
FermionicGibbsState(Γ::AbstractMatrix{<:Real}; kind=:covariance)
BosonicGibbsState(K::AbstractMatrix{<:Real})                  # K 实对称正定
BosonicGibbsState(Γ::AbstractMatrix{<:Real}; kind=:covariance)

# ---- 纯态 ----
FermionicPureState(Γ::AbstractMatrix{<:Real}; kind=:covariance)   # Γ 纯（Γ²=−I）
FermionicPureState(W::AbstractMatrix{<:Number}; kind=:pairing)    # e^{½c†Wc†}|0⟩
BosonicPureState(Γ::AbstractMatrix{<:Real}; kind=:covariance)
BosonicPureState(W::AbstractMatrix{<:Number}; kind=:pairing)      # 复对称，最大奇异值<1
```

**构造校验**（全部在构造时一次完成）：

| 输入 | 校验 | 失败 |
|---|---|---|
| `X` | 方阵、偶数维、有限 | `ArgumentError` |
| `h,A,B` | 尺寸一致；`A,B` 反对称（自动取反对称部分并警告/严格） | `ArgumentError` |
| `K` | 方阵、偶数维、`K = Kᵀ`（复对称） | `ArgumentError` |
| Gibbs `M` | 厄米（容差相对 1e-10） | `ArgumentError` |
| Gibbs `K` | 实对称、正定 | `ArgumentError` |
| 纯态 `Γ` | 实、对称/反对称、纯（费米 `‖Γ²+I‖`；玻色辛谱 `½`） | `ArgumentError` |
| 纯配对 `W` | 费米：复反对称；玻色：复对称、`σ_max(W)<1` | `ArgumentError` |
| 玻色算子 `K` | 复对称 + 可归一化判据（§4.2） | `ArgumentError` + 诊断量 |

**规范化的规则**：

- `FermionicGaussianOperator(X)`：把 `X` 化为规范形式
  `h=(X₁₁−X₂₂ᵀ)/2, A=X₁₂, B=X₂₁`，常数吸收 `½tr(X₂₂)+½tr(h)`（符号在实现时
  用稠密测试钉死）。**不改变** `ρ`。
- `FermionicGibbsState(M)`：同样规范化；若 `M` 非厄米直接报错（不再有
  "厄米 ∧ 反对称"的歧义：算子请用 `FermionicGaussianOperator`）。
- 纯配对 `W`：只取反称部分（`W−Wᵀ`）；实/复走同一代码路径（v2 不再分
  "实 SVD / 复 Stein"两个分支，见 §4.3）。

### 3.2 核心操作

```julia
# 算子 → 纯态（矢量化的 |ρ⟩⟩）
purify(op::AbstractGaussianOperator) -> AbstractPureState

# 纯化/约化
reduce(ps::AbstractPureState, modes::AbstractVector{<:Integer}) -> AbstractGibbsState
reduce(gs::AbstractGibbsState, modes::AbstractVector{<:Integer}) -> AbstractGibbsState

# 熵（Gibbs 态：整态/子系统；纯态：整态=0/子系统=纠缠熵）
entropy(gs::AbstractGibbsState; α=1) -> Float64
entropy(gs::AbstractGibbsState, modes; α=1) -> Float64
entropy(ps::AbstractPureState; α=1) -> Float64            # 恒 0
entropy(ps::AbstractPureState, sites; α=1) -> Float64    # 站点割

# OSEE
osee(op::AbstractGaussianOperator, sites; α=1) -> Float64
osee(ps::AbstractPureState, sites; α=1) -> Float64       # = entropy(ps, sites)
operator_osee(gs::AbstractGibbsState, sites; α=1) -> Float64
    # 密度矩阵的算符空间纠缠（v1 的 osee(gs,·) 语义）；
    # 等价于 osee(asoperator(gs), sites)。v1 的 osee(gs,·) 在兼容层保留。

# Gibbs 态 ↔ 算子（v1 的 osee(gs,·) 语义）
asoperator(gs::AbstractGibbsState) -> AbstractGaussianOperator   # ρ = e^{−H}/Z 作为算子

# 访问器
covariance(x::AbstractGaussian) -> Matrix{Float64}
nmodes(x::AbstractGaussian) -> Int          # 模数（纯态/算子/Gibbs 各自的模）
nsites(x::AbstractGaussian) -> Int          # 纯态 N÷2；算子 n；Gibbs n
bdg(x::Union{AbstractGaussianOperator,AbstractGibbsState}) -> Matrix{ComplexF64}
blocks(op::FermionicGaussianOperator) -> (h, A, B)
constant(x::AbstractGaussianOperator) -> ComplexF64
logZ(gs::AbstractGibbsState) -> Float64
densitymatrix(gs::AbstractGibbsState) -> Matrix{ComplexF64}
densitymatrix(ps::AbstractPureState) -> Matrix{ComplexF64}   # |ψ⟩⟨ψ|
site_modes(ps::AbstractPureState, sites) -> Vector{Int}      # 站点 → 协方差模指标

# 语义查询
ispure(s)  -> Bool
isgibbs(s) -> Bool
isphysical(s) -> Bool
```

**`entropy` 与 `osee` 的分工（必须写进文档）**：

| 对象 | `entropy` | `osee` |
|---|---|---|
| `GaussianOperator` | 无定义（不是态） | **核心**：`entropy(reduce(purify(op), sites))` |
| `GibbsState` | 态的熵（整态/子系统） | 无（旧行为改名为 `operator_osee`） |
| `PureState` | 整态 0 / 子系统的纠缠熵 | 同 `entropy(ps, sites)` |

> v1 的 `osee(gs, sites)`（密度矩阵的算符空间纠缠）与 `entropy(gs, modes)`
> （ρ 自身的子系统熵）是**两个不同的量**。v2 把前者改名 `operator_osee`，
> `osee` 只用于算子/纯态，`entropy` 只用于态。兼容层保留 v1 的
> `osee(gs, ·)` 并 `@deprecate`。

### 3.3 站点 vs 模式（v2 的核心约定修正）

当前包把所有指标都叫 `sites`，用户必须自己拼 `[p, m+p]`。v2 区分：

- **模式（mode）**：协方差 `Γ` 的行/列指标，`1 .. nmodes(Γ)`。
  - 纯态 `N` 个模：`Γ` 是 `2N×2N`，模式 `1..N`。
  - Gibbs 态 `n` 个模：`Γ` 是 `2n×2n`，模式 `1..n`。
- **站点（site）**：物理分组。
  - 纯态 `N` 个模（`N` 偶）：`N÷2` 个站点；站点 `p` ↔ 模 `[p, N÷2+p]`。
  - 算子 `n` 个模：`purify(op)` 是 `2n` 模的纯态、`n` 个站点；站点 `p` ↔ 模
    `[p, n+p]`（即 `c_p, d_p`）。
  - Gibbs 态 `n` 个模：站点 `p` ↔ 模 `p`（一个模就是一个站点；其 2 个 Majorana
    是 `2p−1,2p`）。

**函数签名里的指标类型**：

- `osee(op_or_ps, sites)`：`sites` 是**站点**（1..n）。内部 `site_modes` 展开。
- `entropy(gs, modes)`：`modes` 是 **Gibbs 态的模式**（1..n）。
- `reduce(ps, modes)`：`modes` 是**纯态的模式**（1..2N）。
- `reduce(gs, modes)`：`modes` 是 **Gibbs 态的模式**（1..n）。
- `nsites(ps) = nmodes(ps) ÷ 2`，`nsites(op) = nmodes(op)`，
  `nsites(gs) = nmodes(gs)`。

> 这样 `osee(op, 1:k)` 就是"前 k 个 IF 站点"的 OSEE，不需要用户知道
> `[p, n+p]`。`osee(op, sites; kind=:modes)` 可显式按裸模式割（专家用）。

**费米 Z₂ 约定**（沿用 v1 的结论）：`entropy`/`osee` 的子系统约化是**代数
（模式）约化**，不是占据数基朴素偏迹；连续站点块/单站点下二者一致，非连续
站点块下必须用同一个约定做对照（不能拿朴素偏迹当参照）。文档里引用
Friis–Lee–Bruschi / Amosov–Filippov。

### 3.4 归一化与常数

- `GaussianOperator` 的 `c`（`ρ = e^c e^{½ψ†Xψ}`）只影响整体标量：
  - `purify(op)` 返回的纯态**归一化**（`|ρ⟩⟩/‖ρ‖`）；
  - `logZ(op) = c + …` 记录 `tr ρ†ρ` 或 `tr ρ`（取决于定义，见 §9）；
  - OSEE 与 `c` 无关（测试：`osee(op, sites) == osee(op_with_c_shifted, sites)`）。
- `GibbsState` 的 `logZ` 是 `log tr e^{−½ψ†Mψ}`，用于 `densitymatrix`。

### 3.5 使用示例

```julia
using GaussianStates

# 1) 一般非厄米 IF 算子：H = Σ Δ_pq ā_p a_q 的"算符版"（示例构造）
X = build_if_bdg(Δ)                      # 2n×2n 一般复（用户自己的离散化）
op = FermionicGaussianOperator(X)
S  = osee(op, 1:n)                       # 全部站点 → 0
S1 = osee(op, 1)                         # 站点 1 的 OSEE
Sk = [osee(op, 1:k) for k in 1:n]        # bond profile

# 2) 显式走对象链
psi  = purify(op)                        # |ρ⟩⟩，2n 模纯高斯态
rhoA = reduce(psi, site_modes(psi, 1))   # 站点 1 的 Gibbs 态（2 模）
S1   = entropy(rhoA)                     # = osee(op, 1)
@assert covariance(rhoA) === covariance(reduce(purify(op), site_modes(purify(op), 1)))

# 3) 特例：Hermitian H → Gibbs 态；纯配对 → 纯态
gs = FermionicGibbsState(M)               # ρ = e^{−½ψ†Mψ}/Z
ps = FermionicPureState(W; kind=:pairing) # |Ψ⟩ = e^{½c†Wc†}|0⟩
```

---

## 4. 算法

### 4.1 费米：一般非厄米算子的 `purify`

目标：给定 `X`（2m×2m 一般复），O(m³) 算出 `|ρ⟩⟩` 的 `4m×4m` 纯态协方差 `Γ_d`。

**路线：用 ρ 对 Nambu 自旋量的线性伴随作用 + EPR 镜像条件。**

1. **伴随作用**。对 `H = ½ψ†Xψ`，可证

   ```text
   [H, ψ_a] = −(X ψ)_a        ⟹      ρ ψ_a ρ^{−1} = (e^{−X} ψ)_a =: (W ψ)_a
   ```

   （`W = e^{−X}`，2m×2m；非厄米时 `W` 不是酉的，但仍是线性变换。）

2. **EPR 镜像**。`|I⟩_G = e^{Σ c†d†}|0⟩` 下，c 模式算符到 d 模式算符的镜像映射
   （charge conjugation）为

   ```text
   M(c_i) = d_i† ,      M(c_i†) = −d_i       （符号/模式序由 JW 约定定，稠密测试钉死）
   ```

3. **湮灭子空间**。设 `α(u) = u·ψ_c + v·ψ_d`，要求 `α(u)|ρ⟩⟩ = 0`。用
   `ρψ = (Wψ)ρ` 与 EPR 镜像，得到

   ```text
   v = −M(Wᵀ u)        （即 v 由 u 唯一确定）
   α(u) = u·ψ_c − M(Wᵀ u) ,     u ∈ C^{2m} 任意
   ```

   这给出 `2m` 个线性独立的湮灭算符（`W` 可逆时）。

4. **由湮灭子空间重构协方差**。`α = V Ψ`（`V` 为 `2m×4m`），纯高斯态是 `α` 的
   真空。用标准公式（与 v1 `_majorana_covariance` 同一套 `γ` 约定）由 `V` 得到
   `Γ_d`；最后 `Γ_d² = −I` 校验（纯态），并做反对称化/取实清理。

5. **约化与熵**。`Γ_A = Γ_d[idx,idx]`，`eigvals(iΓ_A)=±ν`，
   `S = Σ h((1+ν)/2)`。`reduce(ps, modes)` 返回 `FermionicGibbsState(Γ_A)`；
   `Γ_A` 的厄米生成元 `M_A` 由逆映射（`M = 2·atanh` 型，见 §4.3）得到。

**为什么这是 O(m³)**：`W = e^{−X}` 是 2m×2m 矩阵指数；`V` 是 2m×4m；
`Γ_d` 是 4m×4m；每割本征值 O(|A|³)。全部是稠密小矩阵运算。

**特殊极限**：
- `X` 厄米：`W` 酉；上式退化为当前 Gibbs doubled 构造（`Γ_d²=−I`，A/B 块 =
  ρ² 关联，交叉块 `t̃=1/(2cosh λ)`）；测试逐元素对齐 v1。
- `X` 纯配对 `[[0,W],[0,0]]`：`e^{−X} = I − X`（`X²=0`），`α` 退化为
  `c − W d†` 型；与态 `e^{½c†Wc†}|0⟩` 的湮灭条件对比，二者不同（§1.4），
  测试显式区分。

> **待钉死的约定**：EPR 镜像的符号（`d†` vs `−d`、模式序、`S₀` 的左/右乘）。
> 这些在 v1 的 `_majorana_covariance` / `_fermionic_gibbs_doubled_gamma` 里已有
> 一套自洽约定；v2 先用小 m（m=1,2）的稠密 `Γ_d = i⟨γ_aγ_b⟩` 逐元素对齐，
> 再推广。这是 §8 的第一个工作包。

### 4.2 玻色：一般复对称 K 的 `purify`

目标：给定 `K`（2n×2n 复对称），算 `|ρ⟩⟩` 的 `4n×4n` 纯态协方差。

**路线：Bloch–Messiah / 复辛分解 + TMSV 乘积。**

1. 复辛伴随作用：`e^H R e^{−H} = W R`，`W ∈ Sp(2n,C)`（`W Ω Wᵀ = Ω`）。
2. `W` 的 Bloch–Messiah 分解：`W = S₁ D S₂`（`S_i` 复辛、`D` 正实对角压缩）。
   非厄米 `H` 下 `S_i` 是复辛的；`D` 的奇异值给出 TMSV 参数。
3. `|ρ⟩⟩` 是 `⊗_k TMSV(r_k)` 经 `S₁,S₂` 的复辛变换；协方差由 `TMSV` 的
   `X₀=½coth(ω)I`、`Y₀=½csch(ω)I` 与 `S` 变换组装（与 v1 的
   `_bosonic_doubled_covariance` 同构，但 `S` 复辛、`Y` 带转置/共轭修正）。
4. **可归一化判据**：TMSV 参数 `|tanh r_k| < 1`；等价于 `W` 的奇异值/复辛谱
   落在单位盘内。构造时校验，失败抛 `ArgumentError`（附 `max|tanh r|`）。
5. 约化/熵同 v1（玻色无 Z₂ 问题；协方差子矩阵直接可用）。

**特殊极限**：`K` 实对称正定 → v1 Gibbs doubled（`X=½coth ω`、
`Y=½csch ω`、`Z` 修正）；纯配对 `W` → v1 `_bosonic_gamma_pairing`。

> 玻色一般非厄米 `K` 的复辛分解是 §8 的第二个工作包；若第一版只做"实对称
> 正定 + 纯配对 + 复对称纯配对"三个特例，也应先把**类型和 API** 定下来，
> 一般 `K` 走 `ArgumentError("暂不支持")` 或稠密回退（小 n）。

### 4.3 `h → Γ` 的闭式（v2 统一）

**费米纯配对 `W`（任意复反对称）**：v1 的 `_correlations_stein` 已换成闭式

```text
G = (I + W†W)⁻¹ W†W = I − (I+W†W)⁻¹ ,    F = W G − W ,    Γ = _majorana_covariance(G,F)
```

O(n³)、恒良态；v2 直接沿用。**不再需要** v1 的 n²×n² Kronecker Stein 系统
（v1 在 `s_i s_j=1` 时静默错值；已在 backup 中修掉）。

**费米 Gibbs `M`（厄米）**：v1 公式（`Ū f(Λ) Uᵀ`、`f(λ)=1/(1+e^λ)`）保留；
`M → Γ` 是 O(n³)。

**费米 Gibbs `Γ → M`（逆映射，供 `reduce(ps,·)` 构造 `GibbsState`）**：
由 `Γ` 的谱 `±iν`（`ν=1−2f`）反解 `f=(1−ν)/2`，`λ=log((1−f)/f)`；
在 `M` 的本征基下拼回。零模 `f=½` 时 `λ=0`，需规范固定（v1 已有零模
gauge-fix 逻辑，v2 沿用）。**注意**：`GibbsState` 由 `Γ` 构造时，`M` 只在
`entropy` 不需要；`densitymatrix` 才需要，可惰性计算。谱端点 `ν=1`（纯）
或 `ν=0`（零占据）对应 `M` 的 `±∞` 本征值：`GibbsState` 以 `Γ` 为主表示，
`M` 惰性/截断，`densitymatrix` 对端点报错或按 `exp(−Λ)` 截断。

### 4.4 约化与熵

- **纯态 → Gibbs**：`Γ_A = Γ[idx,idx]`（`idx = site_modes...`），
  `FermionicGibbsState(Γ_A; kind=:covariance)`；`entropy(gs)` 用 `Γ_A`。
- **Gibbs → Gibbs**：`Γ_A = Γ[modes,modes]`，同型。
- **费米熵**：`eigvals(iΓ_A)=±ν`，`S=Σh((1+ν)/2)`；Rényi 同 v1。
- **玻色熵**：`eigvals(Ω_A Γ_A)=±iν`，`ν≥½`，`S=Σ[(ν+½)ln(ν+½)−(ν−½)ln(ν−½)]`。
- **数值保护**：费米 `ν` clip 到 `[0,1]`；玻色 `ν` clip 到 `[½,∞)`；偏离超过
  容差时抛 `ArgumentError`（而不是静默 clip），`isphysical` 可查询。

### 4.5 复杂度

| 操作 | 复杂度 | 说明 |
|---|---|---|
| `FermionicGaussianOperator(X)` | O(m³) | 规范化 + 可选 `W=e^{−X}`（惰性） |
| `purify(op)`（费米一般） | O(m³) | `W`、`V`、`Γ_d`；缓存 |
| `purify(gs)`（厄米） | O(m³) | v1 doubled 构造 |
| `purify(ps)` | O(1) | 纯态 `Γ_d` 的组装（若已缓存） |
| `reduce` + `entropy` | O(|A|³) | 子矩阵本征值 |
| `osee(op, sites)` | O(m³ + |A|³) | 首次 `purify` 后每割 O(|A|³) |
| `densitymatrix` | O(2^m) | 仅调试 |

---

## 5. 约定、边界与错误处理

### 5.1 归一化

- `purify(op)` 返回**归一化**纯态；`⟨⟨ρ|ρ⟩⟩ = 1`。
  - 费米：`‖ρ‖² = tr(ρ†ρ) = |e^c|² tr(e^{½ψ†Xψ} e^{½ψ†X†ψ})`（Gaussian 积分）。
  - 玻色：同型（若有限）。
- `logZ(op)`：`log tr(ρ†ρ)`（或 `log tr ρ`，二选一并在文档固定）；OSEE 不用它。

### 5.2 纯态/Gibbs 的物理性

- 费米纯态：`Γ² = −I`（机器精度 1e-10）；Gibbs：`|ν|≤1`。
- 玻色纯态：辛谱全 `½`；Gibbs：`ν≥½`。
- 玻色算子：可归一化判据（§4.2）。
- 失败时 `isphysical=false` + 构造报错；不做静默投影（除 v1 已有的 `_realify`
  类数值清理）。

### 5.3 站点/模式越界

- `osee(op, sites)`：`sites ⊆ 1:nmodes(op)`；空集 → `0.0`；重复 → `ArgumentError`。
- `reduce(ps, modes)`：`modes ⊆ 1:nmodes(ps)`；重复 → `ArgumentError`。
- `entropy(gs, modes)`：`modes ⊆ 1:nmodes(gs)`。
- 非连续站点块：允许（费米用 Z₂ 约定；文档写明与朴素偏迹的区别）。

### 5.4 常数/相位

- `purify` 对 `ρ` 的整体标量不敏感；`osee` 与 `c` 无关（测试）。
- 纯态的整体相位 `e^{iθ}` 不影响 `Γ`；`densitymatrix(ps)` 也不受影响。

### 5.5 与 v1 的对照

- 费米 Gibbs：`purify(gs)` 的 `Γ_d` 必须逐元素等于 v1 `GaussianStates._gammad`。
- 费米纯配对：`FermionicPureState(W; kind=:pairing)` 的 `Γ` 等于 v1
  `covariance(FermionicGaussianState(W))`。
- 玻色：同型。
- 旧 `osee(gs, modes)` 的 `sites` 是 doubled-mode 指标；v2 的
  `operator_osee(gs, sites)` 是站点。适配器负责转换（§7.3）。

---

## 6. 测试计划

### 6.1 稠密仲裁（核心）

**费米**：小 `m`（1,2,3）随机复 `X`：
1. 直接构造 `2^m × 2^m` 的 `ρ = exp(½ψ†Xψ)`（JW 矩阵指数）；
2. 矢量化成 `2^{2m}` 的 `|ρ⟩⟩ = (ρ⊗I)|I⟩_G`（含正确 Z₂ 符号）；
3. 逐元素算 `Γ_d = i⟨γ_aγ_b⟩`；
4. 与 `covariance(purify(FermionicGaussianOperator(X)))` 对比（1e-10）。
5. 对每个站点割：稠密 `|ρ⟩⟩` 的（graded）约化熵 = `osee(op, sites)`。

**玻色**：小 `n`（1,2）Fock 截断（`nbas=20–40`），同样对比 `Γ_d` 和割熵
（截断误差 1e-6~1e-8）。

### 6.2 特殊极限

- `X` 厄米：`purify` ≡ v1 doubled；`osee` ≡ v1。
- `X = [[0,W],[0,0]]`：`purify(op)` 的 `Γ_d` 与态 `e^{½c†Wc†}|0⟩` 的 `Γ` **不同**；
  两者都对照稠密。
- 纯配对 `W`（复、实）：`Γ` ≡ v1 `_correlations_stein` 闭式。
- 玻色 `K` 实对称正定 / 纯配对：≡ v1。
- GMPS 联合回归：`osee(op, sites)` 对四组 contour（虚时/实时/混合/BCS）与
  `entanglement_entropy(mpsI; bond)` 一致（复用 `osee_gauss_joint.jl`）。

### 6.3 性质测试

- 纯度：`Γ_d² = −I`；`Γ_d` 实反对称。
- 归一化：`tr(densitymatrix(gs)) = 1`；`norm(densitymatrix(ps))`...（纯态秩 1）。
- 整体标量无关：`osee(op, sites) == osee(GaussianOperator(X; const=c), sites)`。
- 互补：纯态 `S(A) = S(Ā)`（费米 graded；玻色）。
- 单调性/边界：`S(∅)=0`、`S(全)=0`；单站点极限闭式。
- Rényi：`α→1` 连续；`α=2` 与 `−log tr ρ_A²` 稠密一致。
- 零模/奇异 `X`：`e^H` 奇异时仍能算（`|ρ⟩⟩` 非零）；`X` 非正规（`[X,X†]≠0`）
  时稠密对照。
- 错误路径：非方阵、奇数维、非厄米 `M`、非正定 `K`、超界 `sites`、重复指标。

### 6.4 性能

- 费米 `m=16,32,64`：`purify` + 全 bond profile 的耗时（目标 O(m³)）。
- 玻色 `n=8,16`：同上。
- 与 v1 的 `osee` 对比（常数因子可接受，但不能退化到指数）。

---

## 7. 文件布局与迁移

### 7.1 新文件布局（建议）

```text
src/
├── GaussianStates.jl          # module、导出、include
├── abstract.jl                # 抽象类型、公共访问器、站点/模式工具
├── common.jl                  # 熵内核（费米/玻色）、数值工具、Ω/S₀
├── fermionic/
│   ├── operator.jl            # FermionicGaussianOperator + 规范化
│   ├── purify.jl              # 一般非厄米 purify（W=e^{−X}、镜像、Γ_d）
│   ├── pure.jl                # FermionicPureState（含纯配对构造）
│   ├── gibbs.jl               # FermionicGibbsState + M↔Γ
│   ├── correlations.jl        # G,F ↔ Γ、Stein 闭式、SVD 闭式
│   └── entropy.jl             # 费米熵/约化
├── bosonic/
│   ├── operator.jl
│   ├── purify.jl              # 复辛 Bloch–Messiah / 特殊极限
│   ├── pure.jl
│   ├── gibbs.jl
│   ├── covariance.jl
│   └── entropy.jl
├── osee.jl                    # osee 泛型 + 站点展开 + 文档
├── densitymatrix.jl           # 稠密参考（调试）
└── deprecated.jl              # v1 兼容层（FermionicGaussianState 等）
```

测试：`test/operator.jl`、`test/purify.jl`、`test/pure.jl`、`test/gibbs.jl`、
`test/osee.jl`、`test/dense.jl`、`test/migration.jl`、`test/gmps.jl`。

### 7.2 v1 → v2 迁移表

| v1 | v2 | 备注 |
|---|---|---|
| `FermionicGaussianState(M)`（厄米） | `FermionicGibbsState(M)` | 语义不变 |
| `FermionicGaussianState(W)`（反对称） | `FermionicPureState(W; kind=:pairing)` | 语义不变（纯态） |
| `BosonicGaussianState(K)`（实对称 PD） | `BosonicGibbsState(K)` | 语义不变 |
| `BosonicGaussianState(W)`（复对称/非 PD） | `BosonicPureState(W; kind=:pairing)` | 语义不变 |
| `osee(s::Gibbs, modes)` | `operator_osee(gs, sites)`（或 `osee(asoperator(gs), sites)`） | 语义不变，改名；**指标**从 doubled 模式改为站点 |
| `osee(s::PurePairing, modes)` | `osee(ps, sites)` | **指标语义变化** |
| `entropy(s::Gibbs, sites)` | `entropy(gs, modes)` | v1 的 `sites` 就是模式 |
| `covariance(s)` | `covariance(x)` | 不变 |
| `GaussianStates._gammad` | `covariance(purify(gs))` | 内部 |
| — | `FermionicGaussianOperator(X)` | 新增核心 |
| — | `purify(op)`、`reduce(ps, modes)` | 新增核心 |

**兼容层**（`deprecated.jl`）：

```julia
FermionicGaussianState(M; kind=:auto) =
    kind === :pairing || _isantisym(M) ? FermionicPureState(M; kind=:pairing) :
                                         FermionicGibbsState(M)
# 但 v1 的 "厄米 ∧ 反对称" 歧义在 v2 里显式化：
#   kind=:auto 时优先 Gibbs 并 @warn；kind=:pairing 强制纯态。
```

**下游适配**：
- QuAPI `BdG`：`b.state::FermionicPureState`；`osee(b, left_sites)` 的 `left_sites`
  是站点（当前 `left` 是 a/ā 成对的模式列表，正好一一对应）。
- GTEMPO `/osee/osee_gauss_joint.jl`：`pairing_state(Δ) = FermionicPureState(W;
  kind=:pairing)`；`osee(s, acc_sites)` 里把 `acc` 从模式列表换成站点列表。

### 7.3 站点适配器（兼容旧调用）

```julia
# v1 的 osee(s, modes) 里 modes 是 doubled 模式；v2 提供显式入口：
osee(ps, modes; kind=:modes)   # 裸模式
osee(ps, sites)                # 站点（默认）
```

如果不想破坏旧脚本，可以在 `deprecated.jl` 里给旧类型保留 `osee(old, modes)`，
新类型只走站点 API；迁移期两者并存，文档给出对照表。

---

## 8. 工作包与里程碑

### M0：类型与骨架（1–2 天）
- 抽象类型、六个具体类型、构造校验、`covariance`/`nmodes`/`site_modes`。
- `FermionicGibbsState`（= v1 Gibbs）与 `FermionicPureState`（= v1 纯配对）
  的构造与协方差；`purify(gs)` 复用 v1 `_gammad`。
- 旧 API 兼容层；测试全绿（v1 的 259 项）。

### M1：费米一般非厄米 `purify`（核心，3–5 天）
- `W = e^{−X}`、EPR 镜像、湮灭子空间、`Γ_d` 重构。
- m=1,2,3 稠密仲裁（`ρ=e^H` 的 JW 矩阵指数 + 矢量化的 `i⟨γγ⟩`）。
- 与 v1 的厄米/纯配对极限对齐；`osee` 站点 API。
- 这是**最需要小心**的一步：镜像符号、Z₂、共轭位置。建议先写一个
  `debug/` 下的最小稠密脚本，把 m=1 的 `c − λd†` / `d + λc†` 对上，再推广。

### M2：费米 Gibbs/纯态/约化的统一（1–2 天）
- `reduce(ps, modes) -> GibbsState`；`Γ → M` 逆映射；零模。
- `osee` 与 `entropy` 的语义测试（算符空间 vs 子系统）。

### M3：玻色（2–4 天）
- 实对称 PD Gibbs、纯配对（实/复）——直接复用 v1。
- 一般复对称 `K` 的 `purify`（复辛 Bloch–Messiah）；可归一化判据。
- 稠密 Fock 仲裁。

### M4：文档、迁移、回归（2–3 天）
- manual 重写：对象链、站点/模式、Z₂、算法、迁移表。
- v1 API 兼容层 + deprecation。
- GMPS 联合回归（四 contour）。
- 性能基准（m=16/32/64）。

### M5（可选）：便利操作
- `apply(op, ps)`：高斯算子作用于高斯态（`e^H|0⟩` 等）。
- 张量积 `⊗`、部分转置、纠缠谱、互信息。
- 多副本/多时间 IF 的直接构造。

---

## 9. 开放问题（需要在实现前定/验证）

1. **EPR 镜像的精确符号**：`M(c_i)=d_i†`、`M(c_i†)=−d_i` 的模式序与 JW 符号；
   m=1、m=2 的稠密逐元素对齐是硬门槛。
2. **`purify` 的归一化与 `logZ`**：`tr(ρ†ρ)`（Hilbert–Schmidt）还是 `tr ρ`？
   OSEE 与两者都无关，但 `densitymatrix`/`weight` 需要一个定义。
3. **玻色一般复对称 K 的可归一化判据**：复辛谱/奇异值条件；边界情形
   （`|tanh r|=1`）是否允许极限。
4. **`Γ → M` 逆映射的零模/奇异情形**：`f=½` 时的规范固定；v1 的 gauge-fix
   是否足够。
5. **非连续站点割**：费米 Z₂ 约定的文档与测试（沿用 v1 的结论：代数约化，
   非朴素偏迹）。
6. ~~`osee(gs, sites)` 的语义~~：**已决定**——改名 `operator_osee(gs, sites)`；
   `osee` 只用于算子/纯态，`entropy` 只用于态；v1 的 `osee(gs, ·)` 进兼容层。
7. **`FermionicGaussianOperator` 的 `X` 非规范输入**：是否允许直接存非规范
   `X`（惰性规范化），还是构造时强制规范化（推荐）。
8. **性能**：`W=e^{−X}` 对 `m=128` 的 2m×2m 矩阵指数（256×256）是 O(m³)≈1.7e7，
   可接受；`Γ_d` 4m×4m（512×512）本征值也是 O(m³)。需要基准。
9. **与 QuAPI/GTEMPO 的站点映射**：`(branch,time,band)` → 站点的编号规则；
   `osee_bond` 的适配。

---

## 10. 一句话总结

v2 的核心是**把"非厄米二次型算子 `ρ=e^H`"作为第一类对象**，用
`purify(ρ) → |ρ⟩⟩` 把它变成 doubled 空间的纯高斯态，再用
`reduce(|ρ⟩⟩, A) → Gibbs 态` 和 `entropy` 得到 OSEE。
Gibbs 态是这条链的**终点**（约化态），不是起点；纯配对态 `e^{½c†Wc†}|0⟩`
是**另一个**对象（算符作用于真空），二者必须用不同类型/构造器分开。
