# Fermionic/BosonicQuadraticOperator、`*Exp`、GaussianState —— v7 API

状态：**设计草案 v7**（抽象继承 + 厄米变体）
日期：2026-10-29
备份：`backup/GaussianStates`；旧计划（v0）：`plan/api_plan_v0_backup.md`

本版相对 v3 的修订：

1. **不用参数化 `QuadraticOperator{Fermion}`**：直接用
   `FermionicQuadraticOperator` / `BosonicQuadraticOperator`；重载
   `exp(::FermionicQuadraticOperator) -> FermionicQuadraticOperatorExp`
   （玻色同理）。
2. `GaussianState` 的协方差表示（Bogoliubov 或 xp/Majorana）**属于内部细节**，
   文档写清即可；用户不需要直接接触。本版选 **xp/Majorana 实协方差**（熵公式
   最标准、v1 已验证），`purify` 内部从 Bogoliubov block 变换过去。
3. **pairing 的关键澄清**：GMPS 的系数态就是 `e^H` 的纯化，其 bond 熵
   **就是**算子 OSEE；v1 的 `pairing_state(W)` 只是纯配对特例。BCS/一般
   Gaussian 需要一般 Gaussian 纯态（`GaussianState`/`purify`），
   `GaussianState(W; kind=:pairing)` 不进核心 API（纯配对可直接
   `purify(exp(FermionicQuadraticOperator(0,W,0)))`）。
4. **厄米变体**：`Fermionic/BosonicQuadraticHermitianOperator` + `*Exp`；
   `GaussianState(exp(H_herm))` 直接给 Gibbs 态（§1.5/§2.5）。
5. **抽象类型/继承**：`AbstractQuadraticOperator` / `*OperatorExp` /
   `*GibbsState` 三层；`purify` 可作用于任何 `AbstractQuadraticGibbsState`
   （TFD，§0.1/§2.5）。

---

## 0. 逻辑与目标

### 0.1 对象与继承

```text
AbstractQuadraticOperator
├── AbstractFermionicQuadraticOperator
│   ├── FermionicQuadraticOperator              # H 一般（非厄米）
│   └── FermionicQuadraticHermitianOperator     # H 厄米
└── AbstractBosonicQuadraticOperator
    ├── BosonicQuadraticOperator
    └── BosonicQuadraticHermitianOperator

AbstractQuadraticOperatorExp
├── AbstractFermionicQuadraticOperatorExp
│   └── FermionicQuadraticOperatorExp           # e^H（一般 H）
├── AbstractBosonicQuadraticOperatorExp
│   └── BosonicQuadraticOperatorExp             # e^H（一般 H）
└── AbstractQuadraticGibbsState
    ├── FermionicQuadraticGibbsState   # e^H（厄米 H）
    └── BosonicQuadraticGibbsState     # e^H（厄米 H）

具体态：FermionicGaussianState / BosonicGaussianState（只存协方差）
```

- 厄米 `e^H` 是 Gibbs 密度矩阵，所以 `*QuadraticGibbsState`
  **同时**是 `AbstractQuadraticGibbsState`；一般 `*QuadraticOperatorExp`
  不是 Gibbs 态。
- `purify(op::AbstractQuadraticOperatorExp)`：
  - 一般 `*QuadraticOperatorExp` → 算子矢量化（IF/Choi 态，纯）；
  - `AbstractQuadraticGibbsState` → **thermofield double**（`√ρ = e^{H/2}`
    的矢量化，纯），约化态 = Gibbs 态 `ρ`。
- `GaussianState(gs::AbstractQuadraticGibbsState)` → Gibbs 态 `ρ`（混合）。
- 具体态 `FermionicGaussianState`/`BosonicGaussianState` 只存协方差（实
  Majorana/(x,p)）。

### 0.2 主工作流

```julia
H   = FermionicQuadraticOperator(h, A, B)
op  = exp(H)                       # FermionicQuadraticOperatorExp
psi = purify(op)                   # FermionicGaussianState（纯态）
S   = entropy(psi, modes)          # OSEE = |ρ⟩⟩ 的子矩阵熵
rho = partialtr(psi, modes)        # 子系统 GaussianState（一般混合 = Gibbs）
@assert entropy(rho) ≈ S
```

厄米 Hamiltonian（Gibbs 态）：

```julia
Hh  = FermionicQuadraticHermitianOperator(h, A)
gs  = exp(Hh)                      # FermionicQuadraticGibbsState（Gibbs 态）
rho = GaussianState(gs)            # Gibbs 态（n 模式，混合）
psi = purify(gs)                   # TFD（2n 模式，纯）
@assert partialtr(psi, 1:n) ≈ rho
```

- **不提供 `osee`**：`purify` 是 O(n³)，调用一次、复用 `psi`。
- `GaussianState` 只有协方差（实 Majorana/(x,p)），没有 bdg/M 字段。
- 包内只认 1..N 的物理模式和子矩阵；`(branch,time,band)` 由下游胶水处理。

### 0.3 目标核对

| 目标 | 实现 |
|---|---|
| 一般非厄米 `H` | `Fermionic/BosonicQuadraticOperator`（3 独立 block） |
| `ρ=e^H` 不是密度矩阵 | `*QuadraticOperatorExp` 与 `GaussianState` 分离 |
| OSEE = `|ρ⟩⟩` 的二分熵 | `entropy(purify(op), modes)` |
| 子系统是 Gibbs 态 | `partialtr(psi, modes)` 返回 `GaussianState` |
| 反复 purify 贵 | 只提供 `purify`，不提供 `osee` |
| 纯配对/BCS | 见 §0.4：GMPS 系数态 = 纯化；pairing 是特例，BCS 用一般 `GaussianState`/`purify` |
| 玻色/费米 | 独立的具体类型 |

### 0.4 IF 的 OSEE：GMPS = `e^H` 的纯化（更正）

你指出的关键：**后端的 GMPS 就是 `e^H` 的纯化，其 bond 熵就是 OSEE**；
v1 的 `pairing_state` 只是纯配对的特例。上一轮我拿 `exp(Σ Δ d a)`（普通指数）
当 IF 算子，得到"两个量差 0.4–0.5"，那是**算子取错**：

- IF 泛函 `I[ā,a] = exp(Σ Δ ā a)` 对应的算子是**相干态矩阵元等于该泛函**的
  算子 `O`（单模 `O = 1 + λ c†c = :exp(λ c†c):`，normal-ordered Gaussian），
  不是普通指数 `exp(Σ Δ c†c)`。
- 系数态 `|Ψ_IF⟩ = exp(Σ Δ c̄†c'†)|0⟩` 的相干态波函数就是 `I[ā,a]`；
  它正是 `O` 的矢量化 `|O⟩⟩ = (O⊗I)|I⟩`（Choi 态）。
- 所以 `GMPS bond 熵 = 系数态纠缠 = O 的 OSEE`，**同一个量**。
  上一轮用普通指数 `exp(Σ Δ d a)` 去比，才出现假差异。

**BCS bath**：IF 是一般 Gaussian 泛函（number-conserving + pairing），
其系数态是一般 Gaussian 纯态，**不满足** `pairing_state`（纯配对）的要求；
但 GMPS 表示的正是这个一般 Gaussian 系数态，其 bond 熵仍是 `e^H` 的 OSEE。
所以：

- 后端 GMPS 的 OSEE **没有错**。
- `pairing_state` 太窄：只覆盖纯配对；BCS 需要**一般 Gaussian 纯态**
  （系数态）或等价的 `purify(e^H)`。
- 计划的核心 API 用 `purify(exp(H))`（一般纯态，任意 Gaussian 协方差）+
  `partialtr`；不提供 `pairing_state`，纯配对/BCS 都由 `purify` 覆盖。

> 更正记录：上一轮 §0.4 的"系数态 vs 算子 OSEE 差 0.4–0.5"是把普通指数
> `exp(Σ Δ d a)` 当成 IF 算子造成的；正确的 IF 算子是 normal-ordered
> Gaussian，其矢量化就是系数态，二者 bond 熵一致。

---

## 1. `Fermionic/BosonicQuadraticOperator`

### 1.1 统一记号

```text
Ψ = (a₁ … a_n , a₁† … a_n†)ᵀ ,   H = ½ Ψ† X Ψ + c₀
X = ⎡ X₁₁ X₁₂ ⎤   （四个 n×n block；只存独立 block）
    ⎣ X₂₁ X₂₂ ⎦
```

`a` 对费米是 `c`、对玻色是 `a`。

### 1.2 费米

```text
½ Ψ†XΨ = c† h c + ½(c† A c† + c B c) + ½ tr(X₂₂)

h = (X₁₁ − X₂₂ᵀ)/2        n×n 一般复
A = (X₁₂ − X₁₂ᵀ)/2        n×n 反对称
B = (X₂₁ − X₂₁ᵀ)/2        n×n 反对称
```

**只存 `(h, A, B, c₀)`**；规范矩阵

```text
X_c = ⎡ h   A ⎤ ,   ½ Ψ†X_cΨ = c†hc + ½c†Ac† + ½cBc − ½tr(h)
      ⎣ B  −hᵀ⎦
```

矩阵输入 `H = ½Ψ†XΨ + c_in` 时常数：`c₀ = c_in + ½tr(X₂₂) + ½tr(h)`。
参数计数：独立 `2n² − n` 个复参数。厄米特例 `h=h†, B=A†`；纯配对 `h=0, B=0`。

### 1.3 玻色

```text
½ Ψ†XΨ = a† h a + ½(a† A a† + a B a) + ½ tr(X₂₂)

h = (X₁₁ + X₂₂ᵀ)/2       n×n 一般复
A = (X₁₂ + X₁₂ᵀ)/2       n×n 复对称
B = (X₂₁ + X₂₁ᵀ)/2       n×n 复对称
```

**只存 `(h, A, B, c₀)`**；规范矩阵

```text
X_c = ⎡ h   A  ⎤ ,   ½ Ψ†X_cΨ = a†ha + ½a†Aa† + ½aBa + ½tr(h)
      ⎣ B  hᵀ ⎦
```

矩阵输入时常数：`c₀ = c_in + ½tr(X₂₂) − ½tr(h)`。
参数计数：独立 `2n² + n` 个复参数。厄米特例 `h=h†, B=A†`；纯配对 `h=0, B=0`。

### 1.4 稠密验证

`n=2` 费米、`n=1` 玻色随机复 `X`：验证 `H − H_c = const·I` 到 1e-16
（本计划落盘前已做）。

### 1.5 厄米特例：`Fermionic/BosonicQuadraticHermitianOperator`

一般算子的 `(h,A,B)` 三块独立；厄米特例满足

```text
费米：h = h†,   B = A†
玻色：h = h†,   B = A†（A 复对称）
```

所以只存 2 个独立 block `(h,A)`，`B=A†` 导出。规范矩阵：

```text
费米：X_c = [[h, A], [A†, −hᵀ]]      （厄米）
玻色：X_c = [[h, A], [A†,  hᵀ]]      （厄米）
```

`c₀` 必为实数。`exp` 重载（类型定义见 §3.1）：

```julia
Base.exp(op::FermionicQuadraticHermitianOperator) = FermionicQuadraticGibbsState(op)
Base.exp(op::BosonicQuadraticHermitianOperator)   = BosonicQuadraticGibbsState(op)
```

`FermionicQuadraticHermitianOperator(X)` 校验 `X` 厄米，提
`h=(X₁₁−X₂₂ᵀ)/2`、`A=antisym(X₁₂)`；玻色 `h=(X₁₁+X₂₂ᵀ)/2`、`A=sym(X₁₂)`。

---

## 2. `*QuadraticOperatorExp` 与 `GaussianState`

### 2.1 `exp` 重载

```julia
struct FermionicQuadraticOperatorExp
    op :: FermionicQuadraticOperator
end
struct BosonicQuadraticOperatorExp
    op :: BosonicQuadraticOperator
end

Base.exp(op::FermionicQuadraticOperator) = FermionicQuadraticOperatorExp(op)
Base.exp(op::BosonicQuadraticOperator)   = BosonicQuadraticOperatorExp(op)
```

表示 `ρ = e^H`（`c₀` 只给整体标量；`purify` 归一化后不影响熵）。

### 2.2 `GaussianState`：只存协方差

```julia
struct FermionicGaussianState
    Γ :: Matrix{Float64}      # 2N×2N 实反对称，Γ_ab = i⟨γ_aγ_b⟩
end
struct BosonicGaussianState
    Σ :: Matrix{Float64}      # 2N×2N 实对称，Σ_ab = ½⟨{R_a,R_b}⟩
end
```

- **只有协方差**；没有 bdg/M 字段。协方差用哪种基（Majorana/(x,p)）是**内部
  约定**，文档写清即可；用户不需要直接接触。
- v1 的协方差表示就是 Majorana/(x,p)；熵/偏迹公式直接复用（已验证）。
- 纯态：费米 `Γ²=−I`；玻色辛本征值全 `½`。混合态：`|ν|≤1` / `ν≥½`。
  用 `ispure(gs)` 查询，不另立类型。
- Gibbs 态就是"协方差满足物理谱的高斯态"；其纠缠 Hamiltonian 不存储，
  稠密对比在算子层做：`densitymatrix(op)`（见 §3.3）；不对 `GaussianState` 做稠密化。

### 2.3 熵与偏迹（实协方差子矩阵）

- **物理模式**：`1..N`；模式 `p` ↔ 协方差指标 `[2p−1, 2p]`。
- **子系统**：`modes` 是物理模式集合；子矩阵

  ```text
  费米：Γ_A = Γ[2modes−1, 2modes]      （2|A|×2|A|，实反对称）
  玻色：Σ_A = Σ[2modes−1, 2modes]      （2|A|×2|A|，实对称）
  ```

- **费米熵**：`ν_k = abs.(eigvals(iΓ_A))`（每个 ±ν 出现两次）；
  `S = ½ Σ_k h((1+ν_k)/2)`（即 `S = sum(h, νs)/2`，v1 约定）。
- **玻色熵**：`ν_k = abs.(eigvals(Ω_A Σ_A))`（每个 ±ν 出现两次）；
  `S = ½ Σ_k f(ν_k)`（v1 约定）。
- **偏迹**：`partialtr(gs, modes) = GaussianState(submatrix)`；
  `entropy(gs, modes) = entropy(partialtr(gs, modes))`。

**Z₂**：费米子系统约化是代数（模式）约化；连续块/单模下与朴素偏迹一致，
非连续块不能拿朴素偏迹当参照（引用 Friis–Lee–Bruschi / Amosov–Filippov）。

### 2.5 厄米 `exp(H)` → Gibbs 态

对厄米 `H`，`ρ = e^H/Z` 是 Gibbs 密度矩阵。`GaussianState(exp(H))`
**直接**返回 Gibbs 态（n 模式，混合），不是纯化：

```julia
GaussianState(op::FermionicQuadraticGibbsState) -> FermionicGaussianState
GaussianState(op::BosonicQuadraticGibbsState)   -> BosonicGaussianState
```

协方差（`X = bdg(op.op)`，`c₀` 归一化后无关）：

```text
费米：C = (I + e^{X})^{-1}        （Nambu；再 Γ = to_majorana(C)）
玻色：C = ½ coth(−X/2)            （Bogoliubov；再 Σ = to_xp(C)）
```

特征分解实现：

```text
X = UΛU†
费米：C = U diag(1/(1+e^{λ})) U†
玻色：C = U diag(½ coth(−λ/2)) U†
```

校验：单模 `H=−βω c†c` → `X=diag(−βω,βω)` → `C=diag(1−n,n)`，
`n=1/(1+e^{βω})`；玻色 `H=−βω(a†a+½)` → `C=½coth(βω/2) I`。

**`purify` vs `GaussianState`**：

- `purify(gs::AbstractQuadraticGibbsState)` = **TFD**：`√ρ = e^{H/2}` 的矢量化
  （2n 模式，纯态）；不是 `e^H` 的矢量化（后者的约化态是 `ρ²`）。
- `GaussianState(gs)` = Gibbs 态 `ρ`（n 模式，混合态）。
- 返回值是**纯态** `GaussianState`（`ispure(psi) == true`），不是混合 Gibbs 态；
  `GaussianState(gs)` 才是混合 Gibbs 态。
- `GaussianState(gs) == partialtr(purify(gs), 1:n)`（测试固化）。
- 实现：`purify(gs) = purify(asoperator(exp(op(gs)/2)))`（先把半指数转成
  一般 exp，再走 Choi 矢量化）。
- **两种语义别混**：
  - `purify(gs)`（TFD）的纠缠 = Gibbs **态**的纠缠熵，等于
    `entropy(GaussianState(gs))`；
  - Gibbs **密度矩阵**的算符空间 OSEE =
    `entropy(purify(asoperator(gs)), modes)`（完整 `e^H` 的 Choi 态），
    不是 `purify(gs)`。`asoperator(gs)` 把 Gibbs 态转成同一 H 的一般 exp。

**符号约定**：`H` 是指数；Hamiltonian `H_sys` 传 `H = −β H_sys`。

---

## 3. 公开 API

### 3.1 类型

```julia
abstract type AbstractQuadraticOperator end
abstract type AbstractFermionicQuadraticOperator <: AbstractQuadraticOperator end
abstract type AbstractBosonicQuadraticOperator <: AbstractQuadraticOperator end

abstract type AbstractQuadraticOperatorExp end
abstract type AbstractFermionicQuadraticOperatorExp <: AbstractQuadraticOperatorExp end
abstract type AbstractBosonicQuadraticOperatorExp <: AbstractQuadraticOperatorExp end

abstract type AbstractQuadraticGibbsState <: AbstractQuadraticOperatorExp end

# 一般（非厄米）算子
struct FermionicQuadraticOperator <: AbstractFermionicQuadraticOperator
    h  :: Matrix{ComplexF64}
    A  :: Matrix{ComplexF64}   # 反对称
    B  :: Matrix{ComplexF64}   # 反对称
    c0 :: ComplexF64
end
struct BosonicQuadraticOperator <: AbstractBosonicQuadraticOperator
    h  :: Matrix{ComplexF64}
    A  :: Matrix{ComplexF64}   # 复对称
    B  :: Matrix{ComplexF64}   # 复对称
    c0 :: ComplexF64
end

# 厄米算子（B = A†；c0 实）
struct FermionicQuadraticHermitianOperator <: AbstractFermionicQuadraticOperator
    h  :: Matrix{ComplexF64}   # 厄米
    A  :: Matrix{ComplexF64}   # 一般
    c0 :: Float64
end
struct BosonicQuadraticHermitianOperator <: AbstractBosonicQuadraticOperator
    h  :: Matrix{ComplexF64}   # 厄米
    A  :: Matrix{ComplexF64}   # 复对称
    c0 :: Float64
end

# 算子指数
struct FermionicQuadraticOperatorExp <: AbstractFermionicQuadraticOperatorExp
    op :: FermionicQuadraticOperator
end
struct BosonicQuadraticOperatorExp <: AbstractBosonicQuadraticOperatorExp
    op :: BosonicQuadraticOperator
end
struct FermionicQuadraticGibbsState <: AbstractQuadraticGibbsState
    op :: FermionicQuadraticHermitianOperator
end
struct BosonicQuadraticGibbsState <: AbstractQuadraticGibbsState
    op :: BosonicQuadraticHermitianOperator
end


# 态（只存协方差）
struct FermionicGaussianState; Γ :: Matrix{Float64}; end
struct BosonicGaussianState;   Σ :: Matrix{Float64}; end
```

### 3.2 构造

```julia
# ---- 算子（Bogoliubov 独立 block） ----
FermionicQuadraticOperator(h, A, B; c0=0)
BosonicQuadraticOperator(h, A, B; c0=0)
FermionicQuadraticOperator(X::AbstractMatrix; c0=0)   # 2n×2n，自动提 block
BosonicQuadraticOperator(X::AbstractMatrix; c0=0)

# ---- 算子指数 ----
exp(op::FermionicQuadraticOperator) -> FermionicQuadraticOperatorExp
exp(op::BosonicQuadraticOperator)   -> BosonicQuadraticOperatorExp

# ---- 厄米算子 ----
FermionicQuadraticHermitianOperator(h, A; c0=0)   # h 厄米；B=A†
FermionicQuadraticHermitianOperator(X::AbstractMatrix; c0=0)  # X 厄米
BosonicQuadraticHermitianOperator(h, A; c0=0)     # h 厄米；A 复对称；B=A†
BosonicQuadraticHermitianOperator(X::AbstractMatrix; c0=0)

# ---- 厄米算子指数 ----
exp(op::FermionicQuadraticHermitianOperator) -> FermionicQuadraticGibbsState
exp(op::BosonicQuadraticHermitianOperator)   -> BosonicQuadraticGibbsState

# ---- GaussianState：不开放从协方差/M 直接构造 ----
# 只由 purify / partialtr 产生；厄米 Gibbs 态用 GaussianState(exp(H_herm))
# 内部用 _gaussian_state(Γ)（测试可用）
```

> `GaussianState` 类型导出，但**不开放从协方差/M 直接构造**。合法入口：
> 纯态用 `purify(exp(H))`；混合态用 `partialtr(purify(exp(H)), modes)`；
> 厄米 Hamiltonian 的 Gibbs 态用 `GaussianState(exp(H_herm))`（§2.5）。
> 纯配对/BCS 的 GMPS 系数态也都是 `purify(exp(H))` 的输出。

**校验**（内部 `_gaussian_state(Γ)`）：`Γ` 实反对称、`Γ²=−I` 或 `|ν|≤1`；
`Σ` 实对称、辛谱 `ν≥½`。`purify`/`partialtr` 的输出自动满足；失败抛
`ArgumentError`。

### 3.3 核心函数

```julia
# 一般 exp → 算子矢量化（Choi/IF，纯）
purify(op::AbstractQuadraticOperatorExp) -> GaussianState

# Gibbs 态 → TFD（√ρ=e^{H/2} 的矢量化，纯）
purify(gs::AbstractQuadraticGibbsState) -> GaussianState
purify(gs::FermionicQuadraticGibbsState) -> FermionicGaussianState   # 纯（TFD）
purify(gs::BosonicQuadraticGibbsState)   -> BosonicGaussianState     # 纯（TFD）

# Gibbs 态 → 混合 Gibbs 态（协方差）
GaussianState(gs::AbstractQuadraticGibbsState) -> GaussianState

# Gibbs 态 → 同一 H 的一般 exp（用于算符空间 OSEE / Choi 态）
asoperator(gs::AbstractQuadraticGibbsState) -> AbstractQuadraticOperatorExp

# 偏迹（modes = 物理模式 1..N；内部展开到 [2modes−1,2modes]）
partialtr(gs::FermionicGaussianState, modes) -> FermionicGaussianState
partialtr(gs::BosonicGaussianState,   modes) -> BosonicGaussianState

# 熵
entropy(gs::FermionicGaussianState; α=1) -> Float64
entropy(gs::FermionicGaussianState, modes; α=1) -> Float64
entropy(gs::BosonicGaussianState; α=1) -> Float64
entropy(gs::BosonicGaussianState, modes; α=1) -> Float64

# 查询/访问
ispure(gs) -> Bool
covariance(gs::FermionicGaussianState) -> Matrix{Float64}   # Γ
covariance(gs::BosonicGaussianState)   -> Matrix{Float64}   # Σ
nmodes(gs) -> Int                       # size(Γ,1) ÷ 2
# Fock 空间矩阵（只作用于算子，对比测试用）
densitymatrix(op::FermionicQuadraticOperator) -> Matrix{ComplexF64}    # H = ½Ψ†XΨ + c0
densitymatrix(op::FermionicQuadraticOperatorExp) -> Matrix{ComplexF64} # e^H
densitymatrix(op::BosonicQuadraticOperator; nbas=20) -> Matrix{ComplexF64}
densitymatrix(op::BosonicQuadraticOperatorExp; nbas=20) -> Matrix{ComplexF64}

# 纯配对/BCS 系数态：都用 purify(exp(H))；无 pairing_state 公开接口
```

**没有 `osee`。** OSEE 的唯一写法：

```julia
psi = purify(exp(H))       # 只算一次
S   = entropy(psi, modes)
```

### 3.4 OSEE 的指标约定

- `purify(op)` 返回的纯态在 doubled 空间有 `2n` 个模（`c₁..c_n, d₁..d_n`）；
  `modes` 是**物理模式**（1..2n）。
- 物理站点 `p` 的割 = `modes = [p, n+p]`；`entropy` 内部展开成 4 个协方差指标
  `[2p−1,2p, 2(n+p)−1,2(n+p)]`。
- 包不引入 site 抽象；站点由用户/下游胶水给出。

### 3.5 示例

```julia
using GaussianStates

# 一般非厄米 IF 算子
H   = FermionicQuadraticOperator(h, A, B)
op  = exp(H)
psi = purify(op)                 # 纯态
@assert ispure(psi)
S1  = entropy(psi, [1])          # 物理模式 1 的熵
S12 = entropy(psi, [1, 2])
rho = partialtr(psi, [1])        # 子系统 GaussianState
@assert entropy(rho) ≈ S1

# 物理站点割（doubled 模 p 与 n+p）
Ssite2 = entropy(psi, [2, n+2])

# 纯配对 IF：purify 就是系数态（GMPS）的纯化，bond 熵 = OSEE
Hpair = FermionicQuadraticOperator(zeros(n,n), W, zeros(n,n))
Spair = entropy(purify(exp(Hpair)), [1, n+1])

# 纯配对/BCS 的 GMPS 系数态就是上面的 purify 输出；无需额外构造器

# 厄米 Hamiltonian：直接拿 Gibbs 态
Hh = FermionicQuadraticHermitianOperator(h, A)   # h 厄米；A 配对
gs = GaussianState(exp(Hh))                       # Gibbs 态（混合）
@assert !ispure(gs)
S_gibbs = entropy(gs)
psi_tfd = purify(exp(Hh))                         # TFD（纯）
@assert entropy(partialtr(psi_tfd, 1:n)) ≈ S_gibbs
S_osee = entropy(purify(asoperator(exp(Hh))), modes)  # 密度矩阵的算符空间 OSEE
```

---

## 4. 算法

### 4.1 `purify`（费米，一般非厄米）

目标：`|ρ⟩⟩` 的 2n 模纯态，实 Majorana 协方差 `Γ_d`（4n×4n）。

1. `W = e^{−X}`（`[H,Ψ]=−XΨ` 的伴随作用）。
2. EPR 镜像：`M(c_i)=d_i†`、`M(c_i†)=−d_i`（符号由稠密测试钉死）。
3. 湮灭矩阵 `V`（2n×4n）：行 `α(u)=u·Ψ_c − M(Wᵀu)`。
4. Bogoliubov 协方差 `C_d = I − V†(V V†)^{-1}V`（4n×4n 投影子）。
5. 内部变换到实 Majorana `Γ_d = to_majorana(C_d)`（4n×4n）；返回
   `FermionicGaussianState(Γ_d)`。
6. 测试：`ispure`、`Γ_d²=−I`、与稠密 `|ρ⟩⟩` 的 `Γ_d` 逐元素（1e-10）。

> `to_majorana` 是内部函数；用户只看到 `GaussianState` 的实协方差。

### 4.2 `purify`（玻色）

- v4.0 先覆盖：实对称正定（v1 Gibbs）、纯配对（实/复）；一般复对称 `K` 用
  复辛/Bloch–Messiah（或用 `to_xp` 后按 v1 Williamson），作为工作包。
- 可归一化判据：`|tanh r|<1`；失败抛 `ArgumentError`。

### 4.3 `partialtr` / `entropy`

- `partialtr(gs, modes)`：取 `Γ[2modes−1,2modes]`（费米）/ `Σ[...]`（玻色），
  返回对应的 `GaussianState`。
- `entropy(gs)`：整协方差；`entropy(gs, modes)`：子矩阵。
- 费米 `Σh((1+ν)/2)`；玻色 `Σf(ν)`；纯态整态熵 = 0。

### 4.4 复杂度

| 操作 | 复杂度 |
|---|---|
| 算子构造/提 block | O(n²) |
| `purify`（一般费米） | O(n³)（`W=e^{−X}`、`V`、`C_d`、`to_majorana`） |
| `partialtr` | O(1)（子矩阵拷贝） |
| `entropy` | O(|modes|³) |

`purify` 不缓存；调用一次、复用返回的 `GaussianState`（这就是不提供 `osee`
的原因）。

---

## 5. 约定与边界

1. **归一化**：`purify` 归一化纯态；`c₀` 只给 `ρ` 乘标量，不影响协方差/熵。
2. **纯态端点**：`Γ` 的谱端点（ν=1 费米、ν=½ 玻色）合法；`entropy` 按公式。
3. **Z₂**：费米子系统的代数（模式）约化；连续/单模一致，非连续不拿朴素偏迹
   当参照。
4. **错误**：尺寸/对称性/纯度/物理谱/越界/重复 `modes` → `ArgumentError`/
   `BoundsError`；不静默投影。
5. **物理标签**：包内不出现 `(branch,time,band)`；下游胶水把标签映射成 `modes`。
6. **pairing**：GMPS 系数态 = `purify(e^H)`，bond 熵 = OSEE；不提供
   `pairing_state`，纯配对/BCS 都由 `purify` 覆盖。

---

## 6. 测试计划

### 6.1 固化 `purify` 输出是纯态

```julia
@test ispure(purify(op))
@test maximum(abs, psi.Γ^2 + I) < 1e-10          # 费米
@test maximum(abs, eigvals(Ω*psi.Σ) .^ 2 .- 0.25) < 1e-10  # 玻色
```

### 6.2 稠密仲裁

- 费米 m=1,2,3：`Hmat = densitymatrix(op)`（`2^m`）、
  `ρ = densitymatrix(exp(op)) = e^{Hmat}`、`|ρ⟩⟩=(ρ⊗I)|I⟩`（`2^{2m}`），
  逐元素 `Γ_d`；`entropy(purify(op), modes)` = 稠密 graded 子矩阵熵。
- 玻色 n=1,2：`densitymatrix(op; nbas=...)` 的 Fock 截断，同样对拍。
- 特殊极限：厄米 `M`（v1 Gibbs）、纯配对**算子**（本轮的最小例：
  0.691416 / 0.223718）；与 v1 pairing **态**（0.500402）显式区分。
- `partialtr`/`entropy(gs,modes)` = `entropy(partialtr(gs,modes))`。
- 纯配对：`purify(exp(H_pair))` 的协方差与 v1 纯配对分支逐元素一致。
- GMPS 对比：一般 Gaussian 系数态的 bond 熵 = `purify` 的 OSEE；BCS 用
  一般 `purify`，不依赖 `pairing_state`。
- 厄米 Gibbs：`GaussianState(exp(Hh))` 的协方差 vs 稠密 `e^H/Z`；
  与 `partialtr(purify(exp(Hh)), 1:n)` 一致；单模解析
  `n=1/(1+e^{βω})`、`½coth(βω/2)`。

### 6.3 IF：GMPS = 纯化态（回归）

- 纯配对：`purify(exp(H_pair))` 的 bond 熵 = v1 纯配对分支的熵。
- BCS-like：一般 Gaussian 系数态（number-conserving + pairing），
  其 bond 熵 = 对应 normal-ordered IF 算子的 `purify` OSEE。
- 交叉检查：系数态 `exp(Σ Δ c̄†c'†)|0⟩` 与 `purify` 的输出协方差逐元素一致
  （1e-10），`entropy` 相同。
- 上一轮"差 0.4–0.5"是普通指数 `exp(Σ Δ d a)` 的错误对比，已废弃。
- 厄米：`purify(gs)`（TFD）= `purify(asoperator(exp(op(gs)/2)))`；
  `partialtr(purify(gs), 1:n) == GaussianState(gs)`；
  算符空间 OSEE 用 `purify(asoperator(gs))`（Choi），不是 TFD。
- Gibbs 熵 vs v1 `FermionicGaussianState(M)`。

### 6.4 性质

- 整纯态熵 0；`S(∅)=0`；纯态互补 `S(A)=S(Ā)`。
- `c₀` 无关；`α→1` 连续；`α=2` 与稠密一致。
- 错误路径：非方阵、奇数维、非厄米 block、非物理协方差、越界/重复 `modes`。

### 6.5 性能

- 费米 n=16,32,64：`purify` 一次 + 全 bond profile。
- 玻色 n=8,16。

---

## 7. 文件布局与迁移

### 7.1 文件

```text
src/
├── GaussianStates.jl
├── quadratic_fermion.jl     # FermionicQuadraticOperator + block
├── quadratic_boson.jl       # BosonicQuadraticOperator + block
├── quadratic_exp.jl         # *Exp + exp 重载
├── quadratic_hermitian.jl   # *QuadraticHermitianOperator + exp
├── gibbs.jl                 # GaussianState(exp(H_hermitian))
├── gaussian_state.jl        # *GaussianState（协方差 only）+ ispure/covariance
├── purify_fermion.jl        # W=e^{−X}、V、C_d、to_majorana（内部）
├── purify_boson.jl          # 复辛/特例
├── partialtr.jl             # 物理模式 → 协方差子矩阵
├── entropy.jl               # 费米/玻色熵、Rényi
├── fock.jl                  # densitymatrix(op)（Fock 空间矩阵，对比测试）
# 无 pairing_legacy.jl：纯配对/BCS 都由 purify 覆盖
└── deprecated.jl            # v1 兼容层（旧构造器 shim）
```

### 7.2 v1 → v7 迁移

| v1 | v7 |
|---|---|
| `FermionicGaussianState(M)`（厄米） | `GaussianState(exp(FermionicQuadraticHermitianOperator(...)))`，`H = −½Ψ†MΨ` |
| `FermionicGaussianState(W)`（反对称） | `purify(exp(FermionicQuadraticOperator(0,W,0)))` |
| `BosonicGaussianState(K)`（实对称 PD） | `GaussianState(exp(BosonicQuadraticHermitianOperator(...)))`，`H = −½RᵀKR` |
| `BosonicGaussianState(W)`（复对称/非 PD） | `purify(exp(BosonicQuadraticOperator(...)))` |
| `osee(s, modes)` | `entropy(purify(exp(H)), modes)` 或 `entropy(ps, modes)` |
| `entropy(s, sites)` | `entropy(gs, modes)` |
| `covariance(s)` | `covariance(gs)`（Γ 或 Σ） |
| — | `Fermionic/BosonicQuadraticOperator`、`exp`、`purify` |

兼容层：`FermionicGaussianState(M; kind=:auto)` 按对称性分派到 `:gibbs`；
`osee` 不保留（文档给 `purify`+`entropy` 的迁移示例）。

### 7.3 下游胶水（不在本包）

QuAPI/GTEMPO 负责把 `(branch,time,band)`、链位置、bond 映射成 `modes`；包只提供
`purify`/`partialtr`/`entropy`。

---

## 8. 里程碑与开放问题

### M0 类型/block/协方差（1–2 天）
- `Fermionic/BosonicQuadraticOperator`、`*Exp`、`exp` 重载；
  `*GaussianState`（协方差 only）、`ispure`、`partialtr`、`entropy`。
- v1 兼容层；v1 测试保持绿。

### M1 费米一般 `purify`（核心，3–5 天）
- `W=e^{−X}`、镜像、`V`、`C_d`、`to_majorana`；m=1,2,3 稠密仲裁；
  `ispure` 固化；与 v1 极限对齐。

### M2 熵/偏迹/OSEE 工作流（1 天）
- `partialtr`、`entropy(gs,modes)`、纯态端点、零模。

### M3 玻色（2–4 天）
- 实对称 PD/纯配对（v1）；一般复对称 `K` 的 `to_xp`/复辛；可归一化判据；
  Fock 仲裁。

### M4 文档/迁移/回归（2–3 天）
- README/manual 重写（明确 GMPS 系数态 = `purify(e^H)`，bond 熵 = OSEE；
  `GaussianState` 只由 `purify`/`partialtr` 产生）；
  v1 迁移示例；性能基准。

### 开放问题
1. **EPR 镜像符号**（`d_i†`/`−d_i`、模式序、Z₂）：m=1,2 稠密钉死。
2. **`c₀` 的约定**：`densitymatrix(op)` 要把 `c₀` 计入 Fock 矩阵；熵无关。
3. **玻色一般复对称 `K`**：复辛 Bloch–Messiah 稳定性；`|tanh r|=1` 端点。
4. **`GaussianState` 的构造**：默认不开放；只由 `purify`/`partialtr` 产生。
   若下游 GMPS 只给协方差、没有对应的 H，是否需要暴露一个专家接口
   `from_covariance(Γ)`？（当前计划：只保留内部 `_gaussian_state`。）
5. **玻色 `densitymatrix` 的 Fock 截断**：`nbas` 的默认值/误差估计。
6. **厄米 `H` 的符号/常数**：`H` 是指数；`c₀` 实；`X` 的特征分解/矩阵函数
   在 `λ→±∞` 的截断。
7. **`GaussianState(exp(Hh))` 与 `partialtr(purify(exp(Hh)), 1:n)` 的一致性**：
   测试固化；两者应逐元素相等。

---

## 9. 一句话总结

**`Fermionic/BosonicQuadraticOperator` 存 Bogoliubov 的 3 个独立 block；
`exp(op)` 给 `*QuadraticOperatorExp`；`purify` 把它变成只含实协方差的
`GaussianState`（纯态）；`partialtr` 取物理模式子矩阵得到子系统 Gibbs 态；
`entropy(gs[,modes])` 算熵。OSEE = `entropy(purify(exp(H)), modes)`，不提供
`osee`，`purify` 只算一次。GMPS 的系数态就是 `e^H` 的纯化，其 bond 熵就是
OSEE；`GaussianState` 不开放从协方差/M 直接构造，只由 `purify`/`partialtr`
或 `GaussianState(exp(H_herm))`（厄米 Gibbs 态）产生。**
