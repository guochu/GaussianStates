# GaussianStates.jl

```@docs
GaussianStates
```

费米/玻色二次型算子与高斯态的工具包，v7 API：

- **算子**：`Fermionic/BosonicQuadraticOperator(h, A, B)` 存 Bogoliubov 的
  3 个独立 block（一般非厄米）；厄米变体 `*QuadraticHermitianOperator(h, A)`。
- **指数**：`exp(H)` 给 `*QuadraticOperatorExp`；厄米 `exp(H)` 给
  `*QuadraticGibbsState`（归一化后是 Gibbs 密度矩阵）。
- **态**：`purify` 把 `e^H` 变成纯态（Choi/IF 矢量化），`GaussianState(gs)`
  把厄米 `e^H` 变成混合 Gibbs 态；态只存实协方差，`entropy`/`partialtr`
  全部走协方差快速算法（每割 O(模式数³)），不构造稠密矩阵。
- **OSEE**：`entropy(purify(exp(H)), modes)`——`purify` 只算一次、复用返回值。

## 安装与加载

本包位于 `Missile/GaussianStates`（本地包，暂未注册）。把仓库路径加入
`LOAD_PATH` 或在环境中 `Pkg.develop` 即可：

```julia
using GaussianStates
```

## 快速上手

### 一般非厄米算子（IF / OSEE 主工作流）

```julia
n = 4
h = randn(ComplexF64, n, n)
A = randn(ComplexF64, n, n); A = (A - transpose(A)) / 2   # 费米 A 反对称
B = randn(ComplexF64, n, n); B = (B - transpose(B)) / 2

op  = exp(FermionicQuadraticOperator(h, A, B))   # e^H
psi = purify(op)                                 # 纯态（2n 模），O(n³)
@assert ispure(psi)

S1  = entropy(psi, [1])              # 物理模式 1 的熵
rho = partialtr(psi, [1])            # 子系统 GaussianState（一般混合）
@assert entropy(rho) ≈ S1

# 物理站点 p 的割（ket 模 p 与 bra 模 n+p）
Ssite2 = entropy(psi, [2, n + 2])

# 纯配对 IF：GMPS 系数态 = purify 的输出，bond 熵 = OSEE
Hpair = FermionicQuadraticOperator(zeros(n,n), W, zeros(n,n))   # W 反对称
Spair = entropy(purify(exp(Hpair)), [1, n + 1])
```

### 厄米 Hamiltonian（Gibbs 态 / TFD）

```julia
Hh = FermionicQuadraticHermitianOperator(h_herm, A)   # h 厄米；A 配对
gs  = exp(Hh)                       # FermionicQuadraticGibbsState

rho = GaussianState(gs)             # Gibbs 态（n 模，混合）
@assert !ispure(rho)
S_gibbs = entropy(rho)

psi_tfd = purify(gs)                # TFD（2n 模，纯 = √ρ 的矢量化）
@assert entropy(partialtr(psi_tfd, 1:n)) ≈ S_gibbs

# Gibbs 密度矩阵的算符空间 OSEE（完整 e^H 的 Choi，与 TFD 是两个量）
# 厄米算子的 .B 自动合成为 A†，可直接构造一般 exp
op_full = exp(FermionicQuadraticOperator(gs.op.h, gs.op.A, gs.op.B))
S_osee = entropy(purify(op_full), [1, n + 1])
```

### 玻色

```julia
# 指数约定：热态传 H = -β·H_sys（h ≈ -βω < 0）；失稳/无界会抛 ArgumentError
Hb  = BosonicQuadraticHermitianOperator(-β .* h_sys, A)   # A 复对称
gb  = exp(Hb)
rhob = GaussianState(gb)            # Gibbs 态（(x,p) 协方差）
psib = purify(gb)                   # TFD（纯）
Sb   = entropy(psib, 1:n)           # = entropy(rhob)

# 一般非厄米玻色 e^H 的 Choi（任意配对）
opb  = exp(BosonicQuadraticOperator(h, A, B))             # A、B 复对称
psib = purify(opb)
```

## 文档结构

- [理论与算法](@ref)：block 约定与 `c₀`、协方差表示、`purify` 的湮灭子
  构造（费米/玻色）、TFD、熵与偏迹公式、数值验证摘要。
- [API 参考](@ref)：全部导出接口。

## 相关生态

`purify(exp(H_pair))` 的 bond 熵就是费米子影响泛函（IF）系数态的 OSEE，
与 GTEMPO 的 GrassmannMPS bond 熵同一约定（`GTEMPO/osee/osee_report.md`）。
三包 API 设计（GaussianStates ← QuAPI ← GTEMPO）见
`GTEMPO/osee/quapi_osee_api_plan.md`；`(branch,time,band)` 物理标签到
`modes` 的映射由下游胶水负责，本包只认 `1..N` 的模式指标。
