# GaussianStates.jl

```@docs
GaussianStates
```

高斯态纯数学工具包：费米/玻色高斯态的表示、子系统熵（`entropy`）、
operator-space 纠缠熵（`osee`）与稠密密度矩阵（`densitymatrix`）。
熵的计算全部走协方差矩阵快速算法（每割 O(模式数³)），不构造稠密矩阵；
`densitymatrix` 仅用于调试与对照。

## 安装与加载

本包位于 `Missile/GaussianStates`（本地包，暂未注册）。把仓库路径加入
`LOAD_PATH` 或在环境中 `Pkg.develop` 即可：

```julia
using GaussianStates
```

## 快速上手

```julia
using GaussianStates

# ---- 费米 Gibbs 态：厄米 BdG 矩阵（m 个物理模） ----
h = randn(2, 2); h = (h + h') ./ 2          # 厄米
g = 0.5 .* randn(2, 2)                       # 任意；配对块 A = g − gᵀ
M = fermionic_bdg_matrix(h, g)
s = FermionicGaussianState(M)

entropy(s)                                   # 整态 von Neumann 熵
entropy(s, [1])                              # 偏迹子系统熵（物理模式指标）
osee(s, [1, 3])                              # OSEE（doubled 模式指标：site 1 = [1, m+1]）
ρ = densitymatrix(s)                         # 稠密 ρ（调试用，2^m × 2^m）

# ---- 费米纯配对态：复反对称矩阵（doubled 空间 2n 个模，IF 系数态） ----
W = randn(ComplexF64, 4, 4); W = W - transpose(W)
sp = FermionicGaussianState(W)
osee(sp, [1, 3])                             # site p 的割 = 模式 [p, n+p]

# ---- 玻色高斯态：实对称正定矩阵 ----
hb = randn(2, 2); hb = hb * hb' + I
B = bosonic_bdg_matrix(hb, 0.1 .* randn(2, 2))
sb = BosonicGaussianState(B)
osee(sb, [1, 3])                             # site p 的割 = 模式 [p, n+p]
```

## 文档结构

- [理论与算法](@ref)：entropy / osee 的定义、协方差快速算法的完整推导
  （含 doubled 空间构造与复数情形的共轭约定）。
- [API 参考](@ref)：全部导出接口。

## 相关生态

`osee` 的纯配对分支就是费米子影响泛函（IF）系数态的 OSEE，与
GTEMPO 的 GrassmannMPS bond 熵同一约定（见 `GTEMPO/osee/osee_report.md`）。
三包 API 设计（GaussianStates ← QuAPI ← GTEMPO）见
`GTEMPO/osee/quapi_osee_api_plan.md`。
