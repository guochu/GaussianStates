# API 参考

## 抽象类型

```@docs
AbstractQuadraticOperator
AbstractQuadraticOperatorExp
GaussianStates.AbstractGaussianState
```

## 算子（二次型 block）

算子类型参数化 `T<:Number`（block 元素类型，构造时自动提升）；
厄米算子只存 `(h, A)`，`.B` 经 `getproperty` 合成为 `A†`，与一般算子
接口对齐（可直接 `FermionicQuadraticOperator(g.h, g.A, g.B)`）。

block 与 `bdg` 在 **Nambu 表象** `Ψ=(c,c†)` / `(a,a†)`；态协方差在
Majorana（费米）/ (x,p) 四极（玻色）表象——两套表象的基矢与顺序
都不同，不能直接混用（矩阵×基矢对照表见[理论与算法](@ref)）。

```@docs
FermionicQuadraticOperator
BosonicQuadraticOperator
FermionicQuadraticHermitianOperator
BosonicQuadraticHermitianOperator
bdg
nmodes
```

## 算子指数与 Gibbs 态

```@docs
FermionicQuadraticOperatorExp
BosonicQuadraticOperatorExp
FermionicQuadraticGibbsState
BosonicQuadraticGibbsState
```

`exp(::FermionicQuadraticOperator)` 等重载把算子包成上表类型；
厄米算子的 `exp` 直接给 Gibbs 态。

## 高斯态与 purify

```@docs
FermionicGaussianState
BosonicGaussianState
purify
GaussianState
ispure
purity
covariance
```

态类型不参数化（协方差恒为实 `Float64`）；协方差在 Majorana（费米）/
(x,p)（玻色）表象，与算子 block 的 Nambu 表象不同（对照表见
[理论与算法](@ref)）。构造器带物理性校验
（反对称/对称、正定、谱约束），非法协方差抛 `ArgumentError`。
日常入口仍是 `purify` / `partialtr` / `GaussianState(exp(H_herm))`。

## 反推（协方差 → bdg）

```@docs
quadraticoperator
```

## 偏迹与熵

```@docs
partialtr
entropy
```

OSEE 的唯一写法：`psi = purify(exp(H))` 算一次，然后
`entropy(psi, modes)`；站点 `p` 的割是 `modes = [p, n+p]`。
没有 `osee` 函数。

## 稠密构造（调试用）

```@docs
fockmatrix
```
