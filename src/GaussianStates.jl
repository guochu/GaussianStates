# ============================================================================
# GaussianStates.jl —— v7 API
#
# QuadraticOperator / QuadraticOperatorExp / QuadraticGibbsState / GaussianState
#   H = 二次型（Bogoliubov block）→ exp(H) → purify / GaussianState
#   OSEE = entropy(purify(exp(H)), modes)
# ============================================================================

"""
费米/玻色二次型算子与高斯态工具包（v7 API）。

主工作流：`H = FermionicQuadraticOperator(h, A, B)`（玻色同理）→
`psi = purify(exp(H))`（纯态，O(n³)）→ `entropy(psi, modes)` /
`partialtr(psi, modes)`。厄米算子 `exp(H)` 给 Gibbs 态：
`GaussianState(gs)` 直接取混合 Gibbs 协方差，`purify(gs)` 给 TFD。

OSEE 的唯一写法是 `entropy(purify(exp(H)), modes)`（`purify` 只算一次）；
态只存实协方差（Majorana / (x,p)），不开放直接构造。表象约定：算子
block（`bdg`）在 Nambu 基 `Ψ=(c,c†)`/`(a,a†)`，态协方差在 Majorana
（费米）/ (x,p) 四极（玻色）基——两套表象不同，不能直接混用
（基矢对照表见手册"表象"一节）。
"""
module GaussianStates

using LinearAlgebra

export AbstractQuadraticOperator, AbstractFermionicQuadraticOperator,
       AbstractBosonicQuadraticOperator, AbstractQuadraticOperatorExp,
       AbstractQuadraticGibbsState,
       FermionicQuadraticOperator, FermionicQuadraticHermitianOperator,
       BosonicQuadraticOperator, BosonicQuadraticHermitianOperator,
       FermionicQuadraticOperatorExp, BosonicQuadraticOperatorExp,
       FermionicQuadraticGibbsState, BosonicQuadraticGibbsState,
       FermionicGaussianState, BosonicGaussianState,
       purify, GaussianState, partialtr, entropy, ispure, purity, covariance,
       nmodes, fockmatrix, bdg, quadraticoperator

include("abstract.jl")
include("quadratic_fermion.jl")
include("quadratic_boson.jl")
include("gaussian_state.jl")
include("transforms.jl")
include("purify_fermion.jl")
include("purify_boson.jl")
include("partialtr.jl")
include("entropy.jl")
include("fockmatrix.jl")
include("inverse.jl")

end # module
