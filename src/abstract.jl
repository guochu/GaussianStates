# ============================================================================
# abstract.jl —— 抽象类型与访问器
#
# 算子类型参数化 {T<:Number}（block 的元素类型）；GaussianState 只存
# 实协方差（Float64），不参数化。
# ============================================================================

"二次型算子抽象基类型（费米/玻色 × 一般/厄米），参数 T 为 block 元素类型。"
abstract type AbstractQuadraticOperator{T<:Number} end
abstract type AbstractFermionicQuadraticOperator{T<:Number} <: AbstractQuadraticOperator{T} end
abstract type AbstractBosonicQuadraticOperator{T<:Number}   <: AbstractQuadraticOperator{T} end

"算子指数 `e^H` 的抽象基类型（参数 T 同算子）。"
abstract type AbstractQuadraticOperatorExp{T<:Number} end
abstract type AbstractFermionicQuadraticOperatorExp{T<:Number} <: AbstractQuadraticOperatorExp{T} end
abstract type AbstractBosonicQuadraticOperatorExp{T<:Number}   <: AbstractQuadraticOperatorExp{T} end

# 厄米 e^H 是 Gibbs 密度矩阵；具体类型是 Fermionic/BosonicQuadraticGibbsState
abstract type AbstractQuadraticGibbsState{T<:Number} <: AbstractQuadraticOperatorExp{T} end

"只存实协方差的高斯态抽象基类型（元素恒为 Float64，不参数化）。"
abstract type AbstractGaussianState end
abstract type AbstractFermionicGaussianState <: AbstractGaussianState end
abstract type AbstractBosonicGaussianState   <: AbstractGaussianState end

"模式数（算子按 `h` 的维数；态按协方差维数的一半）。"
nmodes(op::AbstractQuadraticOperator) = size(op.h, 1)
nmodes(s::AbstractGaussianState) = size(covariance(s), 1) ÷ 2
