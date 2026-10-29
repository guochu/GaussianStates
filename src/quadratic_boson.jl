# ============================================================================
# quadratic_boson.jl —— 玻色二次型算子
#
# H = a† h a + ½(a† A a† + a B a) + c0        （Bogoliubov / Nambu 基）
# 规范 2n×2n 矩阵 X = [[h, A], [B, hᵀ]]；A,B 复对称
# 厄米：h=h†, B=A†（只存 h、A，.B 由 getproperty 合成）
# ============================================================================

# 玻色 Fock 代数（a†ᵢa†ⱼ = a†ⱼa†ᵢ）中配对矩阵的反对称部分自动抵消：
# 任意输入投影到对称部分，物理上严格等价。
symmetrize(M::AbstractMatrix) = (M + transpose(M)) / 2

"""
    BosonicQuadraticOperator(h, A, B; c0=0)
    BosonicQuadraticOperator(X; c0=0)

一般（非厄米）玻色二次型算子 `H = a†ha + ½(a†Aa† + aBa) + c₀`
（`A`、`B` 任意输入，构造时自动对称化——反对称部分在 Fock 代数中
自动抵消）。矩阵输入时常数 `c₀ = c_in + ½tr(X₂₂) − ½tr(h)`。
`exp(op)` 给 [`BosonicQuadraticOperatorExp`](@ref)。类型参数 `T<:Number}`
是 block 的元素类型（构造时自动提升）。
"""
struct BosonicQuadraticOperator{T<:Number} <: AbstractBosonicQuadraticOperator{T}
    h  :: Matrix{T}
    A  :: Matrix{T}
    B  :: Matrix{T}
    c0 :: T
end

function BosonicQuadraticOperator(h, A, B; c0=0)
    T = promote_type(eltype(h), eltype(A), eltype(B), typeof(c0))
    h = convert(Matrix{T}, h); A = convert(Matrix{T}, A); B = convert(Matrix{T}, B)
    n = size(h, 1); size(h, 2) == n || throw(ArgumentError("h 必须是方阵"))
    size(A) == (n, n) || throw(ArgumentError("A 尺寸应为 ($n,$n)"))
    size(B) == (n, n) || throw(ArgumentError("B 尺寸应为 ($n,$n)"))
    A = symmetrize(A); B = symmetrize(B)
    return BosonicQuadraticOperator{T}(h, A, B, convert(T, c0))
end

function BosonicQuadraticOperator(X::AbstractMatrix; c0=0)
    m, n2 = size(X); m == n2 || throw(ArgumentError("X 必须是方阵"))
    iseven(m) || throw(ArgumentError("X 的维数必须是偶数"))
    n = m ÷ 2
    X11 = X[1:n, 1:n]; X12 = X[1:n, n+1:2n]
    X21 = X[n+1:2n, 1:n]; X22 = X[n+1:2n, n+1:2n]
    h = (X11 + transpose(X22)) / 2
    A = symmetrize(X12); B = symmetrize(X21)
    c0b = c0 + tr(X22) / 2 - tr(h) / 2
    return BosonicQuadraticOperator(h, A, B; c0=c0b)
end

"""规范 2n×2n Bogoliubov 矩阵 X = [[h,A],[B,hᵀ]]；c₀ 见 `bdg`（费米版）的说明。"""
function bdg(op::BosonicQuadraticOperator)
    n = size(op.h, 1)
    return [op.h op.A; op.B transpose(op.h)]
end

Base.exp(op::BosonicQuadraticOperator) = BosonicQuadraticOperatorExp(op)
Base.:/(op::BosonicQuadraticOperator, s::Number) =
    BosonicQuadraticOperator(op.h / s, op.A / s, op.B / s; c0=op.c0 / s)

"""
    BosonicQuadraticOperatorExp(op)

`e^H`（一般非厄米 `H`）；`purify` 给 Choi 矢量化纯态。可归一化性由
`purify` 的物理性检查拒绝（如 `h` 有增长方向）。
"""
struct BosonicQuadraticOperatorExp{T<:Number} <: AbstractBosonicQuadraticOperatorExp{T}
    op :: BosonicQuadraticOperator{T}
end

"""
    BosonicQuadraticHermitianOperator(h, A; c0=0)
    BosonicQuadraticHermitianOperator(X; c0=0)

厄米玻色二次型（`h` 厄米、`A` 复对称、`B = A†` 由 `.B` 合成，不存储）。
`exp(op)` 给 Gibbs 态 [`BosonicQuadraticGibbsState`](@ref)。
**注意指数约定**：热态应传 `H = −β·H_sys`（`h ≈ −βω < 0`）；`e^H`
无界/失稳时抛 `ArgumentError`。
"""
struct BosonicQuadraticHermitianOperator{T<:Number} <: AbstractBosonicQuadraticOperator{T}
    h  :: Matrix{T}
    A  :: Matrix{T}
    c0 :: Real
end

# .B = A† 对齐一般算子接口
function Base.getproperty(op::BosonicQuadraticHermitianOperator, s::Symbol)
    if s === :B
        return adjoint(getfield(op, :A))
    end
    return getfield(op, s)
end
Base.propertynames(::BosonicQuadraticHermitianOperator) = (:h, :A, :B, :c0)

function BosonicQuadraticHermitianOperator(h, A; c0=0)
    T = promote_type(eltype(h), eltype(A))
    h = convert(Matrix{T}, h); A = convert(Matrix{T}, A)
    n = size(h, 1); size(h, 2) == n || throw(ArgumentError("h 必须是方阵"))
    size(A) == (n, n) || throw(ArgumentError("A 尺寸应为 ($n,$n)"))
    _check_herm(h, "h")
    isreal(c0) || throw(ArgumentError("厄米算子的 c0 必须是实数"))
    A = symmetrize(A)                   # 只有对称部分有效
    return BosonicQuadraticHermitianOperator{T}(h, A, real(c0))
end

function BosonicQuadraticHermitianOperator(X::AbstractMatrix; c0=0)
    m, n2 = size(X); m == n2 || throw(ArgumentError("X 必须是方阵"))
    iseven(m) || throw(ArgumentError("X 的维数必须是偶数"))
    _check_herm(X, "X")
    n = m ÷ 2
    X11 = X[1:n, 1:n]; X12 = X[1:n, n+1:2n]
    X21 = X[n+1:2n, 1:n]; X22 = X[n+1:2n, n+1:2n]
    h = (X11 + transpose(X22)) / 2
    A = symmetrize(X12)
    c0b = real(c0 + tr(X22) / 2 - tr(h) / 2)
    return BosonicQuadraticHermitianOperator(h, A; c0=c0b)
end

function bdg(op::BosonicQuadraticHermitianOperator)
    n = size(op.h, 1)
    return [op.h op.A; op.B transpose(op.h)]
end

Base.exp(op::BosonicQuadraticHermitianOperator) = BosonicQuadraticGibbsState(op)
Base.:/(op::BosonicQuadraticHermitianOperator, s::Number) =
    BosonicQuadraticHermitianOperator(op.h / s, op.A / s; c0=op.c0 / s)

"""
    BosonicQuadraticGibbsState(op)

厄米玻色 `e^H`：归一化后是 Gibbs 密度矩阵（要求 H_sys 有下界）。
`GaussianState(gs)` 给混合 Gibbs 态，`purify(gs)` 给 TFD。
"""
struct BosonicQuadraticGibbsState{T<:Number} <: AbstractQuadraticGibbsState{T}
    op :: BosonicQuadraticHermitianOperator{T}
end
Base.:/(gs::BosonicQuadraticGibbsState, s::Number) = BosonicQuadraticGibbsState(gs.op / s)
