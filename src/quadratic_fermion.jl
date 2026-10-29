# ============================================================================
# quadratic_fermion.jl —— 费米二次型算子
#
# H = c† h c + ½(c† A c† + c B c) + c0        （Bogoliubov / Nambu 基）
# 规范 2n×2n 矩阵 X = [[h, A], [B, -hᵀ]]
# 厄米：h=h†, B=A†（只存 h、A，.B 由 getproperty 合成）
# ============================================================================

# 费米 Fock 代数（c†ᵢc†ⱼ = −c†ⱼc†ᵢ）中配对矩阵的对称部分自动抵消，
# 对角元恒为零：任意输入投影到反对称部分，物理上严格等价。
antisymmetrize(M::AbstractMatrix) = (M - transpose(M)) / 2

function _check_herm(M::AbstractMatrix, name::AbstractString)
    n = size(M, 1)
    size(M, 2) == n || throw(ArgumentError("$name 必须是方阵"))
    isapprox(M, adjoint(M); atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("$name 必须厄米"))
    return nothing
end

# ---------------------------------------------------------------------------
# 一般（非厄米）算子
# ---------------------------------------------------------------------------
"""
    FermionicQuadraticOperator(h, A, B; c0=0)
    FermionicQuadraticOperator(X; c0=0)

一般（非厄米）费米二次型算子 `H = c†hc + ½(c†Ac† + cBc) + c₀`
（`A`、`B` 任意输入，构造时自动反对称化——对称部分在 Fock 代数中
自动抵消，对角元恒为零）。2n×2n 矩阵 `X` 输入时自动提取独立块与常数：
`c₀ = c_in + ½tr(X₂₂) + ½tr(h)`。`exp(op)` 给
[`FermionicQuadraticOperatorExp`](@ref)。类型参数 `T<:Number` 是 block
的元素类型（构造时自动提升）。
"""
struct FermionicQuadraticOperator{T<:Number} <: AbstractFermionicQuadraticOperator{T}
    h  :: Matrix{T}
    A  :: Matrix{T}
    B  :: Matrix{T}
    c0 :: T
end

function FermionicQuadraticOperator(h, A, B; c0=0)
    T = promote_type(eltype(h), eltype(A), eltype(B), typeof(c0))
    h = convert(Matrix{T}, h); A = convert(Matrix{T}, A); B = convert(Matrix{T}, B)
    n = size(h, 1); size(h, 2) == n || throw(ArgumentError("h 必须是方阵"))
    size(A) == (n, n) || throw(ArgumentError("A 尺寸应为 ($n,$n)"))
    size(B) == (n, n) || throw(ArgumentError("B 尺寸应为 ($n,$n)"))
    A = antisymmetrize(A); B = antisymmetrize(B)
    return FermionicQuadraticOperator{T}(h, A, B, convert(T, c0))
end

# 2n×2n 矩阵输入：自动提 block
function FermionicQuadraticOperator(X::AbstractMatrix; c0=0)
    m, n2 = size(X); m == n2 || throw(ArgumentError("X 必须是方阵"))
    iseven(m) || throw(ArgumentError("X 的维数必须是偶数"))
    n = m ÷ 2
    X11 = X[1:n, 1:n]; X12 = X[1:n, n+1:2n]
    X21 = X[n+1:2n, 1:n]; X22 = X[n+1:2n, n+1:2n]
    h = (X11 - transpose(X22)) / 2
    A = antisymmetrize(X12); B = antisymmetrize(X21)
    c0b = c0 + tr(X22) / 2 + tr(h) / 2
    return FermionicQuadraticOperator(h, A, B; c0=c0b)
end

Base.exp(op::FermionicQuadraticOperator) = FermionicQuadraticOperatorExp(op)
Base.:/(op::FermionicQuadraticOperator, s::Number) =
    FermionicQuadraticOperator(op.h / s, op.A / s, op.B / s; c0=op.c0 / s)

"""
    bdg(op) -> Matrix

规范 2n×2n Bogoliubov 矩阵：费米 `X = [[h,A],[B,−hᵀ]]`，玻色
`X = [[h,A],[B,hᵀ]]`。

**不含 `c₀`（有意为之，无 bug）**：`c₀` 是单位算符项，`e^{c₀}` 只给
`e^H` 乘一个标量，归一化后对协方差、`purify`、熵全部无影响；只有
Fock 空间矩阵 `fockmatrix` 需要并已经计入 `c₀`。
"""
function bdg(op::FermionicQuadraticOperator)
    n = size(op.h, 1)
    return [op.h op.A; op.B -transpose(op.h)]
end

"""
    FermionicQuadraticOperatorExp(op)

`e^H`（一般非厄米 `H`）。不是密度矩阵；`purify` 把它变成纯态
（Choi/IF 矢量化），OSEE = `entropy(purify(exp(H)), modes)`。
"""
struct FermionicQuadraticOperatorExp{T<:Number} <: AbstractFermionicQuadraticOperatorExp{T}
    op :: FermionicQuadraticOperator{T}
end

# ---------------------------------------------------------------------------
# 厄米算子
# ---------------------------------------------------------------------------
"""
    FermionicQuadraticHermitianOperator(h, A; c0=0)
    FermionicQuadraticHermitianOperator(X; c0=0)

厄米费米二次型（`h` 厄米；`B = A†` 由 `.B` 合成，不存储）。
`exp(op)` 给 Gibbs 态 [`FermionicQuadraticGibbsState`](@ref)。
"""
struct FermionicQuadraticHermitianOperator{T<:Number} <: AbstractFermionicQuadraticOperator{T}
    h  :: Matrix{T}
    A  :: Matrix{T}
    c0 :: Real
end

# .B = A† 对齐一般算子接口
function Base.getproperty(op::FermionicQuadraticHermitianOperator, s::Symbol)
    if s === :B
        return adjoint(getfield(op, :A))
    end
    return getfield(op, s)
end
Base.propertynames(::FermionicQuadraticHermitianOperator) = (:h, :A, :B, :c0)

function FermionicQuadraticHermitianOperator(h, A; c0=0)
    T = promote_type(eltype(h), eltype(A))
    h = convert(Matrix{T}, h); A = convert(Matrix{T}, A)
    n = size(h, 1); size(h, 2) == n || throw(ArgumentError("h 必须是方阵"))
    size(A) == (n, n) || throw(ArgumentError("A 尺寸应为 ($n,$n)"))
    _check_herm(h, "h")
    isreal(c0) || throw(ArgumentError("厄米算子的 c0 必须是实数"))
    A = antisymmetrize(A)               # 只有反对称部分有效
    return FermionicQuadraticHermitianOperator{T}(h, A, real(c0))
end

function FermionicQuadraticHermitianOperator(X::AbstractMatrix; c0=0)
    m, n2 = size(X); m == n2 || throw(ArgumentError("X 必须是方阵"))
    iseven(m) || throw(ArgumentError("X 的维数必须是偶数"))
    _check_herm(X, "X")
    n = m ÷ 2
    X11 = X[1:n, 1:n]; X12 = X[1:n, n+1:2n]
    X21 = X[n+1:2n, 1:n]; X22 = X[n+1:2n, n+1:2n]
    h = (X11 - transpose(X22)) / 2
    A = antisymmetrize(X12)
    c0b = real(c0 + tr(X22) / 2 + tr(h) / 2)
    return FermionicQuadraticHermitianOperator(h, A; c0=c0b)
end

function bdg(op::FermionicQuadraticHermitianOperator)
    n = size(op.h, 1)
    return [op.h op.A; op.B -transpose(op.h)]
end

Base.exp(op::FermionicQuadraticHermitianOperator) = FermionicQuadraticGibbsState(op)
Base.:/(op::FermionicQuadraticHermitianOperator, s::Number) =
    FermionicQuadraticHermitianOperator(op.h / s, op.A / s; c0=op.c0 / s)

# ---------------------------------------------------------------------------
# Gibbs 态
# ---------------------------------------------------------------------------
"""
    FermionicQuadraticGibbsState(op)

厄米 `e^H`：归一化后是 Gibbs 密度矩阵。`GaussianState(gs)` 给混合
Gibbs 态（n 模），`purify(gs)` 给 TFD（2n 模，纯）。
"""
struct FermionicQuadraticGibbsState{T<:Number} <: AbstractQuadraticGibbsState{T}
    op :: FermionicQuadraticHermitianOperator{T}
end
Base.:/(gs::FermionicQuadraticGibbsState, s::Number) = FermionicQuadraticGibbsState(gs.op / s)
