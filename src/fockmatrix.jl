# ============================================================================
# fockmatrix.jl —— 把 QuadraticOperator / *Exp 转成 Fock 空间矩阵
# 仅用于对比测试；不对 GaussianState 定义。
# ============================================================================

# JW：|n₁…n_m⟩，模式 i ↔ bit (i-1)
function _jw_create!(out, v, r::Int)
    fill!(out, zero(eltype(out)))
    bit = 1 << (r - 1); mask = bit - 1
    @inbounds for x in 0:(length(v) - 1)
        (x & bit) != 0 && continue
        sg = iseven(count_ones(x & mask)) ? 1.0 : -1.0
        out[(x | bit) + 1] += sg * v[x + 1]
    end
    return out
end

function _jw_annihilate!(out, v, r::Int)
    fill!(out, zero(eltype(out)))
    bit = 1 << (r - 1); mask = bit - 1
    @inbounds for x in 0:(length(v) - 1)
        (x & bit) == 0 && continue
        sg = iseven(count_ones(x & mask)) ? 1.0 : -1.0
        out[(x ⊻ bit) + 1] += sg * v[x + 1]
    end
    return out
end

function _jw_annihilator(m::Int, i::Int)
    dim = 1 << m
    A = zeros(ComplexF64, dim, dim)
    e = zeros(ComplexF64, dim); out = zeros(ComplexF64, dim)
    for k in 1:dim
        fill!(e, 0); e[k] = 1
        _jw_annihilate!(out, e, i)
        A[:, k] .= out
    end
    return A
end

function _fock_operators(op::FermionicQuadraticOperator; kwargs...)
    n = nmodes(op)
    c = [_jw_annihilator(n, i) for i in 1:n]
    cd = [adjoint(x) for x in c]
    h, A, B = op.h, op.A, op.B
    dim = 1 << n
    H = ComplexF64(op.c0) .* Matrix{ComplexF64}(I, dim, dim)
    @inbounds for i in 1:n, j in 1:n
        H .+= h[i, j] .* (cd[i] * c[j])
    end
    @inbounds for i in 1:n, j in 1:n
        H .+= (A[i, j] / 2) .* (cd[i] * cd[j])
        H .+= (B[i, j] / 2) .* (c[i] * c[j])
    end
    return H
end
_fock_operators(op::FermionicQuadraticHermitianOperator; kwargs...) =
    _fock_operators(FermionicQuadraticOperator(op.h, op.A, adjoint(op.A); c0=op.c0))

"""
    fockmatrix(op; d=20)

把二次型算子 / `e^H` / Gibbs 态转成 Fock 空间稠密矩阵（费米 `2^n` 维；
玻色 `d` 截断）。仅用于调试与对照测试；熵/偏迹请走协方差路径。
"""
fockmatrix(op::FermionicQuadraticOperator; kwargs...) = _fock_operators(op)
fockmatrix(op::FermionicQuadraticHermitianOperator; kwargs...) = _fock_operators(op)
fockmatrix(op::FermionicQuadraticOperatorExp; kwargs...) = exp(_fock_operators(op.op))
fockmatrix(gs::FermionicQuadraticGibbsState; kwargs...) = exp(_fock_operators(gs.op))

# ---- 玻色（Fock 截断） ----
function _boson_annihilator(n::Int, d::Int)
    dim = d + 1
    a = zeros(ComplexF64, dim, dim)
    for k in 1:d
        a[k, k+1] = sqrt(k)      # a|k⟩ = √k|k-1⟩（0 基）；行 k、列 k+1
    end
    full = Vector{Matrix{ComplexF64}}(undef, n)
    for j in 1:n
        M = ComplexF64[1;;]
        for k in 1:n
            M = kron(M, k == j ? a : Matrix{ComplexF64}(I, dim, dim))
        end
        full[j] = M
    end
    return full
end

function _fock_operators(op::BosonicQuadraticOperator; d::Int=20)
    n = nmodes(op)
    a = _boson_annihilator(n, d)
    ad = [adjoint(x) for x in a]
    h, A, B = op.h, op.A, op.B
    dim = (d + 1)^n
    H = ComplexF64(op.c0) .* Matrix{ComplexF64}(I, dim, dim)
    @inbounds for i in 1:n, j in 1:n
        H .+= h[i, j] .* (ad[i] * a[j])
        H .+= (A[i, j] / 2) .* (ad[i] * ad[j])
        H .+= (B[i, j] / 2) .* (a[i] * a[j])
    end
    return H
end
_fock_operators(op::BosonicQuadraticHermitianOperator; kwargs...) =
    _fock_operators(BosonicQuadraticOperator(op.h, op.A, adjoint(op.A); c0=op.c0); kwargs...)

fockmatrix(op::BosonicQuadraticOperator; d::Int=20) = _fock_operators(op; d=d)
fockmatrix(op::BosonicQuadraticHermitianOperator; d::Int=20) = _fock_operators(op; d=d)
fockmatrix(op::BosonicQuadraticOperatorExp; d::Int=20) = exp(_fock_operators(op.op; d=d))
fockmatrix(gs::BosonicQuadraticGibbsState; d::Int=20) = exp(_fock_operators(gs.op; d=d))

# ---- 稠密矢量化（费米，测试仲裁用）----
# |I⟩_G = exp(Σ cᵢ†dᵢ†)|0⟩（ket 模 1..n 在低位、bra 模 n+1..2n 在高位），
# ψ = (ρ⊗I)|I⟩_G 归一化。
function _dense_vectorize(rho::AbstractMatrix, n::Int)
    dim2 = 1 << (2n)
    IG = zeros(ComplexF64, dim2); IG[1] = 1
    term = copy(IG); buf = zeros(ComplexF64, dim2)
    t = zeros(ComplexF64, dim2); t2 = zeros(ComplexF64, dim2)
    for k in 1:n
        fill!(buf, 0)
        for i in 1:n
            fill!(t, 0); _jw_create!(t, term, n + i)
            fill!(t2, 0); _jw_create!(t2, t, i)
            buf .+= t2
        end
        buf ./= k; IG .+= buf; copyto!(term, buf)
    end
    psi = zeros(ComplexF64, dim2)
    for y in 0:(1 << n)-1, x in 0:(1 << n)-1, xp in 0:(1 << n)-1
        psi[xp + y * (1 << n) + 1] += rho[xp + 1, x + 1] * IG[x + y * (1 << n) + 1]
    end
    return psi ./ norm(psi)
end

function _nambu_ops(n::Int)
    c = [_jw_annihilator(2n, i) for i in 1:2n]
    cd = [adjoint(x) for x in c]
    # (c, c†, d, d†)
    return vcat(c[1:n], cd[1:n], c[n+1:2n], cd[n+1:2n])
end
