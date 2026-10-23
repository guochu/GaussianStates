# ============================================================================
# test/helpers.jl
#
# 稠密态调试工具：**不经过**协方差/高斯公式，直接从 h、g 构造
# 玻色/费米密度矩阵，再作偏迹取子系统熵，用来交叉验证 GaussianStates 的
# 快速算法（benchmark / debug）。
#
# 费米：W = [h g; −gᵀ −hᵀ]（也可直接给完整反对称矩阵），
#       态 |Ψ⟩ = exp(½c†Wc†)|0⟩，ρ = |Ψ⟩⟨Ψ|（JW 稠密表示）。
# 玻色：bdg 为 (x₁,p₁,x₂,p₂,…) 上的实对称正定二次型，
#       ρ = exp(−½Rᵀ·bdg·R)/Z（Fock 基截断到每模 nbas 个激发）。
#
# 指标约定与主包一致：
#   费米模式 p ↔ Majorana {2p−1, 2p}；玻色模式 p ↔ 正则坐标 {2p−1, 2p}。
# ============================================================================

using LinearAlgebra
using Random

# --------------------------------------------------------------------------- #
# 随机块
# --------------------------------------------------------------------------- #

"""随机复反对称 n×n 矩阵"""
function rand_antisym(rng, n::Int; scale::Real=1.0)
    A = scale .* randn(rng, ComplexF64, n, n)
    return A .- transpose(A)
end

"""随机实对称 n×n 矩阵"""
function rand_sym(rng, n::Int; scale::Real=1.0)
    A = scale .* randn(rng, n, n)
    return (A + transpose(A)) ./ 2
end

"""随机实对称正定 n×n 矩阵"""
function rand_spd(rng, n::Int; shift::Real=1.0)
    A = randn(rng, n, n)
    return A * transpose(A) + shift * Matrix{Float64}(I, n, n)
end

# --------------------------------------------------------------------------- #
# 通用：偏迹与 von Neumann 熵
# --------------------------------------------------------------------------- #

"""偏迹：保留 `keep` 指定的模式（模式 1 为最低位，与列主序一致）"""
function partial_trace(ρ::AbstractMatrix, keep::Vector{Int}, dims::Vector{Int})
    n = length(dims)
    keep = sort(collect(Int, keep))
    drop = setdiff(1:n, keep)
    dK = prod(dims[keep])
    dD = prod(dims[drop])
    dK == 1 && return reshape([tr(ρ)], 1, 1)

    ck_a = zeros(Int, length(keep)); ck_b = zeros(Int, length(keep))
    cd = zeros(Int, length(drop))
    ca = zeros(Int, n); cb = zeros(Int, n)
    ρA = zeros(eltype(ρ), dK, dK)
    for a in 0:(dK-1)
        _digits!(ck_a, a, dims[keep]); ca[keep] = ck_a
        for b in 0:(dK-1)
            _digits!(ck_b, b, dims[keep]); cb[keep] = ck_b
            s = zero(eltype(ρ))
            for d in 0:(dD-1)
                _digits!(cd, d, dims[drop])
                ca[drop] = cd; cb[drop] = cd
                s += ρ[_index_from_digits(ca, dims)+1, _index_from_digits(cb, dims)+1]
            end
            ρA[a+1, b+1] = s
        end
    end
    return ρA
end

function _digits!(out::Vector{Int}, idx::Int, ds::Vector{Int})
    for m in 1:length(ds)
        out[m] = idx % ds[m]
        idx ÷= ds[m]
    end
    return out
end

function _index_from_digits(c::Vector{Int}, ds::Vector{Int})
    idx = 0
    stride = 1
    for m in 1:length(ds)
        idx += c[m] * stride
        stride *= ds[m]
    end
    return idx
end

"""von Neumann 熵 −Σ p ln p"""
function vn_entropy(ρ::AbstractMatrix; atol::Real=1e-14)
    p = eigvals(Hermitian((ρ + ρ') ./ 2))
    return -sum(x * log(x) for x in p if x > atol)
end

# --------------------------------------------------------------------------- #
# 费米：JW 稠密表示
# --------------------------------------------------------------------------- #

function jw_create!(out, v, r::Int)
    fill!(out, zero(eltype(out)))
    bit = 1 << (r - 1)
    mask = bit - 1
    @inbounds for x in 0:(length(v)-1)
        (x & bit) != 0 && continue
        sg = iseven(count_ones(x & mask)) ? 1.0 : -1.0
        out[(x | bit)+1] += sg * v[x+1]
    end
    return out
end

function jw_annihilate!(out, v, r::Int)
    fill!(out, zero(eltype(out)))
    bit = 1 << (r - 1)
    mask = bit - 1
    @inbounds for x in 0:(length(v)-1)
        (x & bit) == 0 && continue
        sg = iseven(count_ones(x & mask)) ? 1.0 : -1.0
        out[(x ⊻ bit)+1] += sg * v[x+1]
    end
    return out
end

"""稠密构造 |Ψ⟩ = exp(½ c†Wc†)|0⟩（B 幂零，Taylor 级数精确截断）"""
function fermionic_state_vector(W::AbstractMatrix{<:Number})
    M = size(W, 1)                            # 模数
    dim = 1 << M
    T = ComplexF64
    ψ = zeros(T, dim)
    ψ[1] = 1
    Bv = function (out, v)
        fill!(out, zero(T))
        t1 = zeros(T, dim)
        t2 = zeros(T, dim)
        for b in 1:M, a in 1:M
            W[a, b] == 0 && continue
            jw_create!(t1, v, b)
            jw_create!(t2, t1, a)
            @. out += W[a, b] / 2 * t2
        end
        return out
    end
    term = copy(ψ)
    buf = zeros(T, dim)
    for k in 1:M
        Bv(buf, term)
        @. buf /= k
        @. ψ += buf
        copyto!(term, buf)
    end
    return ψ
end

"""稠密密度矩阵 ρ = |Ψ⟩⟨Ψ| / ⟨Ψ|Ψ⟩"""
function fermionic_density_matrix(W::AbstractMatrix{<:Number})
    ψ = fermionic_state_vector(W)
    ψ ./= norm(ψ)
    return ψ * ψ'
end

"""稠密 Majorana 协方差 Γ_ab = i⟨γ_aγ_b⟩"""
function fermionic_covariance(W::AbstractMatrix{<:Number})
    ψ = fermionic_state_vector(W)
    nm = size(W, 1)
    Mγ = 2nm
    dim = length(ψ)
    nrm = dot(ψ, ψ)
    γψ = Vector{Vector{ComplexF64}}(undef, Mγ)
    t1 = zeros(ComplexF64, dim)
    t2 = zeros(ComplexF64, dim)
    for a in 1:Mγ
        m = (a + 1) ÷ 2
        if isodd(a)
            jw_annihilate!(t1, ψ, m)
            jw_create!(t2, ψ, m)
            γψ[a] = t1 .+ t2
        else
            jw_create!(t1, ψ, m)
            jw_annihilate!(t2, ψ, m)
            γψ[a] = 1im .* (t1 .- t2)
        end
    end
    Mmat = zeros(ComplexF64, Mγ, Mγ)
    for a in 1:Mγ, b in 1:Mγ
        Mmat[a, b] = dot(γψ[a], γψ[b]) / nrm
    end
    Γ = 1im .* Mmat
    return real.((Γ .- transpose(Γ)) ./ 2)
end

"""稠密 OSEE：对模集合 `modes`（1-based）作偏迹后取熵

!!! note
    这里的偏迹在 JW（占据数）基下直接进行。它与费米子约化态的熵一致**仅当
    `modes` 是模序中的连续块**（或单个模）——这正是 OSEE 里链割的情形。
    非连续子集需额外计入 JW 串（此时快速协方差算法仍然正确，但本函数不能
    用作对照）。
"""
function dense_osee_fermionic(W::AbstractMatrix{<:Number}, modes::Vector{Int})
    N = size(W, 1)
    ρ = fermionic_density_matrix(W)
    return vn_entropy(partial_trace(ρ, modes, fill(2, N)))
end

# --------------------------------------------------------------------------- #
# 费米：Gibbs 态 ρ ∝ exp(−½ ψ†·M·ψ)，ψ = (c₁..c_m, c†₁..c†_m)
# --------------------------------------------------------------------------- #

"""JW 稠密湮灭算符矩阵（模式 i，模数 m）"""
function jw_annihilator_matrix(m::Int, i::Int)
    dim = 1 << m
    A = zeros(ComplexF64, dim, dim)
    e = zeros(ComplexF64, dim)
    out = zeros(ComplexF64, dim)
    for k in 1:dim
        fill!(e, 0); e[k] = 1
        jw_annihilate!(out, e, i)
        A[:, k] .= out
    end
    return A
end

"""Gibbs 态稠密密度矩阵 ρ = exp(−½ ψ†·M·ψ)/Z"""
function fermionic_gibbs_density_matrix(M::AbstractMatrix{<:Number})
    N = size(M, 1)
    m = N ÷ 2
    dim = 1 << m
    c = [jw_annihilator_matrix(m, i) for i in 1:m]
    cd = [adjoint(x) for x in c]
    ψ = vcat(c, cd)
    ψd = vcat(cd, c)
    H = zeros(ComplexF64, dim, dim)
    for a in 1:N, b in 1:N
        M[a, b] == 0 && continue
        H .+= (M[a, b] / 2) .* (ψd[a] * ψ[b])
    end
    H = (H + H') ./ 2
    E = eigen(Hermitian(H))
    w = exp.(-E.values); w ./= sum(w)
    return E.vectors * Diagonal(w) * E.vectors'
end

"""由密度矩阵算 Majorana 协方差 Γ_ab = i·tr(ρ γ_aγ_b)"""
function covariance_from_density_matrix(ρ::AbstractMatrix, m::Int)
    c = [jw_annihilator_matrix(m, i) for i in 1:m]
    cd = [adjoint(x) for x in c]
    γ = Vector{Matrix{ComplexF64}}(undef, 2m)
    for i in 1:m
        γ[2i-1] = c[i] .+ cd[i]
        γ[2i]   = 1im .* (cd[i] .- c[i])
    end
    Γ = zeros(Float64, 2m, 2m)
    for a in 1:2m, b in 1:2m
        Γ[a, b] = real(1im * tr(ρ * γ[a] * γ[b]))
    end
    return (Γ .- transpose(Γ)) ./ 2
end

"""Gibbs 态的**子系统熵**（partial trace 后取 von Neumann 熵；`modes` 为物理模）"""
function dense_entropy_fermionic_gibbs(M::AbstractMatrix{<:Number}, modes::Vector{Int})
    m = size(M, 1) ÷ 2
    ρ = fermionic_gibbs_density_matrix(M)
    return vn_entropy(partial_trace(ρ, modes, fill(2, m)))
end

# --------------------------------------------------------------------------- #
# OSEE：把密度矩阵当作双重空间中的纯态 |ρ⟩⟩，按 site 合并 (ket_p, bra_p)，
#       在 modes 与补集之间做 SVD，由奇异值求纠缠熵。
# --------------------------------------------------------------------------- #

"""
    operator_space_state(ρ; localdim=2) -> ψ, m

把 ρ（localdim^m × localdim^m）向量化为双重空间纯态 |ρ⟩⟩，指标顺序为
`(ket₁, bra₁, ket₂, bra₂, …)`，即每个 site 的 bra/ket 合并为一个 localdim² 维
局域空间。返回展平的态向量（已归一化）与模数 m。
"""
function operator_space_state(ρ::AbstractMatrix; localdim::Int=2)
    D = size(ρ, 1)
    m = round(Int, log(D) / log(localdim))
    T = reshape(ρ, ntuple(_ -> localdim, 2m)...) # (ket₁..ket_m, bra₁..bra_m)
    perm = Int[]
    for p in 1:m
        push!(perm, p, m + p)                    # (ket_p, bra_p) 相邻
    end
    T = permutedims(T, perm)
    ψ = reshape(T, localdim^(2m))
    ψ = ψ ./ norm(ψ)
    return ψ, m
end

"""任意 localdim 的 operator-space 熵（|ρ⟩⟩ 按 site 二分做 SVD）"""
function operator_space_entropy(ρ::AbstractMatrix, modes::Vector{Int};
                                α::Real=1, localdim::Int=2)
    ψ, m = operator_space_state(ρ; localdim=localdim)
    modes = sort(collect(Int, modes))
    rest = setdiff(1:m, modes)
    T = reshape(ψ, ntuple(_ -> localdim^2, m)...)
    T = permutedims(T, vcat(modes, rest))
    dA = (localdim^2)^length(modes)
    dB = (localdim^2)^length(rest)
    Mat = reshape(T, dA, dB)
    sv = svdvals(Mat)
    p = abs2.(sv)
    p ./= sum(p)
    if isapprox(α, 1; atol=1e-12)
        return -sum(x * log(x) for x in p if x > 1e-15)
    end
    return log(sum(x^α for x in p if x > 0)) / (1 - α)
end

"""
    gauss_vectorized_rho(ρ, m) -> Vector{ComplexF64}

费米 Gibbs 态的高斯向量化 |ρ⟩⟩ = (ρ⊗I)|I⟩_G：
|I⟩_G = exp(Σᵢ c_i† d_i†)|0⟩（ Grassmann/IF 形式一致；与 ρ 矩阵元的 plain
reshape 相差 (−1)^{|s|(|s|−1)/2} 的非局域符号——OSEE 的正确定义用本版本）。
模式布局：A 拷贝模式 i ↔ bit (i−1)（低位），B 拷贝模式 i ↔ bit (m+i−1)。
"""
function gauss_vectorized_rho(ρ::AbstractMatrix, m::Int)
    dim2 = 1 << (2m)
    IG = zeros(ComplexF64, dim2); IG[1] = 1
    term = copy(IG)
    buf = zeros(ComplexF64, dim2)
    t = zeros(ComplexF64, dim2)
    t2 = zeros(ComplexF64, dim2)
    for k in 1:m
        fill!(buf, 0)
        for i in 1:m
            fill!(t, 0)
            jw_create!(t, term, m + i)     # d_i†（jw_create! 会清零 out）
            fill!(t2, 0)
            jw_create!(t2, t, i)           # c_i†（c† 在左）
            buf .+= t2
        end
        buf ./= k
        IG .+= buf
        copyto!(term, buf)
    end
    ψ = zeros(ComplexF64, dim2)
    for y in 0:(1<<m)-1, x in 0:(1<<m)-1, xp in 0:(1<<m)-1
        ψ[xp + y * (1 << m) + 1] += ρ[xp + 1, x + 1] * IG[x + y * (1 << m) + 1]
    end
    ψ ./= norm(ψ)
    return ψ
end

"""doubled 态矢量按 (keep 模式 | 其余) 的比特重组 SVD 熵（模式 r ↔ bit r−1）"""
function svd_entropy_modes(ψ::AbstractVector, m::Int, keep::Vector{Int}; α::Real=1)
    n = length(keep)
    dim = length(ψ)
    drop = setdiff(1:m, keep)
    Mmat = zeros(ComplexF64, 1 << n, dim >> n)
    for v in 0:dim-1
        ki = 0; di = 0
        for (pos, r) in enumerate(keep)
            ki |= ((v >> (r - 1)) & 1) << (pos - 1)
        end
        for (pos, r) in enumerate(drop)
            di |= ((v >> (r - 1)) & 1) << (pos - 1)
        end
        Mmat[ki+1, di+1] = ψ[v+1]
    end
    sv = svdvals(Mmat)
    p = abs2.(sv); p ./= sum(p)
    if isapprox(α, 1; atol=1e-12)
        return -sum(x * log(x) for x in p if x > 1e-15)
    end
    return log(sum(x^α for x in p if x > 0)) / (1 - α)
end

"""
    fermionic_reorder(ψ, m, order) -> Vector

把 doubled 矢量（模式 r ↔ bit r−1）的模式串重排为 `order ++ 其余`（新串从低位
到高位），带费米置换符号（占据模的重排宇称）。重排后对前缀模式的 plain SVD
= 费米（Z₂ 一致）约化的 Schmidt 分解。
"""
function fermionic_reorder(ψ::AbstractVector, m::Int, order::Vector{Int})
    dim = length(ψ)
    neworder = vcat(order, setdiff(1:2m, order))
    posof = Dict(r => i for (i, r) in enumerate(neworder))
    out = zeros(ComplexF64, dim)
    for v in 0:dim-1
        occ = [(v >> (r - 1)) & 1 for r in 1:2m]
        occupied = [r for r in 1:2m if occ[r] == 1]
        # 符号：占据模从旧串（升序）重排到新串位置的置换宇称（逆序数）
        sg = 1
        for i in 1:length(occupied)-1, j in i+1:length(occupied)
            if posof[occupied[i]] > posof[occupied[j]]
                sg = -sg
            end
        end
        newidx = 0
        for (pos, r) in enumerate(neworder)
            newidx |= occ[r] << (pos - 1)
        end
        out[newidx+1] = sg * ψ[v+1]
    end
    return out
end

"""
    dense_osee_fermionic_gibbs(M, modes; α=1)

Gibbs 态的 OSEE：高斯向量化 |ρ⟩⟩ = (ρ⊗I)|I⟩_G 把 `modes` 的 ket/bra 拷贝重排为
连续前缀后做 SVD（费米 Z₂ 一致的约化）。与包内快速算法同一定义。
"""
function dense_osee_fermionic_gibbs(M::AbstractMatrix{<:Number},
                                    modes::Vector{Int}; α::Real=1)
    m = size(M, 1) ÷ 2
    ρ = fermionic_gibbs_density_matrix(M)
    ψ = gauss_vectorized_rho(ρ, m)
    order = Int[]
    for p in modes
        push!(order, p, m + p)
    end
    ψp = fermionic_reorder(ψ, m, order)
    n = length(order)
    return svd_entropy_modes(ψp, 2m, collect(1:n); α=α)
end

"""模指标 → Majorana 指标集合（供 osee 使用）"""
function majorana_indices(modes::AbstractVector{<:Integer})
    out = Int[]
    for p in modes
        push!(out, 2p - 1, 2p)
    end
    return sort(out)
end

# --------------------------------------------------------------------------- #
# 玻色：Fock 基截断稠密表示
# --------------------------------------------------------------------------- #

"""单模湮灭算符（基 index k+1 ↔ |k⟩，截断 k ≤ nbas）"""
function single_mode_a(nbas::Int)
    a = zeros(nbas + 1, nbas + 1)
    for k in 1:nbas
        a[k, k+1] = sqrt(k)
    end
    return a
end

"""把单模算符嵌入多模空间（模式 1 为最低位）"""
function embed_operator(op::AbstractMatrix, m::Int, nm::Int, d::Int)
    mats = [i == m ? op : Matrix{ComplexF64}(I, d, d) for i in 1:nm]
    return reduce(kron, reverse(mats))
end

"""
    bosonic_pairing_vector(W; nbas=6) -> Vector{ComplexF64}

玻色纯配对向量 |Ψ⟩ = exp(½ c†Wc†)|0⟩（W 复对称，最大奇异值 < 1）的 Fock 截断
稠密实现：每模截断到 `nbas` 个激发，对截断生成元做矩阵指数。
"""
function bosonic_pairing_vector(W::AbstractMatrix{<:Number}; nbas::Int=6)
    N = size(W, 1)
    d = nbas + 1
    dim = d^N
    ad = ComplexF64.(transpose(single_mode_a(nbas)))
    Gd = zeros(ComplexF64, dim, dim)
    for i in 1:N, j in 1:N
        iszero(W[i, j]) && continue
        Gd .+= (W[i, j] / 2) .* (embed_operator(ad, i, N, d) *
                                 embed_operator(ad, j, N, d))
    end
    e0 = zeros(ComplexF64, dim); e0[1] = 1
    ψ = exp(Gd) * e0
    return ψ ./ norm(ψ)
end

"""截断向量的协方差 σ_ab = Re⟨ψ|R_aR_b|ψ⟩（R = (x₁,p₁,…)，模式 1 最低位）"""
function covariance_from_bosonic_vector(ψ::AbstractVector{<:Number},
                                        nmodes::Int, d::Int)
    a = single_mode_a(d - 1)
    ad = transpose(a)
    xm = ComplexF64.((a .+ ad) ./ sqrt(2))
    pm = ComplexF64.(-1im .* (a .- ad) ./ sqrt(2))
    R = Vector{Matrix{ComplexF64}}(undef, 2nmodes)
    for m in 1:nmodes
        R[2m-1] = embed_operator(xm, m, nmodes, d)
        R[2m]   = embed_operator(pm, m, nmodes, d)
    end
    σ = zeros(2nmodes, 2nmodes)
    for i in 1:2nmodes, j in i:2nmodes
        σ[i, j] = σ[j, i] = real(dot(ψ, R[i] * (R[j] * ψ)))
    end
    return σ
end

"""截断向量的割熵（SVD）：sites 为模式指标（1..nmodes），模式 1 最低位"""
function dense_osee_bosonic_vector(ψ::AbstractVector{<:Number}, nmodes::Int,
                                   d::Int, sites::Vector{Int})
    T = reshape(ψ, ntuple(_ -> d, nmodes)...)
    perm = vcat(sort(sites), sort(setdiff(1:nmodes, sites)))
    M = reshape(permutedims(T, perm), d^length(sites), d^(nmodes - length(sites)))
    sv = svdvals(M)
    p = sv .^ 2; p ./= sum(p)
    p = p[p .> 1e-16]
    return -sum(p .* log.(p))
end

"""
    bosonic_density_matrix(bdg; nbas=16) -> (ρ, dims)

由 BdG 二次型构造 ρ = exp(−½Rᵀ·bdg·R)/Z，R = (x₁,p₁,x₂,p₂,…)，
Fock 基每模截断到 `nbas` 个激发。
"""
function bosonic_density_matrix(bdg::AbstractMatrix; nbas::Int=16)
    nm = size(bdg, 1) ÷ 2
    d = nbas + 1
    a = single_mode_a(nbas)
    ad = transpose(a)
    xm = ComplexF64.((a .+ ad) ./ sqrt(2))
    pm = ComplexF64.(-1im .* (a .- ad) ./ sqrt(2))

    R = Vector{Matrix{ComplexF64}}(undef, 2nm)
    for m in 1:nm
        R[2m-1] = embed_operator(xm, m, nm, d)
        R[2m]   = embed_operator(pm, m, nm, d)
    end

    dim = d^nm
    H = zeros(ComplexF64, dim, dim)
    for i in 1:2nm, j in 1:2nm
        bdg[i, j] == 0 && continue
        H .+= (bdg[i, j] / 2) .* (R[i] * R[j])
    end
    H = (H + H') ./ 2
    E = eigen(Hermitian(H))
    w = exp.(-E.values)
    w ./= sum(w)
    ρ = E.vectors * Diagonal(w) * E.vectors'
    return ρ, fill(d, nm)
end

"""稠密玻色**子系统熵**（偏迹后取熵；对照 `entropy`，勿与 OSEE 混淆）"""
function dense_entropy_bosonic(bdg::AbstractMatrix, modes::Vector{Int}; nbas::Int=16)
    ρ, dims = bosonic_density_matrix(bdg; nbas=nbas)
    return vn_entropy(partial_trace(ρ, modes, dims))
end

"""稠密玻色 **OSEE**：|ρ⟩⟩ 按 site 合并 (ket_p,bra_p) 后二分做 SVD（对照 `osee`）"""
function dense_osee_bosonic(bdg::AbstractMatrix, modes::Vector{Int}; nbas::Int=16)
    ρ, _ = bosonic_density_matrix(bdg; nbas=nbas)
    return operator_space_entropy(ρ, modes; localdim=nbas + 1)
end

# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #

"""对快速算法与稠密算法计时并返回 (S_fast, S_dense, t_fast, t_dense)"""
function benchmark_fermionic(osee_fast, W::AbstractMatrix, modes::Vector{Int})
    t1 = @elapsed Sf = osee_fast(modes)
    t2 = @elapsed Sd = dense_osee_fermionic(W, modes)
    return Sf, Sd, t1, t2
end
