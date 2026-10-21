# ============================================================================
# fermionic.jl —— 费米高斯态
#
# 两种输入（构造时校验对称性；具体计算惰性进行并缓存）：
#   1. 厄米 BdG 矩阵（一般混合态）：ρ = exp(−½ψ†·bdg·ψ)/Z，m 个物理模
#      （bdg 为 2m×2m，Γ 为 2m×2m）；doubled 纯态协方差 Γd 为 4m×4m。
#   2. 复反对称配对矩阵（纯态）：|Ψ⟩ = exp(½c†·bdg·c†)|0⟩，bdg 为 2n×2n，
#      直接描述 doubled 空间的 2n 个模；此时 Γ = Γd（同一缓存对象）。
# osee 统一为“Γd 限制到给定 doubled 模式子集的协方差熵”，两个分支同一条路径。
#
# 实数性：对任意态，γ_aγ_b − δ_ab 是反厄米算符 ⟹ 期望纯虚 ⟹
# Γ_ab = i⟨γ_aγ_b⟩ 严格实反对称（与 bdg 是否复无关；数值上再取实并反对称化）。
# ============================================================================

"""
    FermionicGaussianState(bdg::Matrix{T}) where {T<:Number}

费米高斯态。`bdg` 是二次型哈密顿量的系数矩阵，态取

```text
ρ = exp(−½ ψ†·bdg·ψ) / Z ,      ψ = (c₁..c_m, c†₁..c†_m)
```

即 `bdg` 已吸收逆温度，给出**一般（混合）高斯态**；纯态是其谱取极限的特例。
协方差 `Γ_ab = i⟨γ_aγ_b⟩`（实反对称，任意复 `bdg` 下均严格实——γ_aγ_b − δ 反厄米）
由 [`covariance`](@ref) 惰性计算并缓存。

`bdg` 允许两种形式（构造时校验，必须是其中之一）：

1. **厄米**（一般情形）：`bdg` 为 2m×2m 厄米矩阵，描述 m 个物理模，`Γ` 为
   2m×2m。不必是规范形式 `[h A; A† −hᵀ]`——任意厄米矩阵与它的规范形式
   （`h = (M₁₁−M₂₂ᵀ)/2`，`A = M₁₂`）作为算符 H = ½ψ†Mψ 只差一个常数，
   构造态时自动规范化。计算用 `⟨ψ_a†ψ_b⟩ = [Ū f(Λ) Uᵀ]_ab`。
2. **复反对称**（纯配对态 `|Ψ⟩ = exp(½c†·bdg·c†)|0⟩`，`bdg` 为 2n×2n，
   描述 doubled 空间的 2n 个模）：`Γ` 为 4n×4n，且 `Γd === Γ`。
   纯配对时走 SVD 闭式，一般情形解湮灭条件给出的 Stein 方程。
   这是 OSEE/影响泛函系数态用的分支。

字段 `isbdg`：`true` ⟺ 输入是厄米二次型（ρ 为真正的密度矩阵，[`entropy`](@ref)
有定义）；`false` ⟺ 纯配对算符 exp(½c†Wc†)（ρ 非密度矩阵，`entropy` 报错，
但 [`osee`](@ref) 经纯化仍有定义——对任意向量 Majorana 协方差恒实）。
`Γ`/`Γd` 惰性计算并缓存；对 `bdg` 的要求：方阵、偶数维、厄米或反对称之一。
"""
mutable struct FermionicGaussianState{T<:Number} <: AbstractGaussianState
    bdg::Matrix{T}
    isbdg::Bool                            # true ⟺ 厄米二次型（ρ 是密度矩阵）
    Γ::Union{Nothing,Matrix{Float64}}     # Majorana 协方差（entropy 用；惰性缓存）
    Γd::Union{Nothing,Matrix{Float64}}   # doubled 纯态协方差（osee 用；惰性缓存）。
                                        # Gibbs 分支不支持 doubled 构造时保持/抛错
    function FermionicGaussianState(bdg::Matrix{T}) where {T<:Number}
        N = size(bdg, 1)
        size(bdg, 2) == N || throw(ArgumentError("bdg 必须是方阵"))
        (N ≥ 2 && iseven(N)) || throw(ArgumentError("bdg 必须是偶数维 2n×2n 矩阵"))
        W = Matrix{ComplexF64}(bdg)
        if _antisymmetry_error(W)
            return new{T}(bdg, false, nothing, nothing)   # 纯配对算符（非密度矩阵）
        elseif _ishermitian_approx(W)
            return new{T}(bdg, true, nothing, nothing)    # Gibbs 密度矩阵
        else
            throw(ArgumentError(
                "费米 bdg 必须是**反对称**（纯高斯态 exp(½c†Wc†)|0⟩）或**厄米**" *
                "（Gibbs 态 exp(−½ψ†Wψ)/Z）矩阵"))
        end
    end
end

# 输入性质判别（bdg 的纯函数，供惰性计算路径使用；构造时已校验合法性）
_ispairing(bdg::AbstractMatrix) = _antisymmetry_error(Matrix{ComplexF64}(bdg))

# 任意厄米 M 与规范 BdG 形式只差一个常数（作为算符 H = ½ψ†Mψ 等价）：
#   H = ½[(M₁₁−M₂₂ᵀ)c†c + M₁₂c†c† + M₂₁cc] + ½tr(M₂₂)
# 取 h = (M₁₁−M₂₂ᵀ)/2、A = M₁₂ 即得规范形式（差常数 ½tr(M₂₂)+½tr(h)，
# 不改变归一化态）。谱公式都在规范形式下推导，故计算前先规范化。
# 注意这不是物理上的粒子–空穴对称（能谱半满对称），只是 Nambu 表示的规范选择。
function _canonical_bdg(W::AbstractMatrix{<:Number})
    N = size(W, 1); m = N ÷ 2
    h = (W[1:m, 1:m] - transpose(W[m+1:N, m+1:N])) ./ 2
    A = (W[1:m, m+1:N] - transpose(W[1:m, m+1:N])) ./ 2   # 反对称化（对称部分在算符中恒为零）
    B = similar(W)
    B[1:m, 1:m] .= h
    B[1:m, m+1:N] .= A
    B[m+1:N, 1:m] .= -conj(A)                            # A† = −Ā（A 反对称）
    B[m+1:N, m+1:N] .= .-transpose(h)
    return B
end

# Γ 的惰性计算与缓存
function _gamma(s::FermionicGaussianState)
    s.Γ === nothing || return s.Γ
    W = Matrix{ComplexF64}(s.bdg)
    s.Γ = _ispairing(W) ?
        _fermionic_gamma_pure(W) :
        _fermionic_gamma_gibbs(eigen(Hermitian(_canonical_bdg(W))))
    return s.Γ
end

# Γd 的惰性计算与缓存（纯配对分支：与 Γ 同一对象）
function _gammad(s::FermionicGaussianState)
    s.Γd !== nothing && return s.Γd
    W = Matrix{ComplexF64}(s.bdg)
    _ispairing(W) && return s.Γd = _gamma(s)
    g = _fermionic_gibbs_doubled_gamma(eigen(Hermitian(_canonical_bdg(W))))
    g === nothing && throw(ArgumentError(
        "Gibbs 分支的 OSEE 需要 BdG 谱不含（近）零模（简并本征矢无法配对）；" *
        "当前 bdg 不满足。entropy（偏迹熵）不受影响"))
    return s.Γd = g
end

"""
    fermionic_bdg_matrix(h::AbstractMatrix, g::AbstractMatrix=zero(h))

由左上的 particle–hole 块 `h` 与右上的配对块 `g` 组装费米 BdG 矩阵
（模式序 `ψ = (c₁..c_N, c†₁..c†_N)`，与 `ImpurityModelBase.bcs_cmatrix` 一致）：

```text
⎡  h     A  ⎤
⎣  A†   −hᵀ ⎦ ,        A = g − gᵀ
```

只有**两个独立块**：`(2,2)` 块由反对易关系 `c_i c†_j = δ_ij − c†_j c_i` 合并进
`(1,1)` 块（只差常数），故规范取 `−hᵀ`；`g` 只有反对称部分进入二次型，这里
按 `bcs_cmatrix` 的约定取 `A = g − gᵀ`。

- `h`：厄米（实数时对称）；
- `g`：任意方阵（其对称部分不影响态）。

结果为**规范 BdG 形式**（满足 `M·S₀ = −S₀·M̄`，`S₀ = [[0,I],[I,0]]`）。
注意这只是 Nambu 表示的规范选择，**不是**物理上的粒子–空穴对称（能谱半满
对称）；任意厄米矩阵与该规范形式只差一个常数（作为算符 H = ½ψ†Mψ 等价），
本包在构造态时会自动规范化，无需手动调用本函数。
"""
function fermionic_bdg_matrix(h::AbstractMatrix, g::AbstractMatrix=zero(h))
    n = size(h, 1)
    size(h, 2) == n || throw(ArgumentError("h 必须是方阵"))
    size(g) == (n, n) || throw(ArgumentError("g 必须与 h 同尺寸（$(n)×$(n)）"))
    tol = 1e-10 * max(1.0, opnorm(h, Inf))
    opnorm(h - h', Inf) ≤ tol ||
        throw(ArgumentError("费米 BdG 矩阵要求 h 厄米（h† = h）"))

    T = promote_type(eltype(h), eltype(g))
    A = Matrix{T}(g) .- Matrix{T}(transpose(g))
    B = Matrix{T}(undef, 2n, 2n)
    B[1:n, 1:n] .= h
    B[1:n, n+1:2n] .= A
    B[n+1:2n, 1:n] .= A'
    B[n+1:2n, n+1:2n] .= .-transpose(h)
    return B
end

# ========================================================================== #
# bdg → Γ（公式与 GTEMPO/osee 已验证实现逐行一致）
# ========================================================================== #

# 纯 ā–a 配对（bdg[ā, a] 块，可为复）：SVD 闭式二点函数。
# 模式序 [a₁..aₙ; ā₁..āₙ]。返回 G = ⟨c†c⟩（块对角）与 F = ⟨cc⟩（反对称）。
function _correlations_pairing(Δ::AbstractMatrix{<:Number})
    n = size(Δ, 1)
    Fs = svd(Δ)
    ε = Fs.S .^ 2 ./ (1 .+ Fs.S .^ 2)                 # 配对模式占据
    s̃ = Fs.S ./ (1 .+ Fs.S .^ 2)
    A = Fs.V * Diagonal(ε) * Fs.V'                    # ⟨c'†c'⟩ = V ε V†
    B = conj(Fs.U) * Diagonal(ε) * transpose(Fs.U)    # ⟨c̄†c̄⟩ = Ū ε Uᵀ
    C = -(Fs.U * Diagonal(s̃) * Fs.V')                # ⟨c̄c'⟩ = −U s̃ V†
    G = zeros(ComplexF64, 2n, 2n)
    F = zeros(ComplexF64, 2n, 2n)
    G[1:n, 1:n] .= A
    G[n+1:2n, n+1:2n] .= B
    F[1:n, n+1:2n] .= .-transpose(C)                  # ⟨c'_q c̄_p⟩ = −⟨c̄_p c'_q⟩
    F[n+1:2n, 1:n] .= C
    return G, F
end

# 一般复反对称 W：α = c − Wc† 湮灭态（BCH 精确终止）⟹
#   ⟨α†c⟩ = 0  ⟹ G + W̄ Ḡ Wᵀ = W̄ Wᵀ（混合共轭 Stein，注意是 Ḡ）
#   ⟨α†c†⟩ = 0 ⟹ F = W G − W（无需求逆，W 奇异亦可用）
# 解耦：K = W⊗W̄，(I − K K̄) vec(G) = c − K c̄，c = vec(W̄Wᵀ)。
function _correlations_stein(W::AbstractMatrix{<:Number})
    n = size(W, 1)
    K = kron(W, conj(W))
    Ibig = Matrix{ComplexF64}(I, n * n, n * n)
    c = vec(conj(W) * transpose(W))
    G = reshape((Ibig - K * conj(K)) \ (c - K * conj(c)), n, n)
    F = W * G - W
    G = (G + G') ./ 2                                 # Hermitian 对称化
    F = (F .- transpose(F)) ./ 2                      # 反对称化（复矩阵不能用 '）
    return G, F
end

# Majorana 协方差：Γ_ab = i⟨γ_aγ_b⟩（实反对称）。
# G_{ij} = ⟨c_i†c_j⟩，F_{ij} = ⟨c_i c_j⟩，δ = Kronecker δ。
# 注意 Γ_{2i,2j−1} 必须直接算（不能用 −M[2j−1,2i]：双重循环会读到未写入元素）。
function _majorana_covariance(G::AbstractMatrix{<:Number}, F::AbstractMatrix{<:Number})
    n = size(G, 1)
    M = zeros(ComplexF64, 2n, 2n)                     # ⟨γ_aγ_b⟩
    for i in 1:n, j in 1:n
        δ = ifelse(i == j, 1.0, 0.0)
        M[2i-1, 2j-1] = F[i,j] + δ - G[j,i] + G[i,j] + conj(F[j,i])
        M[2i-1, 2j]   = 1im * (δ - G[j,i] - G[i,j] - F[i,j] - conj(F[i,j]))
        M[2i, 2j-1]   = 1im * (G[i,j] + G[j,i] + F[j,i] + conj(F[j,i]) - δ)
        M[2i, 2j]     = conj(F[i,j]) + G[i,j] + δ - G[j,i] - F[i,j]
    end
    Γ = real.(1im .* M)
    return (Γ .- Γ') ./ 2                             # 数值反对称化
end

function _antisymmetry_error(W::AbstractMatrix{<:Number})
    s = opnorm(W, Inf)
    tol = 1e-9 * max(s, eps(float(real(one(eltype(W))))))
    return opnorm(W + transpose(W), Inf) ≤ tol
end

function _ishermitian_approx(W::AbstractMatrix{<:Number})
    s = opnorm(W, Inf)
    tol = 1e-9 * max(s, eps(float(real(one(eltype(W))))))
    return opnorm(W - W', Inf) ≤ tol
end

# 是否纯 ā–a 配对（对角块为零）：走 SVD 闭式，否则走 Stein。
function _ispurepairing(W::AbstractMatrix{<:Number}, m::Int)
    tol = 1e-11 * max(1.0, opnorm(W, Inf))
    return opnorm(W[1:m, 1:m], Inf) ≤ tol && opnorm(W[m+1:2m, m+1:2m], Inf) ≤ tol
end

# -------------------------------------------------------------------------- #
# 分支 A：bdg 为复反对称配对矩阵 → 纯高斯态 |Ψ⟩ = exp(½c†·bdg·c†)|0⟩
# -------------------------------------------------------------------------- #
function _fermionic_gamma_pure(W::AbstractMatrix{<:Number})
    N = size(W, 1)
    m = N ÷ 2                                         # a 模式数 = ā 模式数
    if _ispurepairing(W, m)
        G, F = _correlations_pairing(W[m+1:N, 1:m])
    else
        G, F = _correlations_stein(W)
    end
    return _majorana_covariance(G, F)
end

# -------------------------------------------------------------------------- #
# 分支 B：bdg 为厄米二次型系数 → Gibbs 态 ρ ∝ exp(−½ ψ†·bdg·ψ)
#
# ψ = (c₁..c_m, c†₁..c†_m)。令 M = bdg = U Λ U†，则
#   ⟨ψ_a† ψ_b⟩ = [Ū f(Λ) Uᵀ]_ab ,   f(ε) = 1/(1+e^ε)
#（注意共轭在行指标一侧：⟨ψ_a†ψ_b⟩ = Σ_k conj(u_k)_a (u_k)_b f(λ_k)；
#  实本征矢时退化为矩阵函数 f(M) = U f U†）
# 于是
#   ⟨c†_i c_j⟩ = Gψ[i, j] ,      ⟨c_i c_j⟩ = Gψ[m+i, j]
# 再组装 2m×2m Majorana 协方差（m 个模，2m 个 Majorana）。
# -------------------------------------------------------------------------- #
function _fermionic_gamma_gibbs(E::Eigen)
    N = length(E.values)
    m = N ÷ 2
    f = 1 ./ (1 .+ exp.(E.values))
    Gψ = conj(E.vectors) * Diagonal(f) * transpose(E.vectors)
    G = Gψ[1:m, 1:m]                                  # ⟨c†_i c_j⟩
    F = Gψ[m+1:N, 1:m]                                # ⟨c_i c_j⟩
    return _majorana_covariance(G, F)
end

# ========================================================================== #
# Gibbs 分支的 doubled 协方差（OSEE 用）
#
# |ρ⟩⟩ = (ρ⊗I)|I⟩_G 是 doubled (ket,bra) 空间 [c^A(1..m); d^B(m+1..2m)] 的
# **纯高斯态**。设 M = bdg 的规范 BdG 形式（任意厄米输入已先规范化——与原矩阵
# 只差常数，不改变态），本征分解 M u_k = λ_k u_k 取正半谱 λ_k>0：
#   f_k = Σ_a conj(u_{a,k}) ψ_a          （湮灭型模）
#   f_k† 的系数列 = S₀·conj(u_k)          （−λ 伴模，规范结构保证）
# ρ₀ = Π_k exp(−λ_k f_k†f_k)（Bogoliubov 对角化），
# |ρ⟩⟩ = Π_k (|00⟩ − t_k|11⟩)_{(f_k^A, f̄*_k^B)} / √(1+t_k²)，t_k = e^{−λ_k}
#（单对符号已对稠密 |ρ⟩⟩ 仲裁；费米约化用 Z₂ 一致的协方差定义）。
# 由此（模式序 [A 拷贝; B 拷贝]）：
#   A/B 同侧块 = ρ² 的关联：⟨ψ†ψ⟩ = f(2M)（与 tr(ρ²·) 稠密对照一致）
#   交叉配对块 F_AB = −(V₊ t̃ V₊† + V₋ t̃ V₋†)，t̃_k = 1/(2cosh λ_k)
#   交叉 ⟨c†d⟩ = 0（U₊ᵀU₋ 的 c-c 块为零，规范结构保证）
# 公式对简并本征空间的基选择规范不变（t̃ 在简并块上为常数）。
# 返回 4m×4m Majorana 协方差；谱含（近）零模使简并本征矢无法配对时返回 nothing。
# ========================================================================== #

function _nambu_swap(m::Int)
    N = 2m
    S₀ = zeros(ComplexF64, N, N)
    for i in 1:m
        S₀[i, m+i] = 1.0
        S₀[m+i, i] = 1.0
    end
    return S₀
end

function _fermionic_gibbs_doubled_gamma(E::Eigen)
    N = length(E.values)
    m = N ÷ 2
    # A 拷贝 = ρ² 的关联：⟨ψ†ψ⟩ = Ū f(2Λ) Uᵀ（共轭在行指标一侧）
    f2 = 1 ./ (1 .+ exp.(2 .* E.values))
    Gψ = conj(E.vectors) * Diagonal(f2) * transpose(E.vectors)
    G_A = Gψ[1:m, 1:m]                                  # ⟨c†c⟩
    F_A = Gψ[m+1:N, 1:m]                                # ⟨cc⟩
    # 交叉配对块：+λ 半谱与 PH 伴模
    perm = sortperm(E.values, rev=true)[1:m]             # +λ 半谱（BdG ± 对称）
    λ = E.values[perm]
    U₊ = E.vectors[:, perm]                             # Nambu 本征矢（2m×m）
    U₋ = _nambu_swap(m) * conj(U₊)                      # f_k† 的系数列（PH 伴模）
    𝒰 = hcat(U₊, U₋)
    opnorm(𝒰' * 𝒰 - I, Inf) ≤ 1e-8 || return nothing   # （近）零模：无法配对
    V₊ = U₊[1:m, :]                                     # c_i = Σ V₊[i,k] f_k + V₋[i,k] f_k†
    V₋ = U₋[1:m, :]
    t̃ = 1 ./ (2 .* cosh.(λ))
    F_AB = -(V₊ * Diagonal(t̃) * V₊' + V₋ * Diagonal(t̃) * V₋')
    # 交叉 ⟨c_i†d_j⟩ 复数时非零（τ-演化推导 + 稠密仲裁；实数时 H ≡ 0 退化为旧结论）
    H_AB = V₊ * Diagonal(t̃) * transpose(V₋) + V₋ * Diagonal(t̃) * transpose(V₊)
    G_AB = -conj(H_AB)
    G = zeros(ComplexF64, 2m, 2m)
    F = zeros(ComplexF64, 2m, 2m)
    G[1:m, 1:m] = G_A
    G[m+1:2m, m+1:2m] = conj(G_A)
    G[1:m, m+1:2m] = G_AB
    G[m+1:2m, 1:m] = G_AB'
    F[1:m, 1:m] = F_A
    F[m+1:2m, m+1:2m] = -conj(F_A)                   # B 拷贝为共轭表示（稠密仲裁）
    F[1:m, m+1:2m] = F_AB
    F[m+1:2m, 1:m] = -transpose(F_AB)
    return _majorana_covariance(G, F)
end

# ========================================================================== #
# osee（协方差快速算法）
# ========================================================================== #

"""
    osee(s::FermionicGaussianState, sites::AbstractVector{<:Integer}; α=1) -> Float64

**operator-space entanglement entropy**，全部由协方差矩阵计算，复杂度
O(模式数³)，不构造稠密矩阵。`α = 1` 为 von Neumann 熵，`α ≠ 1` 为 Rényi 熵。

`sites` 是 **doubled 纯态空间**（doubled 协方差 `Γd`，惰性计算并缓存）的
模式指标（1-based）：

- **Gibbs 分支**（厄米 `bdg`，m 个物理模）：doubled 空间 = ket/bra 两个拷贝，
  模式 `1..m` 为 ket 拷贝、`m+1..2m` 为 bra 拷贝；物理 site p 的割为
  模式 `[p, m+p]`。
- **纯配对分支**（反对称 `bdg`，2n 个模）：doubled 空间即 Grassmann 系数态
  本身，模式 `1..2n` 直接对应 `(a₁..aₙ, ā₁..āₙ)`；site p 的割为 `[p, n+p]`
  ——与 GTEMPO GrassmannMPS 的块结构/bond 割一致（`GTEMPO/osee/osee_report.md`）。

费米约化采用 Z₂ 一致（协方差）的定义——与 Grassmann/IF 形式体系和
GrassmannMPS bond 熵同一约定；注意它与 plain 张量 reshape 的
operator-Schmidt 在非连续割上不同。

Gibbs 分支的谱公式在规范 BdG 形式下推导；任意厄米输入构造态时已自动规范化
（与原矩阵只差常数，不改变态），无需手动处理。仅当谱含（近）零模使简并
本征矢无法配对时，调用报 `ArgumentError`（`entropy` 不受影响）。
"""
function osee(s::FermionicGaussianState, sites::AbstractVector{<:Integer}; α::Real=1)
    α > 0 || throw(ArgumentError("Rényi 阶数 α 必须为正"))
    allunique(sites) || throw(ArgumentError("sites 含重复指标"))
    Γd = _gammad(s)                                    # 不支持时抛 ArgumentError
    M = size(Γd, 1) ÷ 2
    all(1 .≤ sites .≤ M) || throw(BoundsError(1:M, sites))
    isempty(sites) && return 0.0
    idx = _siteindices(sort(collect(Int, sites)))
    return _entropy_fermionic(Γd[idx, idx], α)
end

# ========================================================================== #
# 稠密构造（仅供 densitymatrix 与测试对照；osee 永不使用）
# ========================================================================== #

function _jw_annihilator(m::Int, i::Int)
    dim = 1 << m
    A = zeros(ComplexF64, dim, dim)
    bit = 1 << (i - 1); mask = bit - 1
    for x in 0:(dim - 1)
        (x & bit) == 0 && continue
        sg = iseven(count_ones(x & mask)) ? 1.0 : -1.0
        A[(x ⊻ bit) + 1, x + 1] = sg
    end
    return A
end

function _fermionic_dense_rho(bdg::AbstractMatrix)
    N = size(bdg, 1); m = N ÷ 2
    dim = 1 << m
    c = [_jw_annihilator(m, i) for i in 1:m]
    cd = [adjoint(x) for x in c]
    ψ = vcat(c, cd); ψd = vcat(cd, c)
    H = zeros(ComplexF64, dim, dim)
    for a in 1:N, b in 1:N
        iszero(bdg[a, b]) && continue
        H .+= (bdg[a, b] / 2) .* (ψd[a] * ψ[b])
    end
    H = (H + H') ./ 2
    E = eigen(Hermitian(H))
    w = exp.(-E.values); w ./= sum(w)
    return E.vectors * Diagonal(w) * E.vectors'
end

# |Ψ⟩ = exp(½ c†Wc†)|0⟩ 的系数矢量（B 幂零，Taylor 级数精确截断）
function _fermionic_pure_vector(W::AbstractMatrix{<:Number})
    N = size(W, 1)
    dim = 1 << N
    ψ = zeros(ComplexF64, dim); ψ[1] = 1
    term = copy(ψ)
    buf = zeros(ComplexF64, dim)
    t1 = zeros(ComplexF64, dim)
    for order in 1:N
        fill!(buf, 0)
        for b in 1:N, a in 1:N
            iszero(W[a, b]) && continue
            fill!(t1, 0)
            _jw_create!(t1, term, b, 1.0)
            _jw_create!(buf, t1, a, W[a, b] / 2)
        end
        buf ./= order
        ψ .+= buf
        copyto!(term, buf)
    end
    return ψ
end

# JW 产生算符作用于矢量（系数累加；模式 r ↔ bit r−1）
function _jw_create!(out::AbstractVector{ComplexF64},
                     v::AbstractVector{ComplexF64}, r::Int, coef)
    bit = 1 << (r - 1); mask = bit - 1
    @inbounds for x in 0:(length(v)-1)
        (x & bit) != 0 && continue
        sg = iseven(count_ones(x & mask)) ? coef : -coef
        out[(x | bit) + 1] += sg * v[x + 1]
    end
    return out
end

"""
    densitymatrix(s::FermionicGaussianState) -> Matrix{ComplexF64}

稠密密度矩阵（调试/对照专用；`osee`/`entropy` 用协方差快速算法，不需要它）。

- Gibbs 分支（厄米 `bdg`）：物理 Fock 空间（m 个模，2^m×2^m）中的
  ρ = exp(−½ψ†·bdg·ψ)/Z；
- 纯配对分支（反对称 `bdg`）：doubled Fock 空间（2n 个模，2^{2n}×2^{2n}）
  中的 |Ψ⟩⟨Ψ|/⟨Ψ|Ψ⟩，|Ψ⟩ = exp(½c†·bdg·c†)|0⟩。
"""
function densitymatrix(s::FermionicGaussianState)
    if _ispairing(s.bdg)
        ψ = _fermionic_pure_vector(s.bdg)
        ψ ./= norm(ψ)
        return ψ * ψ'
    end
    return _fermionic_dense_rho(s.bdg)
end
