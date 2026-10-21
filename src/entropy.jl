# ============================================================================
# entropy.jl —— 偏迹子系统熵（协方差快速算法，以**物理模式/site** 指标索引）
#
# 定义域：仅对 isbdg = true 的输入（ρ = exp(−½ψ†Mψ)/Z，M 厄米 / K 实对称正定，
# 即 ρ 是真正的密度矩阵）有定义。isbdg = false 的输入（如费米纯配对算符
# exp(½c†Wc†)）不是密度矩阵，调用直接报错——其纠缠量请用 osee。
# ============================================================================ #

# 玻色只接受密度矩阵形态输入（恒可定义）；费米按 isbdg 判定
_check_densitymatrix(::BosonicGaussianState) = nothing
_check_densitymatrix(s::FermionicGaussianState) =
    s.isbdg || throw(ArgumentError(
        "entropy 只对密度矩阵形态的输入（isbdg = true）有定义；当前对象是 " *
        "exp(非厄米二次型) 型算符（ρ 不是密度矩阵）。其纠缠量请用 osee" *
        "（经向量化纯化，对算符同样有定义）。"))

"""
    entropy(s::AbstractGaussianState; α=1) -> Float64

整个高斯态的熵：`α = 1` 为 von Neumann 熵，`α ≠ 1` 为 Rényi 熵。
纯态返回 0；混合态返回其全部谱的熵。要求 `s.isbdg == true`（密度矩阵形态）。
"""
function entropy(s::AbstractGaussianState; α::Real=1)
    _check_densitymatrix(s)
    return _entropy_from_gamma(s, collect(1:size(covariance(s), 1)); α=α)
end

"""
    entropy(s::AbstractGaussianState, sites; α=1) -> Float64

**子系统熵**：对补集作偏迹后，约化密度矩阵的 von Neumann / Rényi 熵。
`sites` 为模式（site）指标（1-based，可传单个整数或向量），索引**物理模式**
（协方差 `Γ` 的模式：费米 Gibbs 为 m 个模，玻色为 n 个模；与 `osee` 的
doubled 模式指标不同）。要求 `s.isbdg == true`。

与 [`osee`](@ref) 不同：后者是 operator-space 纠缠熵，对非密度矩阵形态
（`isbdg = false`）的算符输入同样有定义。
"""
function entropy(s::AbstractGaussianState, sites::AbstractVector{<:Integer}; α::Real=1)
    _check_densitymatrix(s)
    α > 0 || throw(ArgumentError("Rényi 阶数 α 必须为正"))
    m = nmodes(s)
    all(1 .≤ sites .≤ m) || throw(BoundsError(1:m, sites))
    allunique(sites) || throw(ArgumentError("sites 含重复指标"))
    isempty(sites) && return 0.0
    return _entropy_from_gamma(s, _siteindices(sites); α=α)
end
entropy(s::AbstractGaussianState, site::Integer; α::Real=1) =
    entropy(s, Int[site]; α=α)

# 由协方差的行指标子集求（偏迹后的）熵
function _entropy_from_gamma(s::AbstractGaussianState, idx::Vector{Int}; α::Real=1)
    Γ = covariance(s)
    if s isa FermionicGaussianState
        return _entropy_fermionic(Γ[idx, idx], α)
    end
    Ω = _omega(size(Γ, 1) ÷ 2)
    return _entropy_bosonic(Γ[idx, idx], Ω[idx, idx], α)
end
