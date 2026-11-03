# ============================================================================
# entropy.jl —— 协方差谱内核：entropy(gs[, modes]) 与 purity(gs)
# ============================================================================

function _binary_entropy(ν::Real)
    p = clamp((1 + ν) / 2, 0.0, 1.0)
    return (p ≤ 0 || p ≥ 1) ? 0.0 : -p * log(p) - (1 - p) * log(1 - p)
end

function _bosonic_entropy(ν::Real)
    n̄ = max(ν - 0.5, 0.0)
    return n̄ ≤ 0 ? 0.0 : (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
end

function _entropy_fermionic(ΓA::AbstractMatrix{<:Real}, α::Real)
    νs = abs.(eigvals(Hermitian(im .* ΓA)))
    if isapprox(α, 1; atol=1e-12)
        return sum(_binary_entropy, νs) / 2
    end
    S = 0.0
    for ν in νs
        p₊ = clamp((1 + ν) / 2, 0.0, 1.0); p₋ = 1 - p₊
        S += log(p₊^α + p₋^α)
    end
    return S / (2 * (1 - α))
end

function _entropy_bosonic(ΓA::AbstractMatrix{<:Real}, ΩA::AbstractMatrix{<:Real}, α::Real)
    νs = abs.(eigvals(ΩA * ΓA))
    if isapprox(α, 1; atol=1e-12)
        return sum(_bosonic_entropy, νs) / 2
    end
    S = 0.0
    for ν in νs
        n̄ = max(ν - 0.5, 0.0)
        r = n̄ / (1 + n̄)
        S += -log1p(-(r^α)) - α * log1p(n̄)
    end
    return S / (2 * (1 - α))
end

function _check_modes(modes, N)
    all(1 .≤ modes .≤ N) || throw(BoundsError(1:N, modes))
    allunique(modes) || throw(ArgumentError("modes 含重复指标"))
    return nothing
end

_entropy_at(s::FermionicGaussianState, idx::Vector{Int}, α::Real) =
    _entropy_fermionic(s.Γ[idx, idx], α)

function _entropy_at(s::BosonicGaussianState, idx::Vector{Int}, α::Real)
    N = nmodes(s)
    Ω = _omega(N)
    return _entropy_bosonic(s.Σ[idx, idx], Ω[idx, idx], α)
end

"""
    entropy(s::GaussianState; α=1)
    entropy(s::GaussianState, modes; α=1)

整态 / 子系统（物理模式 `modes`）的 von Neumann（`α=1`）或 Rényi 熵。
费米子系统按代数（模式）约化；非连续割不要拿朴素偏迹对照。
"""
function entropy(s::AbstractGaussianState; α::Real=1)
    α > 0 || throw(ArgumentError("Rényi 阶数 α 必须为正"))
    idx = collect(1:size(covariance(s), 1))
    return _entropy_at(s, idx, α)
end
entropy(s::AbstractGaussianState, mode::Integer; α::Real=1) = entropy(s, [mode]; α=α)
function entropy(s::AbstractGaussianState, modes::AbstractVector{<:Integer}; α::Real=1)
    α > 0 || throw(ArgumentError("Rényi 阶数 α 必须为正"))
    N = nmodes(s)
    _check_modes(modes, N)
    isempty(modes) && return 0.0
    return _entropy_at(s, _modeindices(modes), α)
end

# purity：Tr ρ²。谱的 ± 对（费米）/辛 ± 对（玻色）在全谱乘积中各计两次，
# 开方即还原每模式一次（与 entropy 内核的 ÷2 同构）
function _purity_fermionic(ΓA::AbstractMatrix{<:Real})
    νs = abs.(eigvals(Hermitian(im .* ΓA)))
    return sqrt(prod((1 .+ νs .^ 2) ./ 2))
end

function _purity_bosonic(ΣA::AbstractMatrix{<:Real}, ΩA::AbstractMatrix{<:Real})
    νs = abs.(eigvals(ΩA * ΣA))
    return sqrt(prod(1 ./ (2 .* νs)))
end

"""
    purity(s::AbstractGaussianState) -> Float64

整态纯度 `Tr ρ²`，由协方差谱解析给出：费米 `Πₖ(1 + νₖ²)/2`
（`νₖ = |eig(iΓ)|`，成 ± 对）；玻色 `Πₖ 1/(2νₖ)`（`νₖ` 为辛本征值
`|eig(ΩΣ)|`，物理态 `νₖ ≥ ½`）。纯态为 1。子系统纯度用
`purity(partialtr(s, modes))`。
"""
purity(s::FermionicGaussianState) = _purity_fermionic(s.Γ)
purity(s::BosonicGaussianState) = _purity_bosonic(s.Σ, _omega(nmodes(s)))
