# ============================================================================
# common.jl —— 抽象类型、访问器、指标工具与熵内核（费米/玻色通用）
# ============================================================================

abstract type AbstractGaussianState end

"协方差矩阵 Γ（惰性计算并缓存；费米为 Majorana 协方差，玻色为正则坐标协方差）。"
covariance(s::AbstractGaussianState) = _gamma(s)

"模式（site）数：`size(covariance(s),1) ÷ 2`（每个模式 2 个自由度）。"
nmodes(s::AbstractGaussianState) = size(covariance(s), 1) ÷ 2

# 模式指标 → Γ 行指标（每个模式占 2 个：Majorana γ_{2p−1},γ_{2p} 或正则坐标 x_p,p_p）
function _siteindices(sites)
    idx = Int[]
    for p in sites
        push!(idx, 2p - 1, 2p)
    end
    return idx
end

# 辛形式 Ω = ⊕ₖ [[0,1],[−1,0]]（2n×2n，正则坐标序 (x₁,p₁,x₂,p₂,…)）
function _omega(n::Int)
    Ω = zeros(2n, 2n)
    for k in 1:n
        Ω[2k-1, 2k] = 1.0
        Ω[2k, 2k-1] = -1.0
    end
    return Ω
end

function _realify(name, m::AbstractMatrix)
    eltype(m) <: Real && return Matrix{Float64}(m)
    maximag = maximum(abs, imag.(m))
    maximag ≤ 1e-9 * max(1.0, maximum(abs, real.(m))) ||
        throw(ArgumentError("$name 必须是实矩阵（虚部范数 $maximag 过大）"))
    return real.(Matrix{Float64}(m))
end

# ========================================================================== #
# 熵内核
# ========================================================================== #

# 费米单对：h((1+ν)/2)，二能级 p± = (1∓ν)/2
function _binary_entropy(ν::Real)
    p = clamp((1 + ν) / 2, 0.0, 1.0)
    return (p ≤ 0 || p ≥ 1) ? 0.0 : -p * log(p) - (1 - p) * log(1 - p)
end

# 玻色单模：n̄ = ν − ½，S = (n̄+1)ln(n̄+1) − n̄ ln n̄
function _bosonic_entropy(ν::Real)
    n̄ = max(ν - 0.5, 0.0)
    return n̄ ≤ 0 ? 0.0 : (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
end

function _entropy_fermionic(ΓA::AbstractMatrix{<:Real}, α::Real)
    νs = abs.(eigvals(Hermitian(1im .* ΓA)))          # ±ν 成对
    if isapprox(α, 1; atol=1e-12)
        return sum(_binary_entropy, νs) / 2
    end
    S = 0.0
    for ν in νs
        p₊ = clamp((1 + ν) / 2, 0.0, 1.0)
        p₋ = 1 - p₊
        S += log(p₊^α + p₋^α)
    end
    return S / (2 * (1 - α))                          # ±ν 各贡献一次，除以 2
end

function _entropy_bosonic(ΓA::AbstractMatrix{<:Real}, ΩA::AbstractMatrix{<:Real}, α::Real)
    νs = abs.(eigvals(ΩA * ΓA))                       # ±iν → 每个值出现两次
    if isapprox(α, 1; atol=1e-12)
        return sum(_bosonic_entropy, νs) / 2
    end
    S = 0.0
    for ν in νs
        n̄ = max(ν - 0.5, 0.0)
        # 热模谱 p_k = n̄^k/(1+n̄)^{k+1} ⟹
        # ln Σ p_k^α = −α ln(1+n̄) − ln(1 − (n̄/(1+n̄))^α)
        r = n̄ / (1 + n̄)
        S += -log1p(-(r^α)) - α * log1p(n̄)
    end
    return S / (2 * (1 - α))
end
