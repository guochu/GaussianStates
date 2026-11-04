# ============================================================================
# gaussian_state.jl —— 只存协方差的高斯态
#
# 费米：Γ_ab = i⟨γ_aγ_b⟩（2N×2N 实反对称；Majorana 基）
# 玻色：Σ_ab = ½⟨{R_a,R_b}⟩（2N×2N 实对称正定；R=(x₁,p₁,…)）
# 协方差为实 Float64，不参数化。物理性校验在构造器里：
# 非法协方差（不对称/非物理/不正定）抛 ArgumentError。
# ============================================================================

function _omega(n::Int)
    Ω = zeros(2n, 2n)
    for k in 1:n
        Ω[2k-1, 2k] = 1.0
        Ω[2k, 2k-1] = -1.0
    end
    return Ω
end

"""
    FermionicGaussianState(Γ)

费米高斯态：只存实 Majorana 协方差 `Γ`（2N×2N 反对称，
`Γ_ab = i⟨γ_aγ_b⟩`）。基矢 `γ₂ₘ₋₁ = cₘ + cₘ†`、`γ₂ₘ = i(cₘ† − cₘ)`，
模式 `m` 占相邻指标 `[2m−1, 2m]`。构造器校验：实反对称且
`|eig(iΓ)| ≤ 1`。表象与算子 block 的 Nambu 基不同（见 `bdg`）。
通常由 `purify` / `partialtr` / `GaussianState(Gibbs)` 产生。
"""
struct FermionicGaussianState <: AbstractFermionicGaussianState
    Γ :: Matrix{Float64}
    function FermionicGaussianState(Γ::Matrix{Float64})
        m, n = size(Γ)
        m == n || throw(ArgumentError("费米协方差必须是方阵"))
        iseven(m) || throw(ArgumentError("费米协方差维数必须是偶数"))
        isapprox(Γ, -transpose(Γ); atol=1e-10, rtol=1e-10) ||
            throw(ArgumentError("费米 Majorana 协方差必须实反对称"))
        ν = abs.(eigvals(im .* Γ))
        maximum(ν; init=0.0) ≤ 1 + 1e-8 ||
            throw(ArgumentError("费米协方差不是物理的：|ν|max = $(maximum(ν)) > 1"))
        return new(Γ)
    end
end
FermionicGaussianState(Γ::AbstractMatrix) = FermionicGaussianState(convert(Matrix{Float64}, Γ))

"""
    BosonicGaussianState(Σ)

玻色高斯态：只存实 (x,p) 协方差 `Σ`（2N×2N 对称正定，
`Σ_ab = ½⟨{R_a,R_b}⟩`）。基矢 `R = (x₁,p₁,…)`，`xₘ = (aₘ+aₘ†)/√2`、
`pₘ = i(aₘ†−aₘ)/√2`，模式 `m` 占相邻指标 `[2m−1, 2m]`。构造器校验：
实对称、正定（不可归一化的 Choi "态"会给负定协方差）、辛本征值
`ν ≥ ½`。表象与算子 block 的 Nambu 基不同（见 `bdg`）。
通常由 `purify` / `partialtr` / `GaussianState(Gibbs)` 产生。
"""
struct BosonicGaussianState <: AbstractBosonicGaussianState
    Σ :: Matrix{Float64}
    function BosonicGaussianState(Σ::Matrix{Float64})
        m, n = size(Σ)
        m == n || throw(ArgumentError("玻色协方差必须是方阵"))
        iseven(m) || throw(ArgumentError("玻色协方差维数必须是偶数"))
        isapprox(Σ, transpose(Σ); atol=1e-10, rtol=1e-10) ||
            throw(ArgumentError("玻色协方差必须实对称"))
        isposdef(Symmetric(Σ)) ||
            throw(ArgumentError("玻色协方差不是物理的：不正定（不可归一化？）"))
        N = m ÷ 2
        ν = abs.(eigvals(_omega(N) * Σ))
        # 容差 1e-6：边界情形（如 Choi 态趋近 |I⟩_G 时 ν→½）的补全数值误差 ~1e-8，
        # 真实的非物理态（热态温度为负等）违例远大于此
        minimum(ν; init=Inf) ≥ 0.5 - 1e-6 ||
            throw(ArgumentError("玻色协方差不是物理的：νmin = $(minimum(ν)) < 1/2"))
        return new(Σ)
    end
end
BosonicGaussianState(Σ::AbstractMatrix) = BosonicGaussianState(convert(Matrix{Float64}, Σ))

"""
    covariance(s::AbstractGaussianState) -> Matrix{Float64}

态的实协方差（**Majorana / (x,p) 表象**，与算子 block 的 Nambu 表象
不同，见 `bdg`）：

- 费米 `Γ_ab = i⟨γ_aγ_b⟩`：Majorana 基 `γ₂ₘ₋₁ = cₘ + cₘ†`、
  `γ₂ₘ = i(cₘ† − cₘ)`，模式 `m` 占相邻指标 `[2m−1, 2m]`；实反对称。
- 玻色 `Σ_ab = ½⟨{R_a,R_b}⟩`：四极基 `R = (x₁,p₁,…)`，
  `xₘ = (aₘ+aₘ†)/√2`、`pₘ = i(aₘ†−aₘ)/√2`，模式 `m` 占相邻指标
  `[2m−1, 2m]`；实对称正定。

物理模式 `p` 对应行/列指标 `[2p−1, 2p]`。
"""
covariance(s::FermionicGaussianState) = s.Γ
covariance(s::BosonicGaussianState)   = s.Σ

"是否纯态：费米 `Γ² = −I`；玻色辛本征值全 `½`。"
ispure(s::FermionicGaussianState) = isapprox(s.Γ * s.Γ, -Matrix{Float64}(I, size(s.Γ)...); atol=1e-8, rtol=1e-8)
function ispure(s::BosonicGaussianState)
    N = nmodes(s)
    ν = abs.(eigvals(_omega(N) * s.Σ))
    return all(isapprox.(ν, 0.5; atol=1e-8))
end

# 物理模式 p 的协方差行指标（Majorana / xp 都是每个模式 2 个）
_modeindices(modes) = reduce(vcat, [[2p - 1, 2p] for p in modes]; init=Int[])
