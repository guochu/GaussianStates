# ============================================================================
# inverse.jl —— 协方差 → bdg 的 O(n³) 反推
#
# 假定态是某个二次型 Gibbs 态 e^H/tr（满秩；费米 |ν|<1、玻色 ν>1/2）。
# 费米：G=⟨c†c⟩、F=⟨cc⟩ 由 Γ 的直接 Majorana 公式给出，
#   C = [[I−Gᵀ, F],[−F̄, G]]，C⁻¹ − I = e^{X} ⟹ X = log(C⁻¹−I)
#   （符号由 A≠0 的稠密 Fock 仲裁钉死）。
# 玻色：Williamson 路线。模哈密顿量 H_mod = ½RᵀGR，
#   G = S·D2·Sᵀ（D2 = ⊕2artanh(1/2ν_k)·I₂，相位不变），
#   op = (−h_mod, −A_mod)；块提取 h=(Gxx+Gpp)/2 − i(Gxp−Gxpᵀ)/2、
#   A=[(Gxx−Gpp)+i(Gxp+Gxpᵀ)]/2（i≠j 的 a†a† 双计统一除 2）。
# h、A 由 X 的 block 投影（数值上投影到严格 bdg 形）；c₀ 与协方差无关，
# 返回 c₀=0。纯态对数发散，抛 ArgumentError。
# ============================================================================

"""
    quadraticoperator(s::FermionicGaussianState) -> FermionicQuadraticHermitianOperator

反推：给定高斯态的协方差，假定它是某个二次型 Gibbs 态 `e^H/tr`
（满秩），O(n³) 重建 bdg 矩阵的 `(h, A)`。满足
`GaussianState(exp(quadraticoperator(s))) == s` 与精确恢复输入 `(h, A)`
（往返测试固化）。`c₀` 与协方差无关，返回 0；纯态（费米 |ν|=1、
玻色 ν=1/2）不是任何满秩 Gibbs 态的协方差，对数发散，抛 `ArgumentError`。

费米：C = log(C⁻¹−I)（Nambu 对数）。玻色：Williamson 组装
G = S·D2·Sᵀ + 块提取（模哈密顿量取负）。
"""
function quadraticoperator(s::FermionicGaussianState)
    Γ = s.Γ
    n = size(Γ, 1) ÷ 2
    νmax = maximum(abs.(eigvals(Hermitian(im .* Γ))))
    νmax > 1 - 1e-8 && throw(ArgumentError(
        "纯态不是任何满秩 Gibbs 态 e^H/tr 的协方差（|ν| = $(νmax) ≈ 1），对数发散"))
    # ⟨c†ᵢcⱼ⟩、⟨cᵢcⱼ⟩ 的直接 Majorana 公式（⟨γ_aγ_b⟩ = δ_ab − iΓ_ab）
    G = zeros(ComplexF64, n, n)
    F = zeros(ComplexF64, n, n)
    @inbounds for i in 1:n, j in 1:n
        G[i, j] = (2*(i == j) - im*(Γ[2i-1, 2j-1] + Γ[2i, 2j])
                   + Γ[2i-1, 2j] + Γ[2j-1, 2i]) / 4
        F[i, j] = (Γ[2i-1, 2j] - Γ[2j-1, 2i]
                   + im*(Γ[2i, 2j] - Γ[2i-1, 2j-1])) / 4
    end
    C = [Matrix{ComplexF64}(I, n, n) - transpose(G)   F
         -conj(F)                                        G]
    Y = C \ Matrix{ComplexF64}(I, 2n, 2n) - Matrix{ComplexF64}(I, 2n, 2n)
    Fy = eigen(Hermitian(Y))
    minimum(Fy.values) ≤ 0 && throw(ArgumentError(
        "协方差不是满秩费米 Gibbs 态（C⁻¹−I 谱非正）"))
    X = Fy.vectors * Diagonal(log.(Fy.values)) * Fy.vectors'
    X = (X + X') / 2                                # 数值厄米化（保复厄米 h 的虚部）
    h = (X[1:n, 1:n] - transpose(X[n+1:2n, n+1:2n])) / 2
    A = (X[1:n, n+1:2n] - transpose(X[1:n, n+1:2n])) / 2
    return FermionicQuadraticHermitianOperator(h, A)
end

# Williamson 分解：Sᵀ Σ S = diag(ν₁,ν₁,ν₂,ν₂,…)（交错 x,p 序），S 辛。
# （相对本征向量相位不变——G = S·D2·Sᵀ 对逐对本征向量旋转不变，D2 逐对各向同性）
function _williamson(Σ::Matrix{Float64})
    m = size(Σ, 1); n = m ÷ 2
    Ω = _omega(n)
    F = eigen(Ω * Σ)
    S = zeros(m, m)
    νs = Float64[]
    k = 0
    for i in 1:m
        imag(F.values[i]) > 1e-8 || continue
        ν = imag(F.values[i])
        v = F.vectors[:, i]
        a = real.(v); b = imag.(v)
        q = dot(a, Ω * b)
        if q < 0
            a, b = b, -a
            q = -q
        end
        k += 1
        S[:, 2k-1] .= a ./ sqrt(q)
        S[:, 2k] .= b ./ sqrt(q)
        push!(νs, ν)
    end
    k == n || throw(ArgumentError("Williamson 分解失败（协方差非正定？）"))
    return S, νs
end

function quadraticoperator(s::BosonicGaussianState)
    Σ = s.Σ
    n = size(Σ, 1) ÷ 2
    # 纯态（辛本征值 ν→1/2）不可逆
    νs_all = abs.(eigvals(_omega(n) * Σ))
    minimum(νs_all) < 0.5 + 1e-8 && throw(ArgumentError(
        "纯态（ν → 1/2）不是任何满秩 Gibbs 态 e^H/tr 的协方差；对数发散"))
    S, νs = _williamson(Σ)
    ε = 2 .* atanh.(1 ./ (2 .* νs))
    D2 = zeros(2n, 2n)
    for k in 1:n
        D2[2k-1, 2k-1] = ε[k]
        D2[2k, 2k] = ε[k]
    end
    G = S * D2 * transpose(S)             # H_mod = ½RᵀGR（模哈密顿量）
    G = (G + G') ./ 2
    Gxx = G[1:2:2n, 1:2:2n]; Gpp = G[2:2:2n, 2:2:2n]; Gxp = G[1:2:2n, 2:2:2n]
    hmod = (Gxx + Gpp) / 2 - im .* (Gxp - Gxp') / 2
    # 玻色 a†ᵢa†ⱼ (i≠j) 双计 ⟹ 统一除 2（对角公式自动一致）
    Amod = ((Gxx - Gpp) .+ im .* (Gxp + Gxp')) ./ 2
    # 模哈密顿量 = −指数算符：op = (−hmod, −Amod)
    return BosonicQuadraticHermitianOperator(-hmod, -Amod)
end
