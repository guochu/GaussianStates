# ============================================================================
# inverse.jl —— 协方差 → bdg 的 O(n³) 反推
#
# 假定态是某个二次型 Gibbs 态 e^H/tr（满秩；费米 |ν|<1、玻色 ν>1/2）。
# 费米：G=⟨c†c⟩、F=⟨cc⟩ 由 Γ 的直接 Majorana 公式给出，
#   C = [[I−Gᵀ, F],[−F̄, G]]，C⁻¹ − I = e^{X} ⟹ X = log(C⁻¹−I)
#   （符号由 A≠0 的稠密 Fock 仲裁钉死）。
# 玻色：三条路线（Nambu 对数 / Williamson 组装 / 矩阵函数）均不封闭，
#   暂不实现（见函数内注释）。
# h、A 由 X 的 block 投影（数值上投影到严格 bdg 形）；c₀ 与协方差无关，
# 返回 c₀=0。纯态对数发散，抛 ArgumentError。
# ============================================================================

"""
    quadraticoperator(s::FermionicGaussianState) -> FermionicQuadraticHermitianOperator

反推：给定费米高斯态的协方差，假定它是某个二次型 Gibbs 态 `e^H/tr`
（满秩，|ν|<1），O(n³) 重建 bdg 矩阵的 `(h, A)`。满足
`GaussianState(exp(quadraticoperator(s))) == s` 与精确恢复输入 `(h, A)`
（往返测试固化）。`c₀` 与协方差无关，返回 0；纯态（|ν|=1）不是任何
满秩 Gibbs 态的协方差，对数发散，抛 `ArgumentError`。

玻色版本尚未实现（辛极分解路线待做），调用抛 `ErrorException`。
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

function quadraticoperator(s::BosonicGaussianState)
    throw(ErrorException(
        "玻色 quadraticoperator 尚未实现。尝试过三条路线均不封闭：" *
        "① Nambu C⁻¹−I 的谱双侧跨越（log 无实分支）；" *
        "② Williamson 组装 G = S⁻¹(⊕ε)S⁻ᵀ 对本征向量相位敏感（同一 Σ 不同相位给不同 G）；" *
        "③ 矩阵函数 ΩG = g(ΩΣ) 不成立（[ΩΣ, ΩG] ≠ 0）。" *
        "正确构造需要辛极分解/复结构路线，作为后续工作包。"))
end
