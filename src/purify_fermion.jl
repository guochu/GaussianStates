# ============================================================================
# purify_fermion.jl —— 费米 purifications
#
# 一般非厄米 e^H：Choi/IF 态 |ρ⟩⟩ = (ρ⊗I)|I⟩_G，|I⟩_G = exp(Σ c†d†)|0⟩。
#   O(n³) 解析构造：e^{-H}Ψe^{H} 的伴随作用矩阵是 W = exp(bdg(op))；
#   |I⟩_G 的镜像恒等式 cᵢ|I⟩=dᵢ†|I⟩、cᵢ†|I⟩=−dᵢ|I⟩ 给出镜像 Pm；
#   湮灭子 V = E − W·Pmᵀ（2n×4n），协方差 = 湮灭子行空间投影子的余
#   （4n×4n）。与稠密零空间版逐元素一致（n=1..4 验证到机器精度）。
# 厄米 e^H：Gibbs 态 C = (I + e^X)^{-1}；TFD = purify(一般 exp 的 e^{H/2})
# ============================================================================

function _mirror_matrix(n::Int)
    # EPR 镜像 M：cᵢ ↦ dᵢ†，cᵢ† ↦ −dᵢ。
    # Pm 的第 k 列 = M(Ψ_k) 在 (c, c†, d, d†) 序下的坐标（2n 个 Nambu 元 → 4n）。
    Pm = zeros(ComplexF64, 4n, 2n)
    for i in 1:n
        Pm[3n+i, i] = 1.0        # M(cᵢ)  = +dᵢ†
        Pm[2n+i, n+i] = -1.0     # M(cᵢ†) = −dᵢ
    end
    return Pm
end

# (c,c†,d,d†) → (c,d,c†,d†) 的重排（to_majorana 的输入序）
function _nambu_reorder(n::Int)
    Po = zeros(ComplexF64, 4n, 4n)
    for i in 1:n
        Po[i, i] = 1.0
        Po[n+i, 2n+i] = 1.0
        Po[2n+i, n+i] = 1.0
        Po[3n+i, 3n+i] = 1.0
    end
    return Po
end

"""
    purify(op::FermionicQuadraticOperatorExp) -> FermionicGaussianState

一般（非厄米）`e^H` 的算子矢量化/Choi 态，纯态，2n 模。
湮灭子 V = E − exp(bdg)·Pmᵀ；协方差为 V 行空间投影子的余，O(n³)。
"""
function purify(e::FermionicQuadraticOperatorExp)
    n = nmodes(e.op)
    W = exp(bdg(e.op))
    Pm = _mirror_matrix(n)
    E = zeros(ComplexF64, 2n, 4n)
    E[1:2n, 1:2n] = Matrix{ComplexF64}(I, 2n, 2n)
    V = E - W * transpose(Pm)
    # 湮灭子行空间投影子：用 QR 正交基构造，避免 (VV†)^{-1} 的平方条件数
    B = Matrix(qr(adjoint(V)).Q)         # 4n×2n，B†B = I，列空间 = 行空间(V)
    C = B * adjoint(B)
    Po = _nambu_reorder(n)
    Γ = to_majorana(Po * C * adjoint(Po))
    return FermionicGaussianState(Γ)
end

"""
    GaussianState(gs::FermionicQuadraticGibbsState) -> FermionicGaussianState

厄米 `e^H` 的 Gibbs 态（n 模，混合）。
"""
function GaussianState(gs::FermionicQuadraticGibbsState)
    X = bdg(gs.op)
    F = eigen(Hermitian(X))
    C = F.vectors * Diagonal(1 ./ (1 .+ exp.(F.values))) * adjoint(F.vectors)
    return FermionicGaussianState(to_majorana(C))
end

"""
    purify(gs::AbstractQuadraticGibbsState) -> FermionicGaussianState

Gibbs 态的 thermofield double（`√ρ=e^{H/2}` 的 Choi 态），纯态，2n 模。
"""
purify(gs::FermionicQuadraticGibbsState) = begin
    H2 = gs.op / 2                   # √ρ = e^{H/2}，经一般 exp 走 Choi 路径
    purify(FermionicQuadraticOperatorExp(FermionicQuadraticOperator(
        H2.h, H2.A, H2.B; c0=H2.c0)))
end
