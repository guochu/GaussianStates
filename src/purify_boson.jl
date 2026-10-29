# ============================================================================
# purify_boson.jl —— 玻色 Gibbs 态 / purification
#
# Gibbs 协方差：伴随生成元 𝒜 = [[h, A], [-Ā, -hᵀ]]（[H,Ψ] = -𝒜Ψ，厄米类
#   B=A†）本征值成 ± 对；负支 α 的 Bogoliubov 模热权重 ν = ½coth(-α/2)，
#   C = Σ ν (T₋T₋† + T₊T₊†)（σ₃-正交基，对简并谱按子空间 σ₃-Gram 组装）。
#   失稳/无界由 σ₃-Gram 符号检查与 α<0 要求拒绝（谱全实本身不保证稳定）。
#
# purify：湮灭子 V = E − exp(𝒜)·Pmᵀ（一般 𝒜 = [[h,Aᵀ],[−Bᵀ,−hᵀ]]；
#   镜像 M(aᵢ)=+dᵢ†、M(aᵢ†)=+dᵢ；A、B 复对称 ⟹ 𝒜 ∈ sp ⟹ V 迷向
#   ⟹ Choi 态高斯）；协方差由 kernel(Σ − (i/2)Ω) ∋ rowspace(V) 的线性
#   系统补全，对 n ≥ 1、任意（非）厄米配对成立。n=1 与稠密 Fock Choi
#   对拍至机器精度，n=2 与独立 Williamson 构造逐元素一致。不可归一化
#   时补全结果不正定，抛 ArgumentError。
# ============================================================================

# 𝒜 的本征值按实部分组；返回 [(α, V)]，V 列张成该本征值子空间
function _eig_groups(𝒜::AbstractMatrix; tol=1e-8)
    F = eigen(𝒜)
    scale = max(1.0, opnorm(𝒜, 1))
    maximum(abs.(imag.(F.values))) < 1e-9 * scale ||
        throw(ArgumentError("玻色 Gibbs 态不存在：伴随生成元谱非实（H 无下界/不稳定）"))
    vals = real.(F.values)
    order = sortperm(vals)
    groups = Tuple{Float64,Matrix{ComplexF64}}[]
    i = 1
    m = length(vals)
    while i ≤ m
        j = i
        while j < m && abs(vals[order[j+1]] - vals[order[i]]) < tol * scale
            j += 1
        end
        α = sum(vals[order[i:j]]) / (j - i + 1)
        V = F.vectors[:, order[i:j]]
        push!(groups, (α, V))
        i = j + 1
    end
    return groups
end

function _boson_gibbs_C(h::AbstractMatrix, A::AbstractMatrix)
    n = size(h, 1)
    𝒜 = [h A; -conj(A) -transpose(h)]
    σ = [Matrix{ComplexF64}(I, n, n) zeros(ComplexF64, n, n);
         zeros(ComplexF64, n, n) -Matrix{ComplexF64}(I, n, n)]
    groups = _eig_groups(𝒜)
    length(groups) == 2n || throw(ArgumentError("玻色 Gibbs 态：本征值分组异常"))
    C = zeros(ComplexF64, 2n, 2n)
    used = falses(length(groups))
    for (gi, (α, V)) in enumerate(groups)
        used[gi] && continue
        α < -1e-10 * max(1.0, opnorm(𝒜, 1)) ||
            throw(ArgumentError(α > 1e-10 * max(1.0, opnorm(𝒜, 1)) ?
                "玻色 Gibbs 态不可归一化（𝒜 有正本征值 $(α)：e^H 无界；热态应传 H = -β·H_sys）" :
                "玻色 Gibbs 态存在零模（𝒜 本征值 0），协方差发散；请加质量/截断"))
        gj = findfirst(k -> !used[k] && abs(groups[k][1] + α) < 1e-8 * max(1.0, opnorm(𝒜, 1)), 1:length(groups))
        gj === nothing && throw(ArgumentError("玻色 Gibbs 态：本征值 ± 配对失败"))
        used[gj] = used[gi] = true
        β, Ṽ = groups[gj]
        # σ₃-Gram：负支子空间须正定、正支子空间须负定（失稳/无界时翻转）
        G = V' * σ * V
        maximum(real.(eigvals(Hermitian(G)))) > 1e-10 ||
            throw(ArgumentError("玻色 Gibbs 态不可归一化（σ₃ 符号结构异常：e^H 无界/失稳）"))
        Gb = Ṽ' * σ * Ṽ
        minimum(real.(eigvals(Hermitian(Gb)))) < -1e-10 ||
            throw(ArgumentError("玻色 Gibbs 态不可归一化（σ₃ 符号结构异常：e^H 无界/失稳）"))
        ν = 0.5 / tanh(-α / 2)
        C .+= ν .* (V * (Hermitian(G) \ V') + Ṽ * ((-Hermitian(Gb)) \ Ṽ'))
    end
    return (C .+ C') ./ 2
end

"""
    GaussianState(gs::BosonicQuadraticGibbsState) -> BosonicGaussianState

厄米 `e^H` 的 Gibbs 态（n 模，混合）。Bogoliubov 组装协方差，Σ = to_xp(C)。
"""
function GaussianState(gs::BosonicQuadraticGibbsState)
    return BosonicGaussianState(to_xp(_boson_gibbs_C(gs.op.h, gs.op.A)))
end

"""
    purify(e::BosonicQuadraticOperatorExp) -> BosonicGaussianState

一般（非厄米）`e^H` 的算子矢量化/Choi 态 `(ρ⊗I)|I⟩`，纯态，2n 模。
湮灭子 V = E − exp(𝒜)·Pmᵀ（𝒜 = [[h,Aᵀ],[−Bᵀ,−hᵀ]] 为 [H,Ψ]=𝒜Ψ 的生成元，
镜像 M(aᵢ)=+dᵢ†、M(aᵢ†)=+dᵢ；A、B 复对称 ⟹ V 迷向 ⟹ Choi 态高斯，
对 n ≥ 1、任意配对成立）；协方差由 kernel(Σ−(i/2)Ω)∋rowspace(V) 的
线性系统补全。不可归一化（如谱含增长方向）时补全结果不正定，
抛 ArgumentError。
"""
function purify(e::BosonicQuadraticOperatorExp)
    h, A, B = e.op.h, e.op.A, e.op.B
    n = size(h, 1)
    𝒜 = [h transpose(A); -transpose(B) -transpose(h)]
    W = exp(𝒜)
    Pm = zeros(ComplexF64, 4n, 2n)
    for i in 1:n
        Pm[3n+i, i] = 1.0            # M(aᵢ)  = +dᵢ†
        Pm[2n+i, n+i] = 1.0          # M(aᵢ†) = +dᵢ
    end
    E = zeros(ComplexF64, 2n, 4n)
    E[1:2n, 1:2n] = Matrix{ComplexF64}(I, 2n, 2n)
    V = E - W * transpose(Pm)
    # Nambu 系数 → 四极协变式（交错 x,p；ket 模 1..n 前、bra 模 n+1..2n 后）
    Wq = zeros(ComplexF64, 2n, 4n)
    for j in 1:n, r in 1:2n
        Wq[r, 2j-1]     = (V[r, j]   + V[r, n+j])   / sqrt(2)
        Wq[r, 2j]       = im * (V[r, j]   - V[r, n+j])   / sqrt(2)
        Wq[r, 2n+2j-1]  = (V[r, 2n+j] + V[r, 3n+j]) / sqrt(2)
        Wq[r, 2n+2j]    = im * (V[r, 2n+j] - V[r, 3n+j]) / sqrt(2)
    end
    Ω = _omega(2n)
    G = [Wq; conj(Wq)]
    rhs = [(im / 2) * Wq * Ω; -(im / 2) * conj(Wq) * Ω]
    Σt = try
        G \ rhs
    catch err
        isa(err, SingularException) && throw(ArgumentError(
            "玻色一般 purify：湮灭子协变式线性相关（Choi 态不可归一化？）"))
        rethrow()
    end
    Σ = real.((Σt .+ Σt') ./ 2)
    return BosonicGaussianState(Σ)
end

"""
    purify(gs::BosonicQuadraticGibbsState) -> BosonicGaussianState

Gibbs 态的 thermofield double（`(ρ^{1/2}⊗I)|I⟩`，纯态，2n 模）。
与费米侧同一路径：`√ρ = e^{H/2}` 经一般 exp 走 Choi 路径。
"""
purify(gs::BosonicQuadraticGibbsState) = begin
    H2 = gs.op / 2
    purify(BosonicQuadraticOperatorExp(BosonicQuadraticOperator(
        H2.h, H2.A, H2.B; c0=H2.c0)))
end
