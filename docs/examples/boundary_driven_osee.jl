# ============================================================================
# boundary_driven_osee.jl —— 边界驱动费米链：快速协方差 OSEE vs 稠密 OSEE
#
# 系统：n 站 tight-binding 链（随机在位 + hopping 1），两端零温损耗
#   Lindblad 驱动：L₁ = √γ₁ c₁、Lₙ = √γₙ cₙ（粒子从两端漏出、经 hopping
#   输运——经典 boundary-driven 输运设置）。线性 Lindblad + 二次 H 使
#   𝓛† 在 Majorana 双线性上严格闭合，且整条轨迹保持在 e^{quadratic}
#   高斯族内（|ρ−Gaussian(Γ)| 列 ~1e-15 作全时监测），协方差完全刻画。
# 初态：混合直积态 ⊗ᵢ[fᵢ|1⟩⟨1| + (1−fᵢ)|0⟩⟨0|]（数守恒 ⟹ ⟨cc⟩=0 全程）。
#
# 快路径（GaussianStates 快速算法，全程 O(n³)）：
#   1) 𝓛† 在 Majorana 双线性上严格闭合（系数用稠密矩阵一次性提取，
#      闭合残差 ~1e-17，无需手推公式）；仿射 EOM 的增广矩阵指数给精确
#      Γ(t)。
#   2) 由 Γ(t) 重建模算符 H(t)：直接 Majorana 公式取 G = ⟨c†c⟩，自然
#      轨道 f_k ⟹ ε_k = ln(f/(1−f))，h = W̄εWᵀ（复厄米；注意 ⟨c†c⟩ =
#      f(h)ᵀ 的转置约定，实对称时不显形）。
#   3) OSEE = entropy(purify(exp(H)), 左半模 [1..m] ∪ [n+1..n+m])。
# 稠密路径：𝓛 超算符矩阵指数演化 ρ(t)；态级验证 ρ(t) == Gaussian(Γ(t))；
#   OSEE = |ρ⟩⟩ = (ρ⊗I)|I⟩_G（graded）经 FSWAP 网络重排到站点交错序
#   (c₁d₁c₂d₂…)，站点割 = 连续前缀位，对右半偏迹、约化密度矩阵本征值
#   熵。费米模重排带 (−1)^{n₁n₂} 符号，不能朴素换轴。
#
# 运行：cd docs/examples && julia --project=../.. boundary_driven_osee.jl
# ============================================================================
using LinearAlgebra, Random, Printf
let root = joinpath(@__DIR__, "..", "..")
    push!(LOAD_PATH, root)
end
using GaussianStates
const GS = GaussianStates

function main()
    # ---------- 1. 系统 ----------
    n = 6                       # 链长
    mcut = n ÷ 2                # 中间二分：左 = 站点 1..mcut
    rng = MersenneTwister(7)
    hmat = Matrix(SymTridiagonal(randn(rng, n), fill(1.0, n - 1)))
    γg, γl = 1.2, 0.9           # 两端损耗速率

    c  = [GS._jw_annihilator(n, i) for i in 1:n]
    cd = adjoint.(c)
    γop = Vector{Matrix{ComplexF64}}(undef, 2n)
    for i in 1:n
        γop[2i-1] = c[i] + cd[i]
        γop[2i]   = im .* (cd[i] - c[i])
    end
    H = sum(hmat[i, j] .* (cd[i] * c[j]) for i in 1:n, j in 1:n)
    Ls = [sqrt(γg) .* c[1], sqrt(γl) .* c[n]]
    dim = 1 << n
    Id = Matrix{ComplexF64}(I, dim, dim)

    𝓛adj(O) = im * (H * O - O * H) +
               sum(L' * O * L - (O * L' * L + L' * L * O) / 2 for L in Ls)

    # ---------- 2. 双线性 EOM：𝓛†(γ_aγ_b) = m0 + Σ κ γ_cγ_d 数值提取 ----------
    idx = [(a, b) for a in 1:2n, b in 1:2n if a < b]
    Nb = length(idx)
    K = zeros(ComplexF64, Nb, Nb)
    m = zeros(ComplexF64, Nb)
    closu = 0.0
    for (k, (a, b)) in enumerate(idx)
        Lag = 𝓛adj(γop[a] * γop[b])
        m[k] = im * tr(Lag) / dim             # dΓ_ab = i·(m0 + Σκ⟨γγ⟩)
        recon = (tr(Lag) / dim) * Id
        for (l, (cc_, dd)) in enumerate(idx)
            κ = -tr(Lag * (γop[cc_] * γop[dd])) / dim
            K[k, l] = κ
            recon += κ * (γop[cc_] * γop[dd])
        end
        closu = max(closu, maximum(abs.(Lag - recon)) / dim)
    end
    @printf("双线性闭合残差 = %.1e（应 ~1e-16）\n", closu)
    A = [K reshape(m, Nb, 1); zeros(ComplexF64, 1, Nb) 0]   # 增广仿射生成元（d/dt 1 = 0）

    # ---------- 3. 初态与稠密演化 ----------
    f0 = [0.85, 0.25, 0.7, 0.3, 0.6, 0.2]    # 混合直积态
    Γ0 = zeros(Float64, 2n, 2n)
    for i in 1:n
        Γ0[2i-1, 2i] = 2 * f0[i] - 1
        Γ0[2i, 2i-1] = -(2 * f0[i] - 1)
    end
    ρ0 = zeros(ComplexF64, dim, dim)
    for x in 0:dim-1
        p = 1.0
        for i in 1:n
            ni = (x >> (i - 1)) & 1
            p *= ni == 1 ? f0[i] : 1 - f0[i]
        end
        ρ0[x+1, x+1] = p
    end
    ρ0 ./= tr(ρ0)

    𝓛vec = -im * (kron(Id, H) - kron(transpose(H), Id))
    for L in Ls
        𝓛vec += kron(conj(L), L) - kron(Id, L' * L) / 2 - kron(transpose(L' * L), Id) / 2
    end

    evolve_fast(t) = begin
        w = t == 0 ? vcat(complex.([Γ0[a, b] for (a, b) in idx]), [1.0 + 0im]) :
                     exp(A * t) * vcat(complex.([Γ0[a, b] for (a, b) in idx]), [1.0 + 0im])
        G = zeros(ComplexF64, 2n, 2n)
        for (k, (a, b)) in enumerate(idx)
            G[a, b] = w[k]; G[b, a] = -w[k]
        end
        real.((G .- G') ./ 2)
    end
    function evolve_dense(t)
        ρ = reshape(exp(𝓛vec * t) * vec(ρ0), dim, dim)
        return ρ ./ tr(ρ)
    end
    function Γ_dense(ρ)
        G = [-imag(tr(ρ * (γop[a] * γop[b]))) for a in 1:2n, b in 1:2n]
        return (G .- G') ./ 2
    end

    # ---------- 4. 模算符重建（数守恒子族：⟨cc⟩=0 ⟹ A=0） ----------
    function modular_op(Γ)
        # 直接 Majorana 公式：⟨c†ᵢcⱼ⟩ = ¼[2δᵢⱼ − i(Γ₂ᵢ₋₁,₂ⱼ₋₁ + Γ₂ᵢ,₂ⱼ)
        #                                    + Γ₂ᵢ₋₁,₂ⱼ + Γ₂ⱼ₋₁,₂ᵢ]
        G = [(2*(i == j) - im*(Γ[2i-1, 2j-1] + Γ[2i, 2j])
              + Γ[2i-1, 2j] + Γ[2j-1, 2i]) / 4 for i in 1:n, j in 1:n]
        F = eigen(Hermitian(G))                        # 自然轨道
        fk = clamp.(F.values, 1e-13, 1 - 1e-13)
        ε = log.(fk ./ (1 .- fk))
        # 复情形 ⟨c†c⟩ = f(h)ᵀ（转置！实对称时不可见）⟹ 逆变换用 W̄ ε Wᵀ
        h = conj(F.vectors) * Diagonal(ε) * transpose(F.vectors)
        return GS.FermionicQuadraticHermitianOperator(h, zeros(ComplexF64, n, n))
    end

    # ---------- 5. OSEE 两条路径 ----------
    function osee_fast(Γ)
        Hm = modular_op(Γ)
        psi = purify(exp(GS.FermionicQuadraticOperator(Hm.h, Hm.A, Hm.B)))
        modes = vcat(1:mcut, n .+ (1:mcut))            # 左 ket + 左 bra 模
        return entropy(psi, collect(modes))
    end
    # 相邻位 (k,k+1)（0 基）的费米换位 FSWAP：符号 (−1)^{n₁n₂}
    function fswap!(ψ, k)
        out = similar(ψ)
        for i in 0:length(ψ)-1
            b1 = (i >> k) & 1; b2 = (i >> (k + 1)) & 1
            src = (i & ~(1 << k) & ~(1 << (k + 1))) | (b1 << (k + 1)) | (b2 << k)
            out[src + 1] = (b1 * b2 == 1 ? -1 : 1) * ψ[i + 1]
        end
        ψ .= out
    end
    function osee_dense(ρ)
        # |ρ⟩⟩ = (ρ⊗I)|I⟩_G（_dense_vectorize 的验证约定，模序 c₁..cₙd₁..dₙ），
        # 再用 FSWAP 网络重排到站点交错序 (c₁d₁c₂d₂…)——费米模重排带 JW 符号，
        # 不能朴素换轴。站点割在前 2·mcut 位 = 连续前缀，偏迹 == 模式熵。
        ψ = copy(GS._dense_vectorize(ρ, n))
        for i in 1:n
            pos = n + i - 1
            while pos > 2(i - 1)
                fswap!(ψ, pos - 1)
                pos -= 1
            end
        end
        ψ ./= norm(ψ)
        Aψ = reshape(ψ, 1 << (2mcut), :)               # 左 2mcut 位（站点 1..mcut）vs 右
        ρL = Aψ * Aψ'
        p = real.(eigvals(Hermitian(ρL)))
        p = p[p .> 1e-14]; p ./= sum(p)
        return -sum(p .* log.(p))
    end

    # ---------- 6. 对拍 ----------
    println("n=$n 链，割在 $mcut | $(n - mcut)；γ₁=$γg, γₙ=$γl")
    @printf("%-6s %-10s %-10s %-10s %-12s %-12s\n",
            "t", "S_fast", "S_dense", "|ΔS|", "|ΔΓ|", "|ρ−Gauss(Γ)|")
    maxdS = 0.0; maxdΓ = 0.0
    for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        Γf = evolve_fast(t)
        ρ = evolve_dense(t)
        Γd = Γ_dense(ρ)
        dΓ = maximum(abs.(Γf .- Γd))
        Hm = modular_op(Γf)
        ρg = fockmatrix(exp(Hm)); ρg ./= tr(ρg)
        dρ = maximum(abs.(ρ .- ρg))
        Sf = osee_fast(Γf)
        Sd = osee_dense(ρ)
        maxdS = max(maxdS, abs(Sf - Sd)); maxdΓ = max(maxdΓ, dΓ)
        @printf("%-6.2f %-10.6f %-10.6f %-10.2e %-12.2e %-12.2e\n",
                t, Sf, Sd, abs(Sf - Sd), dΓ, dρ)
    end
    @printf("\nmax |ΔOSEE| = %.2e, max |ΔΓ| = %.2e  →  %s\n", maxdS, maxdΓ,
            maxdS < 1e-8 && maxdΓ < 1e-9 ? "全程对齐 ✓" : "未对齐 ✗")
    maxdS < 1e-8 && maxdΓ < 1e-9 || error("快速与稠密路径未对齐")
end
main()
