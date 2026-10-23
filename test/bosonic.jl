# ---------------- 4. 玻色 ----------------
@testset "稠密仲裁：玻色（随机 h,g）" begin
    for trial in 1:2
        nm = 2
        hb = rand_spd(rng, nm; shift=1.0)
        gb = 0.1 .* randn(rng, nm, nm)
        B = bosonic_bdg_matrix(hb, gb)
        s = BosonicGaussianState(B)

        # 子系统熵（偏迹）
        for m in 1:nm
            @test entropy(s, [m]) ≈
                  dense_entropy_bosonic(B, [m]; nbas=48) atol=1e-8
        end
        # OSEE（operator space，doubled 模式指标：site m = [m, nm+m]）：
        # doubled 协方差快速算法 vs 稠密向量化
        for modes in ([1], [2], [1, 2])
            @test osee(s, vcat(modes, nm .+ modes)) ≈
                  dense_osee_bosonic(B, modes; nbas=24) atol=1e-7
        end
        # doubled 协方差是纯态：辛本征值全为 ½
        m = nmodes(s)
        Γd = GaussianStates._gammad(s)
        Ωd = zeros(4m, 4m)
        for k in 1:2m
            Ωd[2k-1, 2k] = 1.0
            Ωd[2k, 2k-1] = -1.0
        end
        ν = abs.(eigvals(Ωd * Γd))
        @test maximum(abs.(ν .- 0.5)) < 1e-9
    end
end

@testset "玻色：bdg = ωI 的热态" begin
    for ω in (0.5, 1.0, 3.0)
        s = BosonicGaussianState(ω * Matrix{Float64}(I, 2, 2))
        @test covariance(s) ≈ 0.5 / tanh(ω / 2) * Matrix{Float64}(I, 2, 2) atol=1e-10
        n̄ = 1 / (exp(ω) - 1)
        Sexact = (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
        @test entropy(s, 1) ≈ Sexact atol = 1e-10   # 单模：整态熵
        @test entropy(s) ≈ Sexact atol = 1e-10
        @test osee(s, [1, 2]) ≈ 0.0 atol = 1e-10    # 单模无 bipartition
    end
end

@testset "玻色：双模热态熵可加 + Rényi" begin
    ω1, ω2 = 0.7, 2.3
    B = zeros(4, 4)
    B[1, 1] = B[2, 2] = ω1
    B[3, 3] = B[4, 4] = ω2
    s = BosonicGaussianState(B)
    n̄(ω) = 1 / (exp(ω) - 1)
    SvN(ω) = (n̄(ω) + 1) * log(n̄(ω) + 1) - n̄(ω) * log(n̄(ω))
    @test entropy(s, 1) ≈ SvN(ω1) atol = 1e-10
    @test entropy(s, 2) ≈ SvN(ω2) atol = 1e-10
    @test entropy(s) ≈ SvN(ω1) + SvN(ω2) atol = 1e-10
    @test entropy(s, 1; α=2) ≈ log(1 + 2 * n̄(ω1)) atol = 1e-10
    @test entropy(s, 1; α=1 + 1e-3) ≈ SvN(ω1) rtol = 1e-3
    # 两模是直积热态 ⇒ operator-space OSEE 为 0
    @test osee(s, [1, 3]) ≈ 0.0 atol=1e-8
end

# ---------------- 4b. 玻色纯配对（IF 系数态，非密度矩阵） ----------------
@testset "玻色纯配对：TMSV 解析 + 稠密仲裁" begin
    # 解析：W = [[0,M],[Mᵀ,0]]，M = m·[[0,1],[1,0]] ⟹ 配对 (c₁,d₂)、(c₂,d₁)
    # 均跨 site ⟹ site 割（模式对 [p, n+p]）= 单模热熵（tanh r = m）
    m = 0.45
    M = m .* [0.0 1.0; 1.0 0.0]
    Wd = [zeros(2, 2) M; transpose(M) zeros(2, 2)]
    s = BosonicGaussianState(Wd)
    @test !s.isbdg
    @test_throws ArgumentError entropy(s)
    @test_throws ArgumentError entropy(s, [1])
    n̄(x) = x^2 / (1 - x^2)
    SvN(x) = (n̄(x) + 1) * log(n̄(x) + 1) - n̄(x) * log(n̄(x))
    # site 割 {c_p,d_p} 同时劈开两条配对（c₁–d₂ 与 c₂–d₁）⟹ 2 × 单模热熵
    @test osee(s, [1, 3]) ≈ 2SvN(m) atol = 1e-12         # site 1 = 模式 {c₁, d₁}
    @test osee(s, [2, 4]) ≈ 2SvN(m) atol = 1e-12         # 互补割同熵（纯向量）
    @test osee(s, collect(1:4)) ≈ 0.0 atol = 1e-12      # 纯向量整体系
    @test osee(s, [1, 3]; α=2) ≈ 2log(1 + 2n̄(m)) atol = 1e-12

    # 随机实对称 W（谱 ⊂ (−1,1)）：协方差与割熵对稠密截断向量仲裁
    n = 2                                               # 2n = 4 个模
    rng = MersenneTwister(20241021)                     # 固定 seed：截断误差不随全局 rng 浮动
    A = randn(rng, 2n, 2n)
    W = (A + A') ./ 2
    W ./= (8 * opnorm(W, 1))                            # 奇异值 ≤ 1/8，nbas=6 截断尾巴 ~1e-13
    s = BosonicGaussianState(W)
    @test !s.isbdg
    Γ = covariance(s)
    nbas = 6
    d = nbas + 1
    ψ = bosonic_pairing_vector(W; nbas=nbas)
    @test maximum(abs, Γ - covariance_from_bosonic_vector(ψ, 2n, d)) < 5e-8
    # 纯态：辛本征值全 ½；Γd === Γ
    Ω = GaussianStates._omega(2n)
    ν = abs.(eigvals(Ω * Γ))
    @test maximum(abs.(ν .- 0.5)) < 1e-10
    @test GaussianStates._gammad(s) === Γ
    # 割熵 vs 稠密 SVD（残差 ~ 截断尾巴）
    for sites in ([1, 3], [2, 4], [1, 2, 3, 4])
        @test osee(s, sites) ≈
              dense_osee_bosonic_vector(ψ, 2n, d, sites) atol = 1e-8
    end
    # 互补对称（纯向量）
    @test osee(s, [1, 3]) ≈ osee(s, [2, 4]) atol = 1e-10
    # densitymatrix：截断 |Ψ⟩⟨Ψ| 的纯度
    ρp = densitymatrix(s; d=5)
    @test tr(ρp) ≈ 1 atol=1e-12
    @test maximum(abs, eigvals(Hermitian((ρp + ρp') ./ 2))) ≥ 1 - 1e-6
end

@testset "玻色纯配对：复对称 W（一般非 BdG 输入）" begin
    # 复对称 W（实时/混合时间 IF 的系数态）；4 个 doubled 模，nbas 取小以免稠密
    # Fock 空间（d⁴）膨胀
    N = 4
    rngc = MersenneTwister(20241023)
    A = randn(rngc, ComplexF64, N, N)
    W = (A + transpose(A)) ./ 2
    W ./= (40 * opnorm(W, 2))                   # 最大奇异值 ≪ 1，截断尾巴极小
    s = BosonicGaussianState(W)
    @test !s.isbdg
    @test_throws ArgumentError entropy(s)

    Γ = covariance(s)
    @test eltype(Γ) == Float64
    @test maximum(abs.(Γ - transpose(Γ))) < 1e-12
    # 纯态：辛本征值全 ½；Γd === Γ
    Ω = GaussianStates._omega(N)
    ν = abs.(eigvals(Ω * Γ))
    @test maximum(abs.(ν .- 0.5)) < 1e-8
    @test GaussianStates._gammad(s) === Γ

    nbas = 5
    d = nbas + 1
    ψ = bosonic_pairing_vector(W; nbas=nbas)
    @test maximum(abs.(Γ - covariance_from_bosonic_vector(ψ, N, d))) < 5e-8

    # 割熵 vs 稠密 SVD（复 W 的 Γ 含 σ_xp = Im K，正确性由稠密态仲裁）
    for sites in ([1, 3], [2, 4], [1, 2, 3, 4])
        @test osee(s, sites) ≈
              dense_osee_bosonic_vector(ψ, N, d, sites) atol=1e-7
    end
    @test osee(s, [1, 3]) ≈ osee(s, [2, 4]) atol=1e-8

    # densitymatrix（复 W 分支）与独立截断向量对拍
    ρp = densitymatrix(s; d=5)
    ψ4 = bosonic_pairing_vector(W; nbas=4)
    @test ρp ≈ ψ4 * ψ4' atol=1e-12
    @test tr(ρp) ≈ 1 atol=1e-12

    # 非法输入：非对称复矩阵 / 最大奇异值 ≥ 1
    @test_throws ArgumentError BosonicGaussianState(A)
    @test_throws ArgumentError BosonicGaussianState(W .* (1.5 / maximum(svdvals(W))))
end
