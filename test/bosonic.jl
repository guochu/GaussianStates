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
