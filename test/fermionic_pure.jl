# ---------------- 3. 纯配对分支：反对称 bdg ----------------
@testset "稠密仲裁：纯配对分支（反对称，随机）" begin
    for n in (2, 3)                              # 2n 个模（doubled）
        W = rand_antisym(rng, 2n; scale=0.5)
        s = FermionicGaussianState(W)
        @test maximum(abs, covariance(s) - fermionic_covariance(W)) < 1e-10
        @test maximum(abs, covariance(s)^2 + I) < 1e-10     # 纯态
        for k in 1:n
            modes = collect(1:k)
            @test osee(s, modes) ≈ dense_osee_fermionic(W, modes) atol = 1e-10
        end
        @test osee(s, 1:(2n)) ≈ 0.0 atol = 1e-9   # 纯态整态熵 = 0
    end
end

@testset "纯配对：单模解析 S = h(s²/(1+s²))" begin
    for Δ in (0.1, 0.5, 1.0, 2.0)
        W = [0.0 -Δ; Δ 0.0]                      # 2 个模的纯配对
        s = FermionicGaussianState(W)
        ε = Δ^2 / (1 + Δ^2)
        Sexact = -ε * log(ε) - (1 - ε) * log(1 - ε)
        @test osee(s, [1]) ≈ Sexact atol = 1e-12
        @test osee(s, [2]) ≈ Sexact atol = 1e-12
        @test osee(s, [1]; α=2) ≈ -log(ε^2 + (1 - ε)^2) atol = 1e-12
    end
end

@testset "纯配对：双独立配对可加" begin
    s1, s2 = 0.5, 1.3
    Δ = [s1 0.0; 0.0 s2]
    W = zeros(4, 4); W[3:4, 1:2] .= Δ; W = W - transpose(W)
    st = FermionicGaussianState(W)
    ε1 = s1^2 / (1 + s1^2); ε2 = s2^2 / (1 + s2^2)
    h2(p) = -p * log(p) - (1 - p) * log(1 - p)
    @test osee(st, [1]) ≈ h2(ε1) atol = 1e-12
    @test osee(st, [2]) ≈ h2(ε2) atol = 1e-12
    @test osee(st, [1, 2]) ≈ h2(ε1) + h2(ε2) atol = 1e-12
    @test osee(st, 1:4) ≈ 0.0 atol = 1e-12
    @test_throws BoundsError osee(st, [0])
    @test_throws BoundsError osee(st, [5])
    @test_throws ArgumentError osee(st, [1, 1])
end
