# 玻色子：quadraticoperator 反推（协方差 → bdg，Williamson 路线）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "quadraticoperator inverse" begin
        for (n, seed) in ((1, 3), (2, 5), (2, 11), (3, 9), (3, 21))
            rng = MersenneTwister(seed)
            X = randn(rng, ComplexF64, n, n)
            h = -((X + X') / 2 + 3I)               # 稳定（冷）算子
            A0 = 0.5 .* randn(rng, ComplexF64, n, n)
            A = (A0 + transpose(A0)) / 2
            H = BosonicQuadraticHermitianOperator(h, A)
            s = GaussianState(exp(H))
            H2 = quadraticoperator(s)
            @test maximum(abs.(H2.h .- H.h)) < 1e-9    # 直接恢复
            @test maximum(abs.(H2.A .- H.A)) < 1e-9
            s2 = GaussianState(exp(H2))                # 往返
            @test maximum(abs.(covariance(s2) .- covariance(s))) < 1e-9
        end
        # 纯 TFD（ν=1/2）不可逆（1 模 Gibbs 的 TFD = 2 模纯态）
        psi = purify(exp(BosonicQuadraticHermitianOperator(
            ComplexF64[-1.5;;], ComplexF64[0.0;;])))
        @test_throws ArgumentError quadraticoperator(psi)
    end
end
