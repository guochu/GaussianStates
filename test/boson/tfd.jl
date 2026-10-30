# 玻色子：TFD（随机稳定算符 + n=1 纯热态解析）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "TFD" begin
        for (n, seed) in ((1, 3), (2, 5), (3, 9))
            rng = MersenneTwister(seed)
            X = randn(rng, ComplexF64, n, n)
            h = -((X + X') / 2 + 3I)
            A0 = 0.5 .* randn(rng, ComplexF64, n, n)
            A = (A0 + transpose(A0)) / 2
            H = BosonicQuadraticHermitianOperator(h, A)
            gs = exp(H)
            G = GaussianState(gs)
            psi = purify(gs)
            @test ispure(psi)
            red = partialtr(psi, collect(1:n))
            @test maximum(abs.(covariance(red) .- covariance(G))) < 1e-10
            @test entropy(psi) < 1e-10
        end
        # n=1 纯热态解析：TFD = (ρ^{1/2}⊗I)|I⟩ = 双模压缩（x-x 交叉 +g）
        H = BosonicQuadraticHermitianOperator([-0.9;;], [0.0;;])
        psi = purify(exp(H))
        ν = 0.5 / tanh(0.45)
        g = sqrt(ν^2 - 0.25)
        Σref = [ν 0 g 0; 0 ν 0 -g; g 0 ν 0; 0 -g 0 ν]
        @test maximum(abs.(covariance(psi) .- Σref)) < 1e-10
        n̄ = 1 / (exp(0.9) - 1)
        @test isapprox(entropy(partialtr(psi, [1])), (n̄+1)*log(n̄+1) - n̄*log(n̄); atol=1e-10)
    end
end
