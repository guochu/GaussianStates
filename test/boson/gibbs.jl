# 玻色子：Gibbs 态（单模解析 + 压缩 vs 稠密）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "Gibbs thermal" begin
        ω = 0.7
        H = BosonicQuadraticHermitianOperator([-ω;;], [0.0;;])
        G = GaussianState(exp(H))
        ν = 0.5 / tanh(ω / 2)
        @test maximum(abs.(covariance(G) - ν * Matrix{Float64}(I, 2, 2))) < 1e-10
        n̄ = 1 / (exp(ω) - 1)
        S = (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
        @test isapprox(entropy(G), S; atol=1e-10)
    end

    @testset "Gibbs squeezed (A≠0) vs dense" begin
        # H = -h0 a†a + ½(λ a†² + λ a²)；稠密 Fock 仲裁（annihilator 已修正）
        d = 40
        aop = zeros(ComplexF64, d + 1, d + 1)
        for k in 1:d; aop[k, k+1] = sqrt(k); end
        ad = adjoint(aop)
        for (h0, λ) in ((3.0, 0.0), (3.0, 0.8), (2.0, 1.4))
            H = BosonicQuadraticHermitianOperator(ComplexF64[-h0;;], ComplexF64[λ;;])
            ρ = fockmatrix(exp(H); d=d); ρ ./= tr(ρ)
            ops = [aop, ad]
            Cd = zeros(ComplexF64, 2, 2)
            for i in 1:2, j in 1:2
                Cd[i,j] = 0.5 * tr(ρ * (ops[i] * ops[j]' + ops[j]' * ops[i]))
            end
            Σd = GS.to_xp(Cd)
            G = GaussianState(exp(H))
            @test maximum(abs.(covariance(G) .- Σd)) < 5e-3   # 截断极限
            @test all(eigvals(Symmetric(covariance(G))) .> 0)  # 正定
        end
        # 不可归一化（h > 0：e^H 无界）与不稳定（A 过大）都应抛 ArgumentError
        @test_throws ArgumentError GaussianState(exp(
            BosonicQuadraticHermitianOperator(ComplexF64[1.0;;], ComplexF64[0.0;;])))
        @test_throws ArgumentError GaussianState(exp(
            BosonicQuadraticHermitianOperator(ComplexF64[-1.0;;], ComplexF64[2.0;;])))
    end
end
