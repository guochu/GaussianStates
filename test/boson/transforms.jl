# 玻色子：Bogoliubov ↔ (x,p) 协方差变换（稠密 Fock 仲裁 + 往返）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "Bogoliubov ↔ (x,p)" begin
        for (n, seed) in ((1, 3), (2, 5))
            rng = MersenneTwister(seed)
            X = randn(rng, ComplexF64, n, n)
            h = -((X + X') / 2 + 3I)                  # 稳定（冷）算子
            A0 = 0.5 .* randn(rng, ComplexF64, n, n)
            A = (A0 + transpose(A0)) / 2
            H = BosonicQuadraticHermitianOperator(h, A)

            # 往返：精确协方差
            Σ = covariance(GaussianState(exp(H)))
            @test maximum(abs.(GS.to_xp(GS.to_bogoliubov(Σ)) .- Σ)) < 1e-12

            # 稠密 Fock 独立构造两侧（截断极限内）
            d = 30
            rho = fockmatrix(exp(H); d=d); rho ./= tr(rho)
            a = GS._boson_annihilator(n, d)
            ops = [a; adjoint.(a)]                    # Ψ = (a, a†)
            C = zeros(ComplexF64, 2n, 2n)
            for p in 1:2n, q in 1:2n
                C[p, q] = 0.5 * tr(rho * (ops[p] * ops[q]' + ops[q]' * ops[p]))
            end
            x = [(a[i] + a[i]') / sqrt(2) for i in 1:n]
            p = [1im * (a[i]' - a[i]) / sqrt(2) for i in 1:n]
            R = vcat([[x[i], p[i]] for i in 1:n]...)  # R = (x₁,p₁,…)
            Σd = zeros(ComplexF64, 2n, 2n)
            for k in 1:2n, l in 1:2n
                Σd[k, l] = 0.5 * tr(rho * (R[k] * R[l] + R[l] * R[k]))
            end
            @test maximum(abs.(GS.to_xp(C) .- real.(Σd))) < 5e-3
            @test maximum(abs.(GS.to_bogoliubov(real.(Σd)) .- C)) < 5e-3
        end
    end
end
