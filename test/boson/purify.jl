# 玻色子：一般 e^H 的 Choi 纯化（数守恒解析锚 + 配对 vs 稠密 Fock Choi）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "general purify" begin
        # 数守恒解析锚：t = e^h（复），Choi = Σ t^m |m,m⟩ 的双模压缩型协方差
        h0 = -1.0 + 0.3im
        op = exp(BosonicQuadraticOperator(ComplexF64[h0;;],
                 zeros(ComplexF64,1,1), zeros(ComplexF64,1,1)))
        psi = purify(op)
        @test ispure(psi)
        t = exp(h0); d = 1 - abs(t)^2
        ν = 0.5 * (1 + abs(t)^2) / d
        Σref = [real(t)/d imag(t)/d; imag(t)/d -real(t)/d]
        @test maximum(abs.(covariance(psi)[1:2,1:2] .- ν*Matrix{Float64}(I,2,2))) < 1e-12
        @test maximum(abs.(covariance(psi)[1:2,3:4] .- Σref)) < 1e-12
        # 配对 vs 稠密 Fock Choi（含复 h）
        for (h, A, B) in ((ComplexF64[-1.5;;], ComplexF64[0.3;;], ComplexF64[0.3;;]),
                          (ComplexF64[-2.0+0.5im;;], ComplexF64[0.15;;], ComplexF64[-0.1;;]))
            Hop = BosonicQuadraticOperator(h, A, B)
            N = 20
            g = fockmatrix(exp(Hop); d=N)
            Iᵈ = Matrix{ComplexF64}(I, N+1, N+1)
            vecI = sum(kron(v, v) for v in eachcol(Iᵈ))
            ψd = kron(g, Iᵈ) * vecI; ψd ./= norm(ψd)
            a = GS._boson_annihilator(1, N)[1]
            a1 = kron(a, Iᵈ); a2 = kron(Iᵈ, a)
            s2 = 1/sqrt(2)
            ops = [s2*(a1+a1'), s2*im*(a1'-a1), s2*(a2+a2'), s2*im*(a2'-a2)]
            Σd = zeros(Float64, 4, 4)
            for i in 1:4, j in 1:4
                Σd[i,j] = real(dot(ψd, (ops[i]*ops[j] + ops[j]*ops[i])*ψd))/2
            end
            @test maximum(abs.(covariance(purify(exp(Hop))) .- Σd)) < 5e-3
        end
        # 不可归一化（|tanh r| ≥ 1）应抛 ArgumentError
        @test_throws ArgumentError purify(exp(BosonicQuadraticOperator(
            ComplexF64[1.0;;], zeros(ComplexF64,1,1), zeros(ComplexF64,1,1))))
    end

    @testset "symmetrize equivalence" begin
        rng = MersenneTwister(21)
        # 玻色：任意 A、B（含反对称部分）与对称化等价
        hb = -(randn(rng, ComplexF64, 3, 3) + 2I)                  # 稳定（冷）算子
        Ab = (0.2/sqrt(3)) .* randn(rng, ComplexF64, 3, 3)
        Bb = (0.2/sqrt(3)) .* randn(rng, ComplexF64, 3, 3)
        opb1 = BosonicQuadraticOperator(hb, Ab, Bb)
        opb2 = BosonicQuadraticOperator(hb, GS.symmetrize(Ab), GS.symmetrize(Bb))
        @test fockmatrix(opb1; d=10) ≈ fockmatrix(opb2; d=10)
        @test covariance(purify(exp(opb1))) ≈ covariance(purify(exp(opb2)))
    end
end
