# 玻色子：§6.4 性质（互补/α=2 解析）与错误路径
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "properties (§6.4)" begin
        # 纯态互补 S(A) = S(Ā)；整态熵 0
        Hb = BosonicQuadraticHermitianOperator(
            ComplexF64[-1.5 0.2; 0.2 -2.1], ComplexF64[0.1 -0.05; -0.05 0.08])
        psib = purify(exp(Hb))
        @test entropy(psib) < 1e-10
        for modes in ([1], [1,3])
            comp = setdiff(1:4, modes)
            @test isapprox(entropy(psib, modes), entropy(psib, comp); atol=1e-9)
        end
        # α=2 解析：单模热态 p_n = (1-t)t^n ⟹ S₂ = log((1+t)/(1-t))
        ω = 0.8
        Gb = GaussianState(exp(BosonicQuadraticHermitianOperator(
            ComplexF64[-ω;;], ComplexF64[0.0;;])))
        t = exp(-ω)
        @test isapprox(entropy(Gb; α=2), log((1+t)/(1-t)); atol=1e-10)
    end

    @testset "error paths" begin
        # 厄米性 / c0 实数校验
        @test_throws ArgumentError BosonicQuadraticHermitianOperator(
            rand(2,2) + im .* rand(2,2), zeros(ComplexF64,2,2); c0=im)
        # 非物理协方差
        @test_throws ArgumentError BosonicGaussianState(ones(2, 2))                   # ν < 1/2
        @test_throws ArgumentError BosonicGaussianState(-Matrix{Float64}(I,2,2))       # 不正定
        # 越界 / 重复 modes / 非法 α
        psi = purify(exp(BosonicQuadraticOperator(ComplexF64[-2.0+0.5im;;],
                 ComplexF64[0.15;;], ComplexF64[-0.1;;])))
        @test_throws BoundsError entropy(psi, [5])
        @test_throws BoundsError partialtr(psi, [0])
        @test_throws ArgumentError entropy(psi, [1, 1])
        @test_throws ArgumentError partialtr(psi, [2, 2])
        @test_throws ArgumentError entropy(psi; α=0)
        @test_throws ArgumentError entropy(psi; α=-1)
    end
end
