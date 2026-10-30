# 费米子：§6.4 性质（纯态熵/互补/c₀ 无关/Rényi）与错误路径
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Fermionic" begin
    @testset "properties (§6.4)" begin
        # 整纯态熵 0（含 Rényi）；S(∅)=0
        op = exp(random_op(2, 7))
        psi = purify(op)
        @test entropy(psi) < 1e-10
        @test entropy(psi; α=2) < 1e-10
        @test entropy(psi, Int[]) == 0.0
        # 纯态互补 S(A) = S(Ā)（费米 2n=4 模）
        for modes in ([1], [2], [1,2], [1,3])
            comp = setdiff(1:4, modes)
            @test isapprox(entropy(psi, modes), entropy(psi, comp); atol=1e-9)
        end
        # c₀ 无关（只给 ρ 乘标量，协方差/熵不变）
        h, A, B = op.op.h, op.op.A, op.op.B
        op2 = exp(FermionicQuadraticOperator(h, A, B; c0=5.0))
        @test maximum(abs.(covariance(purify(op2)) .- covariance(psi))) < 1e-12
        # Rényi α→1 连续、单调 S(α=2) ≤ S(α=1)
        for modes in ([1], [1,3])
            S1 = entropy(psi, modes)
            @test abs(entropy(psi, modes; α=1.000001) - S1) < 1e-4
            @test entropy(psi, modes; α=2) ≤ S1 + 1e-12
        end
        # α=2 与稠密偏迹一致（n=2 Gibbs，模式 1）
        Hg = random_op(2, 3; herm=true)
        G = GaussianState(exp(Hg))
        ρ = fockmatrix(exp(Hg)); ρ ./= tr(ρ)
        ρred = zeros(ComplexF64, 2, 2)
        for n1 in 0:1, m1 in 0:1, n2 in 0:1
            ρred[n1+1, m1+1] += ρ[n1 + 2n2 + 1, m1 + 2n2 + 1]
        end
        p = real.(eigvals(Hermitian(ρred)))
        S2dense = -log(sum(p .^ 2))
        @test isapprox(entropy(G, [1]; α=2), S2dense; atol=1e-10)
    end

    @testset "error paths" begin
        op = exp(random_op(2, 7))
        psi = purify(op)
        # 尺寸/对称性校验
        @test_throws ArgumentError FermionicQuadraticOperator(
            zeros(ComplexF64,2,3), zeros(ComplexF64,2,2), zeros(ComplexF64,2,2))
        @test_throws ArgumentError FermionicQuadraticOperator(rand(3,3))
        @test_throws ArgumentError FermionicQuadraticHermitianOperator(
            rand(2,2) + im .* rand(2,2), zeros(ComplexF64,2,2))    # h 非厄米
        # 非物理协方差
        @test_throws ArgumentError FermionicGaussianState([1.0 0.0; 0.0 1.0])         # 须反对称
        # 越界 / 重复 modes / 非法 α
        @test_throws BoundsError entropy(psi, [5])
        @test_throws BoundsError partialtr(psi, [0])
        @test_throws ArgumentError entropy(psi, [1, 1])
        @test_throws ArgumentError partialtr(psi, [2, 2])
        @test_throws ArgumentError entropy(psi; α=0)
        @test_throws ArgumentError entropy(psi; α=-1)
    end
end
