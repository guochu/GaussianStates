# 费米子：算子 block 与反/对称化投影
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Fermionic" begin
    @testset "operator blocks" begin
        for (n, seed) in ((1,1),(2,2),(3,3))
            op = random_op(n, seed)
            @test size(bdg(op)) == (2n, 2n)
            @test op.c0 == 0.3 + 0im
            H = random_op(n, seed; herm=true)
            @test isapprox(bdg(H), bdg(H)'; atol=1e-12)
        end
    end

    @testset "antisymmetrize equivalence" begin
        rng = MersenneTwister(21)
        # 费米：任意 A、B（含对称部分与任意对角元）与反对称化严格等价
        h = randn(rng, ComplexF64, 3, 3)
        A = randn(rng, ComplexF64, 3, 3); B = randn(rng, ComplexF64, 3, 3)
        op1 = FermionicQuadraticOperator(h, A, B; c0=0.4)
        op2 = FermionicQuadraticOperator(h, GS.antisymmetrize(A), GS.antisymmetrize(B); c0=0.4)
        @test fockmatrix(op1) ≈ fockmatrix(op2)                    # Fock 矩阵逐元素
        @test covariance(purify(exp(op1))) ≈ covariance(purify(exp(op2)))
        @test all(diag(op1.A) .== 0) && all(diag(op1.B) .== 0)     # 反对称化 ⟹ 零对角
        # 厄米算子的 A 同样自动投影
        Ah = randn(rng, ComplexF64, 3, 3)
        Hh = FermionicQuadraticHermitianOperator((h + h') / 2, Ah)
        @test Hh.A ≈ GS.antisymmetrize(Ah)
    end

    @testset "hermitian c0 typing" begin
        rng = MersenneTwister(31)
        h = randn(rng, ComplexF64, 3, 3); h = (h + h') / 2
        A = GS.antisymmetrize(randn(rng, ComplexF64, 3, 3))
        # c0 任意数值输入、虚部为 0 即可；存储类型固化为 T
        H = FermionicQuadraticHermitianOperator(h, A; c0=0.7 + 0.0im)
        @test H.c0 === 0.7 + 0.0im                  # T = ComplexF64
        Hr = FermionicQuadraticHermitianOperator(
            [0.5 0.2; 0.2 -0.3], [0.0 0.1; -0.1 0.0]; c0=2)
        @test Hr.c0 === 2.0                          # T = Float64
        @test_throws ArgumentError FermionicQuadraticHermitianOperator(h, A; c0=1 + im)
        # X 矩阵输入同校验
        @test FermionicQuadraticHermitianOperator(bdg(H); c0=0.5 + 0.0im).c0 isa ComplexF64
        @test_throws ArgumentError FermionicQuadraticHermitianOperator(bdg(H); c0=im)
    end
end
