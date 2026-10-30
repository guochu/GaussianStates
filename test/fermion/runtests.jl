# 费米子测试：算子 block、purify、Gibbs/TFD、偏迹熵、性质、错误路径
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

    @testset "purify vs dense" begin
        for (n, seed) in ((1,1),(2,1),(2,2),(3,1),(3,7),(4,3))
            op = exp(random_op(n, seed))
            psi = purify(op)
            @test ispure(psi)
            rho = fockmatrix(op)
            psid = dense_vec(rho, n)
            Gref = cov_from_state(psid, 2n)
            @test maximum(abs.(covariance(psi) - Gref)) < 1e-10
        end
    end

    @testset "Gibbs state covariance" begin
        for (n, seed) in ((1,1),(2,1),(3,1))
            H = random_op(n, seed; herm=true)
            gs = exp(H)
            G = GaussianState(gs)
            rho = fockmatrix(gs)
            rho = rho / tr(rho)
            Gref = cov_from_rho(rho, n)
            @test maximum(abs.(covariance(G) - Gref)) < 1e-10
        end
    end

    @testset "TFD = partialtr(purify)" begin
        for (n, seed) in ((1,1),(2,1),(3,4))
            H = random_op(n, seed; herm=true)
            gs = exp(H)
            psi = purify(gs)
            @test ispure(psi)
            rho = GaussianState(gs)
            red = partialtr(psi, collect(1:n))
            @test maximum(abs.(covariance(red) - covariance(rho))) < 1e-10
        end
    end

    @testset "partialtr/entropy" begin
        op = exp(random_op(2, 7))
        psi = purify(op)
        @test entropy(psi) < 1e-10
        for modes in ([1], [2], [1,2], [1,3], [1,2,3,4])
            rho = partialtr(psi, modes)
            @test isapprox(entropy(rho), entropy(psi, modes); atol=1e-10)
        end
    end

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
