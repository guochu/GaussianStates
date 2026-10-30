# 费米子：一般 e^H 的 Choi 纯化（purify）与偏迹/熵
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Fermionic" begin
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

    @testset "partialtr/entropy" begin
        op = exp(random_op(2, 7))
        psi = purify(op)
        @test entropy(psi) < 1e-10
        for modes in ([1], [2], [1,2], [1,3], [1,2,3,4])
            rho = partialtr(psi, modes)
            @test isapprox(entropy(rho), entropy(psi, modes); atol=1e-10)
        end
    end
end
