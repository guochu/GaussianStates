# 费米子：Gibbs 态协方差与 TFD
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Fermionic" begin
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
end
