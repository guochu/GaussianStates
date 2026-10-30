# 费米子：quadraticoperator 反推（协方差 → bdg）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Fermionic" begin
    @testset "quadraticoperator inverse" begin
        for (n, seed) in ((1,5),(2,11),(3,13))
            H = random_op(n, seed; herm=true)      # 含随机 A ≠ 0
            s = GaussianState(exp(H))
            H2 = quadraticoperator(s)
            @test maximum(abs.(H2.h .- H.h)) < 1e-9    # 直接恢复
            @test maximum(abs.(H2.A .- H.A)) < 1e-9
            s2 = GaussianState(exp(H2))                # 往返
            @test maximum(abs.(covariance(s2) .- covariance(s))) < 1e-9
        end
        # 纯态（|ν|=1）不可逆
        psi = purify(exp(random_op(2, 7)))
        @test_throws ArgumentError quadraticoperator(psi)
    end
end
