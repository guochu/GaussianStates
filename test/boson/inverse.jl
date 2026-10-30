# 玻色子：quadraticoperator 反推尚未实现，断言抛错并注明路线
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "quadraticoperator (未实现)" begin
        s = GaussianState(exp(BosonicQuadraticHermitianOperator(
            ComplexF64[-1.5 0.2; 0.2 -2.1], ComplexF64[0.1 0.0; 0.0 0.1])))
        @test_throws ErrorException quadraticoperator(s)
    end
end
