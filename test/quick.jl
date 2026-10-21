# 快速冒烟：只跑本次改动相关的测试集（构造校验 + 玻色全部 + entropy 接口）
push!(LOAD_PATH, dirname(@__DIR__))
using GaussianStates
using GaussianStates: covariance
using LinearAlgebra, Random, Test
include("helpers.jl")
const rng = MersenneTwister(20240915)

@testset "quick" begin
    include("bdg_assembly.jl")
    include("bosonic.jl")
    include("entropy.jl")
end
