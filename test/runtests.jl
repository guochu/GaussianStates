# 主测试入口：按统计量分成 fermion/ 与 boson/ 两组脚本，每组内再按主题分文件
using Test

@testset "GaussianStates v7" begin
    for f in ("fermion/operators.jl", "fermion/purify.jl", "fermion/gibbs.jl",
              "fermion/properties.jl",
              "boson/gibbs.jl", "boson/purify.jl", "boson/tfd.jl",
              "boson/properties.jl")
        include(f)
    end
end
