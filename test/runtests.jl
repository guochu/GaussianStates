using Test

@testset "GaussianStates v7" begin
    include("fermion/runtests.jl")
    include("boson/runtests.jl")
end
