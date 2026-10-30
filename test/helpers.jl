# 共享测试工具：稠密仲裁与随机算子（被 fermion/boson 各 include 一次，加守卫）
if !@isdefined(_GS_HELPERS_LOADED)
    const _GS_HELPERS_LOADED = true
using LinearAlgebra, Random
using GaussianStates
const GS = GaussianStates

function dense_vec(rho, n)
    GS._dense_vectorize(rho, n)
end

function cov_from_state(psi, m)
    c = [GS._jw_annihilator(m, i) for i in 1:m]
    cd = [adjoint(x) for x in c]
    g = Vector{Matrix{ComplexF64}}(undef, 2m)
    for i in 1:m
        g[2i-1] = c[i] .+ cd[i]
        g[2i] = 1im .* (cd[i] .- c[i])
    end
    G = zeros(Float64, 2m, 2m)
    for a in 1:2m, b in 1:2m
        G[a,b] = real(1im * dot(g[a]*psi, g[b]*psi))
    end
    return (G .- transpose(G)) ./ 2
end

function cov_from_rho(rho, m)
    c = [GS._jw_annihilator(m, i) for i in 1:m]
    cd = [adjoint(x) for x in c]
    g = Vector{Matrix{ComplexF64}}(undef, 2m)
    for i in 1:m
        g[2i-1] = c[i] .+ cd[i]
        g[2i] = 1im .* (cd[i] .- c[i])
    end
    G = zeros(Float64, 2m, 2m)
    for a in 1:2m, b in 1:2m
        G[a,b] = real(1im * tr(rho * g[a] * g[b]))
    end
    return (G .- transpose(G)) ./ 2
end

function random_op(n, seed; herm=false)
    rng = MersenneTwister(seed)
    if herm
        h = randn(rng, ComplexF64, n, n); h = (h + h')/2
        A = randn(rng, ComplexF64, n, n)
        return FermionicQuadraticHermitianOperator(h, A; c0=0.0)
    end
    h = randn(rng, ComplexF64, n, n)
    A = randn(rng, ComplexF64, n, n); A = (A - transpose(A))/2
    B = randn(rng, ComplexF64, n, n); B = (B - transpose(B))/2
    return FermionicQuadraticOperator(h, A, B; c0=0.3)
end
end # guard
