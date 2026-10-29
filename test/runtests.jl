using Test, LinearAlgebra, Random
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

@testset "GaussianStates v7" begin
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

    @testset "boson Gibbs thermal" begin
        ω = 0.7
        H = BosonicQuadraticHermitianOperator([-ω;;], [0.0;;])
        G = GaussianState(exp(H))
        ν = 0.5 / tanh(ω / 2)
        @test maximum(abs.(covariance(G) - ν * Matrix{Float64}(I, 2, 2))) < 1e-10
        n̄ = 1 / (exp(ω) - 1)
        S = (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
        @test isapprox(entropy(G), S; atol=1e-10)
    end

    @testset "boson Gibbs squeezed (A≠0) vs dense" begin
        # H = -h0 a†a + ½(λ a†² + λ a²)；稠密 Fock 仲裁（annihilator 已修正）
        d = 40
        aop = zeros(ComplexF64, d + 1, d + 1)
        for k in 1:d; aop[k, k+1] = sqrt(k); end
        ad = adjoint(aop)
        for (h0, λ) in ((3.0, 0.0), (3.0, 0.8), (2.0, 1.4))
            H = BosonicQuadraticHermitianOperator(ComplexF64[-h0;;], ComplexF64[λ;;])
            ρ = exp(GS._fock_operators(H; d=d)); ρ ./= tr(ρ)
            ops = [aop, ad]
            Cd = zeros(ComplexF64, 2, 2)
            for i in 1:2, j in 1:2
                Cd[i,j] = 0.5 * tr(ρ * (ops[i] * ops[j]' + ops[j]' * ops[i]))
            end
            Σd = GS.to_xp(Cd)
            G = GaussianState(exp(H))
            @test maximum(abs.(covariance(G) .- Σd)) < 5e-3   # 截断极限
            @test all(eigvals(Symmetric(covariance(G))) .> 0)  # 正定
        end
        # 不可归一化（h > 0：e^H 无界）与不稳定（A 过大）都应抛 ArgumentError
        @test_throws ArgumentError GaussianState(exp(
            BosonicQuadraticHermitianOperator(ComplexF64[1.0;;], ComplexF64[0.0;;])))
        @test_throws ArgumentError GaussianState(exp(
            BosonicQuadraticHermitianOperator(ComplexF64[-1.0;;], ComplexF64[2.0;;])))
    end

    @testset "boson general purify" begin
        # 数守恒解析锚：t = e^h（复），Choi = Σ t^m |m,m⟩ 的双模压缩型协方差
        h0 = -1.0 + 0.3im
        op = exp(BosonicQuadraticOperator(ComplexF64[h0;;],
                 zeros(ComplexF64,1,1), zeros(ComplexF64,1,1)))
        psi = purify(op)
        @test ispure(psi)
        t = exp(h0); d = 1 - abs(t)^2
        ν = 0.5 * (1 + abs(t)^2) / d
        Σref = [real(t)/d imag(t)/d; imag(t)/d -real(t)/d]
        @test maximum(abs.(covariance(psi)[1:2,1:2] .- ν*Matrix{Float64}(I,2,2))) < 1e-12
        @test maximum(abs.(covariance(psi)[1:2,3:4] .- Σref)) < 1e-12
        # 配对 vs 稠密 Fock Choi（含复 h）
        for (h, A, B) in ((ComplexF64[-1.5;;], ComplexF64[0.3;;], ComplexF64[0.3;;]),
                          (ComplexF64[-2.0+0.5im;;], ComplexF64[0.15;;], ComplexF64[-0.1;;]))
            Hop = BosonicQuadraticOperator(h, A, B)
            N = 20
            g = exp(GS._fock_operators(Hop; d=N))
            Iᵈ = Matrix{ComplexF64}(I, N+1, N+1)
            vecI = sum(kron(v, v) for v in eachcol(Iᵈ))
            ψd = kron(g, Iᵈ) * vecI; ψd ./= norm(ψd)
            a = GS._boson_annihilator(1, N)[1]
            a1 = kron(a, Iᵈ); a2 = kron(Iᵈ, a)
            s2 = 1/sqrt(2)
            ops = [s2*(a1+a1'), s2*im*(a1'-a1), s2*(a2+a2'), s2*im*(a2'-a2)]
            Σd = zeros(Float64, 4, 4)
            for i in 1:4, j in 1:4
                Σd[i,j] = real(dot(ψd, (ops[i]*ops[j] + ops[j]*ops[i])*ψd))/2
            end
            @test maximum(abs.(covariance(purify(exp(Hop))) .- Σd)) < 5e-3
        end
        # 不可归一化（|tanh r| ≥ 1）应抛 ArgumentError
        @test_throws ArgumentError purify(exp(BosonicQuadraticOperator(
            ComplexF64[1.0;;], zeros(ComplexF64,1,1), zeros(ComplexF64,1,1))))
    end

    @testset "boson TFD" begin
        for (n, seed) in ((1, 3), (2, 5), (3, 9))
            rng = MersenneTwister(seed)
            X = randn(rng, ComplexF64, n, n)
            h = -((X + X') / 2 + 3I)
            A0 = 0.5 .* randn(rng, ComplexF64, n, n)
            A = (A0 + transpose(A0)) / 2
            H = BosonicQuadraticHermitianOperator(h, A)
            gs = exp(H)
            G = GaussianState(gs)
            psi = purify(gs)
            @test ispure(psi)
            red = partialtr(psi, collect(1:n))
            @test maximum(abs.(covariance(red) .- covariance(G))) < 1e-10
            @test entropy(psi) < 1e-10
        end
        # n=1 纯热态解析：TFD = (ρ^{1/2}⊗I)|I⟩ = 双模压缩（x-x 交叉 +g）
        H = BosonicQuadraticHermitianOperator([-0.9;;], [0.0;;])
        psi = purify(exp(H))
        ν = 0.5 / tanh(0.45)
        g = sqrt(ν^2 - 0.25)
        Σref = [ν 0 g 0; 0 ν 0 -g; g 0 ν 0; 0 -g 0 ν]
        @test maximum(abs.(covariance(psi) .- Σref)) < 1e-10
        n̄ = 1 / (exp(0.9) - 1)
        @test isapprox(entropy(partialtr(psi, [1])), (n̄+1)*log(n̄+1) - n̄*log(n̄); atol=1e-10)
    end
    @testset "properties (§6.4)" begin
        # 整纯态熵 0（含 Rényi）；S(∅)=0
        op = exp(random_op(2, 7))
        psi = purify(op)
        @test entropy(psi) < 1e-10
        @test entropy(psi; α=2) < 1e-10
        @test entropy(psi, Int[]) == 0.0
        # 纯态互补 S(A) = S(Ā)（费米 2n=4 模、玻色 TFD）
        for modes in ([1], [2], [1,2], [1,3])
            comp = setdiff(1:4, modes)
            @test isapprox(entropy(psi, modes), entropy(psi, comp); atol=1e-9)
        end
        Hb = BosonicQuadraticHermitianOperator(
            ComplexF64[-1.5 0.2; 0.2 -2.1], ComplexF64[0.1 -0.05; -0.05 0.08])
        psib = purify(exp(Hb))
        @test entropy(psib) < 1e-10
        for modes in ([1], [1,3])
            comp = setdiff(1:4, modes)
            @test isapprox(entropy(psib, modes), entropy(psib, comp); atol=1e-9)
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
        # 费米 α=2 与稠密偏迹一致（n=2 Gibbs，模式 1）
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
        # 玻色 α=2 解析：单模热态 p_n = (1-t)t^n ⟹ S₂ = log((1+t)/(1-t))
        ω = 0.8
        Gb = GaussianState(exp(BosonicQuadraticHermitianOperator(
            ComplexF64[-ω;;], ComplexF64[0.0;;])))
        t = exp(-ω)
        @test isapprox(entropy(Gb; α=2), log((1+t)/(1-t)); atol=1e-10)
    end

    @testset "error paths" begin
        op = exp(random_op(2, 7))
        psi = purify(op)
        # 尺寸/对称性校验
        @test_throws ArgumentError FermionicQuadraticOperator(
            zeros(ComplexF64,2,3), zeros(ComplexF64,2,2), zeros(ComplexF64,2,2))
        @test_throws ArgumentError FermionicQuadraticOperator(rand(3,3))
        @test_throws ArgumentError FermionicQuadraticOperator(
            rand(2,2), rand(2,2), zeros(ComplexF64,2,2))          # A 非反对称
        @test_throws ArgumentError FermionicQuadraticHermitianOperator(
            rand(2,2) + im .* rand(2,2), zeros(ComplexF64,2,2))    # h 非厄米
        @test_throws ArgumentError BosonicQuadraticOperator(
            rand(2,2), rand(2,2), zeros(ComplexF64,2,2))           # A 非对称
        @test_throws ArgumentError BosonicQuadraticHermitianOperator(
            rand(2,2) + im .* rand(2,2), zeros(ComplexF64,2,2); c0=im)
        # 非物理协方差
        @test_throws ArgumentError FermionicGaussianState([1.0 0.0; 0.0 1.0])         # 费米须反对称
        @test_throws ArgumentError BosonicGaussianState(ones(2, 2))                   # ν < 1/2
        @test_throws ArgumentError BosonicGaussianState(-Matrix{Float64}(I,2,2))       # 不正定
        # 越界 / 重复 modes / 非法 α
        @test_throws BoundsError entropy(psi, [5])
        @test_throws BoundsError partialtr(psi, [0])
        @test_throws ArgumentError entropy(psi, [1, 1])
        @test_throws ArgumentError partialtr(psi, [2, 2])
        @test_throws ArgumentError entropy(psi; α=0)
        @test_throws ArgumentError entropy(psi; α=-1)
    end
end
