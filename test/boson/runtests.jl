# 玻色子测试：Gibbs（解析 + 压缩 vs 稠密）、一般 purify、TFD、性质、错误路径
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Bosonic" begin
    @testset "Gibbs thermal" begin
        ω = 0.7
        H = BosonicQuadraticHermitianOperator([-ω;;], [0.0;;])
        G = GaussianState(exp(H))
        ν = 0.5 / tanh(ω / 2)
        @test maximum(abs.(covariance(G) - ν * Matrix{Float64}(I, 2, 2))) < 1e-10
        n̄ = 1 / (exp(ω) - 1)
        S = (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
        @test isapprox(entropy(G), S; atol=1e-10)
    end

    @testset "Gibbs squeezed (A≠0) vs dense" begin
        # H = -h0 a†a + ½(λ a†² + λ a²)；稠密 Fock 仲裁（annihilator 已修正）
        d = 40
        aop = zeros(ComplexF64, d + 1, d + 1)
        for k in 1:d; aop[k, k+1] = sqrt(k); end
        ad = adjoint(aop)
        for (h0, λ) in ((3.0, 0.0), (3.0, 0.8), (2.0, 1.4))
            H = BosonicQuadraticHermitianOperator(ComplexF64[-h0;;], ComplexF64[λ;;])
            ρ = fockmatrix(exp(H); d=d); ρ ./= tr(ρ)
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

    @testset "general purify" begin
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
            g = fockmatrix(exp(Hop); d=N)
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

    @testset "TFD" begin
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
        # 纯态互补 S(A) = S(Ā)；整态熵 0
        Hb = BosonicQuadraticHermitianOperator(
            ComplexF64[-1.5 0.2; 0.2 -2.1], ComplexF64[0.1 -0.05; -0.05 0.08])
        psib = purify(exp(Hb))
        @test entropy(psib) < 1e-10
        for modes in ([1], [1,3])
            comp = setdiff(1:4, modes)
            @test isapprox(entropy(psib, modes), entropy(psib, comp); atol=1e-9)
        end
        # α=2 解析：单模热态 p_n = (1-t)t^n ⟹ S₂ = log((1+t)/(1-t))
        ω = 0.8
        Gb = GaussianState(exp(BosonicQuadraticHermitianOperator(
            ComplexF64[-ω;;], ComplexF64[0.0;;])))
        t = exp(-ω)
        @test isapprox(entropy(Gb; α=2), log((1+t)/(1-t)); atol=1e-10)
    end

    @testset "symmetrize equivalence" begin
        rng = MersenneTwister(21)
        # 玻色：任意 A、B（含反对称部分）与对称化等价
        hb = -(randn(rng, ComplexF64, 3, 3) + 2I)                  # 稳定（冷）算子
        Ab = (0.2/sqrt(3)) .* randn(rng, ComplexF64, 3, 3)
        Bb = (0.2/sqrt(3)) .* randn(rng, ComplexF64, 3, 3)
        opb1 = BosonicQuadraticOperator(hb, Ab, Bb)
        opb2 = BosonicQuadraticOperator(hb, GS.symmetrize(Ab), GS.symmetrize(Bb))
        @test fockmatrix(opb1; d=10) ≈ fockmatrix(opb2; d=10)
        @test covariance(purify(exp(opb1))) ≈ covariance(purify(exp(opb2)))
    end

    @testset "error paths" begin
        # 厄米性 / c0 实数校验
        @test_throws ArgumentError BosonicQuadraticHermitianOperator(
            rand(2,2) + im .* rand(2,2), zeros(ComplexF64,2,2); c0=im)
        # 非物理协方差
        @test_throws ArgumentError BosonicGaussianState(ones(2, 2))                   # ν < 1/2
        @test_throws ArgumentError BosonicGaussianState(-Matrix{Float64}(I,2,2))       # 不正定
        # 越界 / 重复 modes / 非法 α
        psi = purify(exp(BosonicQuadraticOperator(ComplexF64[-2.0+0.5im;;],
                 ComplexF64[0.15;;], ComplexF64[-0.1;;])))
        @test_throws BoundsError entropy(psi, [5])
        @test_throws BoundsError partialtr(psi, [0])
        @test_throws ArgumentError entropy(psi, [1, 1])
        @test_throws ArgumentError partialtr(psi, [2, 2])
        @test_throws ArgumentError entropy(psi; α=0)
        @test_throws ArgumentError entropy(psi; α=-1)
    end
end
