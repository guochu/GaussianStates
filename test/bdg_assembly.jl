# ---------------- 0. BdG 矩阵组装 ----------------
@testset "BdG 矩阵组装" begin
    n = 4                                        # 4 个物理模（2 格点 × 2 自旋）
    h = rand_sym(rng, n)                         # 厄米
    g = randn(rng, n, n)                         # 任意；只有反对称部分有效
    B = fermionic_bdg_matrix(h, g)

    A = g - transpose(g)
    @test size(B) == (2n, 2n)
    @test B[1:n, 1:n] == h
    @test B[1:n, n+1:2n] == A
    @test B[n+1:2n, 1:n] == A'                   # 左下 = A†
    @test B[n+1:2n, n+1:2n] == -transpose(h)     # 右下 = −hᵀ
    @test B ≈ B'                                 # 厄米
    @test fermionic_bdg_matrix(h) == fermionic_bdg_matrix(h, zeros(n, n))

    # g 的对称部分不影响矩阵
    gsym = (g + transpose(g)) ./ 2
    @test fermionic_bdg_matrix(h, gsym) == fermionic_bdg_matrix(h, zeros(n, n))

    # 玻色：Bogoliubov 块（复厄米 h + 复任意 g）→ 实 (x,p) K
    Xc = randn(rng, ComplexF64, n, n)
    hb = (Xc + Xc') ./ 2                       # 复厄米
    gb = randn(rng, ComplexF64, n, n)
    Bb = bosonic_bdg_matrix(hb, gb)
    gs = (gb + transpose(gb)) ./ 2
    Kxp = 2 .* (imag.(gs) .- imag.(hb))
    @test Bb == [real.(hb).+real.(gs) Kxp; transpose(Kxp) real.(hb).-real.(gs)]
    @test Bb isa Matrix{Float64}
    @test Bb == transpose(Bb)                   # 实对称
    @test bosonic_bdg_matrix(hb) == bosonic_bdg_matrix(hb, zeros(n, n))
    # g 的反对称部分无效
    @test bosonic_bdg_matrix(hb, gb + (gb - transpose(gb)) ./ 2) ≈ Bb
    # 尺寸 / 对称性校验
    @test_throws ArgumentError bosonic_bdg_matrix(randn(2, 3))
    @test_throws ArgumentError bosonic_bdg_matrix(randn(n, n), randn(2, 2))
    @test_throws ArgumentError bosonic_bdg_matrix(randn(n, n))   # 非厄米

    # 类型只做自然提升：实输入不复数化
    @test fermionic_bdg_matrix(h, g) isa Matrix{Float64}
    @test bosonic_bdg_matrix(hb, gb) isa Matrix{Float64}
    hc = ComplexF64.(h) .+ 0.0im
    gc = ComplexF64.(g)
    @test fermionic_bdg_matrix(hc, gc) isa Matrix{ComplexF64}
    @test FermionicGaussianState(fermionic_bdg_matrix(h, g)) isa
          FermionicGaussianState{Float64}

    # 尺寸 / 对称性校验
    @test_throws ArgumentError fermionic_bdg_matrix(randn(2, 3))
    @test_throws ArgumentError fermionic_bdg_matrix(randn(n, n))   # 非厄米
    @test_throws MethodError bosonic_bdg_matrix(hb, gb, randn(n, n))
end

@testset "玻色 BdG 转换：稠密 Fock 仲裁" begin
    # H（a,a† Bogoliubov 形式，复系数）≡ ½RᵀKR + const，
    # 在非截断边界态的子空间上逐元素成立（边界态仅有 Fock 截断效应）
    n, nb = 2, 8
    d = nb + 1
    a = single_mode_a(nb); ad = transpose(a)
    ai = [ComplexF64.(embed_operator(a, k, n, d)) for k in 1:n]
    adi = [adjoint(x) for x in ai]
    xm = (a .+ ad) ./ sqrt(2); pm = (-1im .* (a .- ad)) ./ sqrt(2)
    xi = [ComplexF64.(embed_operator(xm, k, n, d)) for k in 1:n]
    pim = [ComplexF64.(embed_operator(pm, k, n, d)) for k in 1:n]
    X = randn(rng, ComplexF64, n, n)
    h = 0.4 .* ((X + X') ./ 2)
    g = 0.3 .* randn(rng, ComplexF64, n, n)
    Ha = sum(h[i, j] * adi[i] * ai[j] for i in 1:n, j in 1:n)
    Ha += 0.5 .* sum(g[i, j] * adi[i] * adi[j] for i in 1:n, j in 1:n)
    Ha += 0.5 .* sum(conj(transpose(g))[i, j] * ai[i] * ai[j] for i in 1:n, j in 1:n)
    K = bosonic_bdg_matrix(h, g)
    Kxx = K[1:n, 1:n]; Kpp = K[n+1:2n, n+1:2n]; Kxp = K[1:n, n+1:2n]
    HK = 0.5 .* (sum(Kxx[i, j] * xi[i] * xi[j] + Kpp[i, j] * pim[i] * pim[j]
                     for i in 1:n, j in 1:n))
    HK += 0.25 .* sum(Kxp[i, j] * (xi[i] * pim[j] + pim[j] * xi[i]) for i in 1:n, j in 1:n)
    D = Ha - HK
    safe = [i for i in 0:(d^n - 1) if all(digits(i; base = d, pad = n) .< nb)] .+ 1
    Ds = D[safe, safe]
    II = Matrix{ComplexF64}(I, length(safe), length(safe))
    @test maximum(abs, Ds .- Ds[1, 1] .* II) < 1e-9
end

# ---------------- 1. 构造校验 ----------------
@testset "构造校验" begin
    # isbdg：费米 Gibbs = true，纯配对 = false（玻色无此字段——恒为密度矩阵输入）
    @test FermionicGaussianState(fermionic_bdg_matrix(rand_sym(rng, 2), randn(rng, 2, 2))).isbdg
    @test !FermionicGaussianState(rand_antisym(rng, 4; scale=0.5)).isbdg
    # 既非厄米也非反对称
    @test_throws ArgumentError FermionicGaussianState([1.0 2.0; 0.0 1.0])
    # 玻色必须对称、正定、实
    @test_throws ArgumentError BosonicGaussianState([1.0 0.2; 0.0 1.0])
    @test_throws ArgumentError BosonicGaussianState([1.0 0.0; 0.0 -1.0])
    @test_throws ArgumentError BosonicGaussianState(ComplexF64[1.0 0.5im; -0.5im 1.0])
    # 奇数维 / 非方阵
    @test_throws ArgumentError FermionicGaussianState(zeros(3, 3))
    @test_throws ArgumentError BosonicGaussianState(Matrix{Float64}(I, 3, 3))
    @test_throws ArgumentError FermionicGaussianState(zeros(2, 4))
    # 只接受具体 Matrix
    B = ComplexF64[0 -0.4; 0.4 0]
    @test_throws MethodError FermionicGaussianState(@view(B[:, :]))
    @test_throws MethodError BosonicGaussianState(Symmetric(Matrix{Float64}(I, 2, 2)))
    # 类型参数与字段
    s = FermionicGaussianState(B)
    @test s isa FermionicGaussianState{ComplexF64}
    @test s.bdg === B
    @test BosonicGaussianState(Matrix{Float64}(I, 2, 2)) isa BosonicGaussianState{Float64}
end
