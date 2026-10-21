# ---------------- entropy 接口 ----------------
@testset "entropy 接口" begin
    # 费米 Gibbs：与稠密态、与 osee 双重对照
    m = 2
    h = rand_sym(rng, m; scale=1.0)
    g = 0.5 .* randn(rng, m, m)
    M = fermionic_bdg_matrix(h, g)
    s = FermionicGaussianState(M)
    ρ = fermionic_gibbs_density_matrix(M)

    @test nmodes(s) == m
    @test entropy(s) ≈ vn_entropy(ρ) atol = 1e-10
    for site in 1:m
        @test entropy(s, site) ≈ dense_entropy_fermionic_gibbs(M, [site]) atol = 1e-10
        @test entropy(s, site) ≈ entropy(s, [site]) atol = 1e-12
        @test osee(s, [site, m+site]) ≈ dense_osee_fermionic_gibbs(M, [site]) atol = 1e-10
    end
    @test entropy(s, Int[]) == 0.0
    # Rényi-2 的整态熵 = −ln tr ρ²
    @test entropy(s; α=2) ≈ -log(sum(abs2, eigvals(Hermitian((ρ + ρ') ./ 2)))) atol = 1e-10

    # 费米纯配对算符：isbdg = false，entropy 报错，osee 仍有定义
    # （整态纯向量 ⟹ 全割 osee = 0；子系统割与稠密向量化一致）
    W = rand_antisym(rng, 6; scale=0.5)
    sp = FermionicGaussianState(W)
    @test !sp.isbdg
    @test nmodes(sp) == 6
    @test_throws ArgumentError entropy(sp)
    @test_throws ArgumentError entropy(sp, [1, 2])
    @test osee(sp, collect(1:6)) ≈ 0.0 atol = 1e-10
    @test osee(sp, [1, 2]) ≈ dense_osee_fermionic(W, [1, 2]) atol = 1e-10

    # 玻色：整态熵与子系统熵
    B = zeros(4, 4); B[1, 1] = B[2, 2] = 0.7; B[3, 3] = B[4, 4] = 2.3
    sb = BosonicGaussianState(B)
    n̄(ω) = 1 / (exp(ω) - 1)
    SvN(ω) = (n̄(ω) + 1) * log(n̄(ω) + 1) - n̄(ω) * log(n̄(ω))
    @test nmodes(sb) == 2
    @test entropy(sb, 1) ≈ SvN(0.7) atol = 1e-10
    @test entropy(sb) ≈ SvN(0.7) + SvN(2.3) atol = 1e-10
end

# ---------------- 5. 熵的对称性 ----------------
@testset "二分割互补对称 S(A) = S(B)" begin
    W = rand_antisym(rng, 6; scale=0.6)          # 纯态，6 个模
    s = FermionicGaussianState(W)
    N = nmodes(s)
    for L in ([1], [1, 2, 3], [2, 5])
        R = setdiff(1:N, L)
        @test osee(s, L) ≈ osee(s, R) atol = 1e-10
    end
end

@testset "covariance 返回 Γ（惰性缓存）" begin
    s = FermionicGaussianState(rand_antisym(rng, 4; scale=0.5))
    @test s.Γ === nothing                       # 尚未计算
    Γ1 = covariance(s)                           # 首次调用计算
    @test s.Γ === Γ1                             # 缓存
    @test covariance(s) === Γ1                   # 再次调用同一对象
    @test Γ1 isa Matrix{Float64}
    # 纯配对分支：Γd 与 Γ 同一对象
    @test GaussianStates._gammad(s) === Γ1
    # osee_covariance（非导出）在新对象上返回并缓存 Γd
    s2 = FermionicGaussianState(rand_antisym(rng, 4; scale=0.5))
    @test s2.Γd === nothing
    @test GaussianStates.osee_covariance(s2) === GaussianStates._gammad(s2)
    @test s2.Γd === s2.Γ
    # 未导出：using GaussianStates 后裸名不可见
    @test_throws UndefVarError Main.eval(:osee_covariance)
end
