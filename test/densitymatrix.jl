# ---------------- 5. densitymatrix ----------------
@testset "densitymatrix" begin
    # 费米 Gibbs：与 helpers 的独立稠密实现对拍
    m = 2
    h = rand_sym(rng, m; scale=1.0)
    g = rand_sym(rng, m; scale=0.5)
    M = fermionic_bdg_matrix(h, g)
    s = FermionicGaussianState(M)
    ρ = densitymatrix(s)
    @test size(ρ) == (2^m, 2^m)
    @test ρ ≈ fermionic_gibbs_density_matrix(M) atol=1e-10
    @test tr(ρ) ≈ 1 atol=1e-12
    @test maximum(abs, ρ - ρ') < 1e-12

    # 费米纯配对：doubled 空间的 |Ψ⟩⟨Ψ|，与 helpers 的 Taylor 实现对拍
    W = rand_antisym(rng, 6; scale=0.5)
    sp = FermionicGaussianState(W)
    ρp = densitymatrix(sp)
    @test size(ρp) == (2^6, 2^6)
    @test ρp ≈ fermionic_density_matrix(W) atol=1e-10
    @test tr(ρp) ≈ 1 atol=1e-12
    @test maximum(abs, eigvals(Hermitian((ρp + ρp') ./ 2))) ≥ 1 - 1e-10  # 纯态

    # 玻色：Fock 截断，随 d 收敛到协方差熵
    B = bosonic_bdg_matrix(rand_spd(rng, 2; shift=1.0), 0.1 .* randn(rng, 2, 2))
    sb = BosonicGaussianState(B)
    ρ5 = densitymatrix(sb; d=5)
    @test size(ρ5) == (25, 25)
    @test tr(ρ5) ≈ 1 atol=1e-12
    @test maximum(abs, ρ5 - ρ5') < 1e-12
    ρ8 = densitymatrix(sb; d=8)
    @test ρ8 ≈ bosonic_density_matrix(B; nbas=7)[1] atol=1e-12
    # d ↑ ⟹ 熵收敛到协方差算法的整态熵（残差为 Fock 截断，随 d 指数下降）
    ρ20 = densitymatrix(sb; d=20)
    @test vn_entropy(ρ20) ≈ entropy(sb) atol=1e-3
    @test_throws ArgumentError densitymatrix(sb; d=1)
end

# ---------------- 6. benchmark ----------------
@testset "benchmark" begin
    n = 5                                        # 10 个模 → 稠密维数 1024
    W = rand_antisym(rng, 2n; scale=0.3)
    s = FermionicGaussianState(W)
    modes = collect(1:5)
    Sf, Sd, tf, td = benchmark_fermionic(modes -> osee(s, modes), W, modes)
    @test Sf ≈ Sd atol = 1e-9
    @test tf < td
    @info "benchmark（10 模）" S_fast = Sf S_dense = Sd t_fast = tf t_dense = td speedup = td / tf
end
