# ---------------- 2. Gibbs 分支：随机 h,g 对稠密态 ----------------
@testset "稠密仲裁：Gibbs 分支（随机 h,g）" begin
    for m in (2, 3)                              # m 个物理模
        h = rand_sym(rng, m; scale=1.0)
        g = 0.5 .* randn(rng, m, m)              # 非对称 ⟹ A = g−gᵀ ≠ 0（含配对）
        M = fermionic_bdg_matrix(h, g)
        s = FermionicGaussianState(M)

        ρ = fermionic_gibbs_density_matrix(M)
        @test maximum(abs, covariance(s) - covariance_from_density_matrix(ρ, m)) < 1e-10

        # doubled 协方差是纯态：Γd² = −I
        @test maximum(abs, GaussianStates._gammad(s)^2 + I) < 1e-10

        # OSEE（doubled 模式指标：物理 site p = [p, m+p]）：
        # 协方差快速算法 vs 稠密向量化
        for modes in ([1], [1, 2])
            Sf = osee(s, vcat(modes, m .+ modes))
            Sd = dense_osee_fermionic_gibbs(M, modes)
            @test Sf ≈ Sd atol = 1e-10
        end
        # 互补对称：|ρ⟩⟩ 是纯态 ⟹ S(A) = S(Ā)
        if m ≥ 2
            @test osee(s, [1, m+1]) ≈ osee(s, vcat(2:m, (m+2):2m)) atol = 1e-10
        end
        # 全割的 OSEE = 0（无 bipartition）
        @test osee(s, collect(1:2m)) ≈ 0.0 atol = 1e-10
        # 整态熵走 entropy，不是 osee
        @test entropy(s) ≈ vn_entropy(ρ) atol = 1e-10
        @test osee(s, Int[]) == 0.0
    end
end

@testset "Gibbs：非规范厄米输入自动规范化" begin
    # 任意厄米 M 与规范 BdG 形式只差常数（作为算符 H = ½ψ†Mψ 等价），
    # 构造态时自动规范化；结果与稠密参考（直接从 M 构造 H 算符）一致
    Xr = randn(rng, 4, 4)
    Hr = (Xr + Xr') ./ 2                    # 非规范厄米（M₂₂ ≠ −M₁₁ᵀ）
    @test maximum(abs, Hr[3:4, 3:4] + transpose(Hr[1:2, 1:2])) > 1e-3  # 确认非规范
    s0 = FermionicGaussianState(Hr)
    @test s0.isbdg
    @test entropy(s0, [1]) ≈ dense_entropy_fermionic_gibbs(Hr, [1]) atol = 1e-10
    @test osee(s0, [1, 3]) ≈ dense_osee_fermionic_gibbs(Hr, [1]) atol = 1e-10
    # 单位矩阵（完全非规范）⟺ H = 常数 ⟺ 最大混合；熵 = m ln2，
    # 而 site-OSEE = 0（|I⟩ = ⊗ 模式对的最大纠缠对，是 site 直积态）
    s1 = FermionicGaussianState(Matrix{Float64}(I, 4, 4))
    @test entropy(s1) ≈ 2 * log(2) atol = 1e-8
    @test osee(s1, [1, 3]) ≈ 0.0 atol = 1e-8
end

@testset "Gibbs：温度极限" begin
    # 规范谱 ±λ → λ→+∞：空态 / 满态（Γ² = −I）
    Mvac = fermionic_bdg_matrix(fill(50.0, 1, 1))
    sv = FermionicGaussianState(Mvac)
    @test maximum(abs, covariance(sv)^2 + I) < 1e-10
    @test entropy(sv) ≈ 0.0 atol = 1e-12
    Mfull = fermionic_bdg_matrix(fill(-50.0, 1, 1))
    sfull = FermionicGaussianState(Mfull)
    @test maximum(abs, covariance(sfull)^2 + I) < 1e-10
    @test entropy(sfull) ≈ 0.0 atol = 1e-12

    # 高温（规范谱 → 0）⇒ 最大混合，熵 = m·ln2（m=2 个模）
    s0 = FermionicGaussianState(fermionic_bdg_matrix(1e-12 .* Matrix{Float64}(I, 2, 2)))
    @test entropy(s0) ≈ 2 * log(2) atol = 1e-8
end

@testset "稠密仲裁：Gibbs 分支（复 h,g；Γ/Γd 实数性）" begin
    # 对任意态 γ_aγ_b − δ_ab 反厄米 ⟹ 期望纯虚 ⟹ i⟨γ_aγ_b⟩ 严格实，
    # 与 bdg 是否复无关。此处用复厄米 h + 复 g 实证（对照稠密 ρ）。
    for m in (2, 3)
        X = randn(rng, ComplexF64, m, m)
        h = (X + X') ./ 2                      # 复厄米
        g = randn(rng, ComplexF64, m, m)
        M = fermionic_bdg_matrix(h, g)
        @test !(eltype(M) <: Real)             # 确认输入确实复
        s = FermionicGaussianState(M)
        ρ = fermionic_gibbs_density_matrix(M)
        Γ = covariance(s)
        @test Γ isa Matrix{Float64}
        @test maximum(abs, Γ - covariance_from_density_matrix(ρ, m)) < 1e-10
        Γd = GaussianStates._gammad(s)
        @test Γd isa Matrix{Float64}
        @test maximum(abs, Γd^2 + I) < 1e-10   # 纯态
        for modes in ([1], [1, 2])
            @test osee(s, vcat(modes, m .+ modes)) ≈
                  dense_osee_fermionic_gibbs(M, modes) atol = 1e-10
        end
    end
end
