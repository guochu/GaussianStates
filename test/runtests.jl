# ============================================================================
# GaussianStates.jl 测试
#
# 两条费米分支各自用**稠密密度矩阵**独立对照：
#   Gibbs：ρ = exp(−½ψ†·bdg·ψ)/Z 的 Nambu-Fock 稠密表示 → 偏迹 → 熵
#   纯配对：|Ψ⟩ = exp(½c†Wc†)|0⟩ 的 JW 稠密表示 → ρ=ψψ† → 偏迹 → 熵
# 玻色同样用 Fock 截断密度矩阵对照。全部用随机 h、g。
# ============================================================================

using GaussianStates
using LinearAlgebra
using Random
using Test

include("helpers.jl")

@testset "GaussianStates" begin
    rng = MersenneTwister(20240915)

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

    # ---------------- 3. 纯配对分支：反对称 bdg ----------------
    @testset "稠密仲裁：纯配对分支（反对称，随机）" begin
        for n in (2, 3)                              # 2n 个模（doubled）
            W = rand_antisym(rng, 2n; scale=0.5)
            s = FermionicGaussianState(W)
            @test maximum(abs, covariance(s) - fermionic_covariance(W)) < 1e-10
            @test maximum(abs, covariance(s)^2 + I) < 1e-10     # 纯态
            for k in 1:n
                modes = collect(1:k)
                @test osee(s, modes) ≈ dense_osee_fermionic(W, modes) atol = 1e-10
            end
            @test osee(s, 1:(2n)) ≈ 0.0 atol = 1e-9   # 纯态整态熵 = 0
        end
    end

    @testset "纯配对：单模解析 S = h(s²/(1+s²))" begin
        for Δ in (0.1, 0.5, 1.0, 2.0)
            W = [0.0 -Δ; Δ 0.0]                      # 2 个模的纯配对
            s = FermionicGaussianState(W)
            ε = Δ^2 / (1 + Δ^2)
            Sexact = -ε * log(ε) - (1 - ε) * log(1 - ε)
            @test osee(s, [1]) ≈ Sexact atol = 1e-12
            @test osee(s, [2]) ≈ Sexact atol = 1e-12
            @test osee(s, [1]; α=2) ≈ -log(ε^2 + (1 - ε)^2) atol = 1e-12
        end
    end

    @testset "纯配对：双独立配对可加" begin
        s1, s2 = 0.5, 1.3
        Δ = [s1 0.0; 0.0 s2]
        W = zeros(4, 4); W[3:4, 1:2] .= Δ; W = W - transpose(W)
        st = FermionicGaussianState(W)
        ε1 = s1^2 / (1 + s1^2); ε2 = s2^2 / (1 + s2^2)
        h2(p) = -p * log(p) - (1 - p) * log(1 - p)
        @test osee(st, [1]) ≈ h2(ε1) atol = 1e-12
        @test osee(st, [2]) ≈ h2(ε2) atol = 1e-12
        @test osee(st, [1, 2]) ≈ h2(ε1) + h2(ε2) atol = 1e-12
        @test osee(st, 1:4) ≈ 0.0 atol = 1e-12
        @test_throws BoundsError osee(st, [0])
        @test_throws BoundsError osee(st, [5])
        @test_throws ArgumentError osee(st, [1, 1])
    end

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

    # ---------------- 4. 玻色 ----------------
    @testset "稠密仲裁：玻色（随机 h,g）" begin
        for trial in 1:2
            nm = 2
            hb = rand_spd(rng, nm; shift=1.0)
            gb = 0.1 .* randn(rng, nm, nm)
            B = bosonic_bdg_matrix(hb, gb)
            s = BosonicGaussianState(B)

            # 子系统熵（偏迹）
            for m in 1:nm
                @test entropy(s, [m]) ≈
                      dense_entropy_bosonic(B, [m]; nbas=48) atol=1e-8
            end
            # OSEE（operator space，doubled 模式指标：site m = [m, nm+m]）：
            # doubled 协方差快速算法 vs 稠密向量化
            for modes in ([1], [2], [1, 2])
                @test osee(s, vcat(modes, nm .+ modes)) ≈
                      dense_osee_bosonic(B, modes; nbas=24) atol=1e-7
            end
            # doubled 协方差是纯态：辛本征值全为 ½
            m = nmodes(s)
            Γd = GaussianStates._gammad(s)
            Ωd = zeros(4m, 4m)
            for k in 1:2m
                Ωd[2k-1, 2k] = 1.0
                Ωd[2k, 2k-1] = -1.0
            end
            ν = abs.(eigvals(Ωd * Γd))
            @test maximum(abs.(ν .- 0.5)) < 1e-9
        end
    end

    @testset "玻色：bdg = ωI 的热态" begin
        for ω in (0.5, 1.0, 3.0)
            s = BosonicGaussianState(ω * Matrix{Float64}(I, 2, 2))
            @test covariance(s) ≈ 0.5 / tanh(ω / 2) * Matrix{Float64}(I, 2, 2) atol=1e-10
            n̄ = 1 / (exp(ω) - 1)
            Sexact = (n̄ + 1) * log(n̄ + 1) - n̄ * log(n̄)
            @test entropy(s, 1) ≈ Sexact atol=1e-10   # 单模：整态熵
            @test entropy(s) ≈ Sexact atol=1e-10
            @test osee(s, [1, 2]) ≈ 0.0 atol=1e-10    # 单模无 bipartition
        end
    end

    @testset "玻色：双模热态熵可加 + Rényi" begin
        ω1, ω2 = 0.7, 2.3
        B = zeros(4, 4)
        B[1, 1] = B[2, 2] = ω1
        B[3, 3] = B[4, 4] = ω2
        s = BosonicGaussianState(B)
        n̄(ω) = 1 / (exp(ω) - 1)
        SvN(ω) = (n̄(ω) + 1) * log(n̄(ω) + 1) - n̄(ω) * log(n̄(ω))
        @test entropy(s, 1) ≈ SvN(ω1) atol = 1e-10
        @test entropy(s, 2) ≈ SvN(ω2) atol = 1e-10
        @test entropy(s) ≈ SvN(ω1) + SvN(ω2) atol = 1e-10
        @test entropy(s, 1; α=2) ≈ log(1 + 2 * n̄(ω1)) atol = 1e-10
        @test entropy(s, 1; α=1 + 1e-3) ≈ SvN(ω1) rtol = 1e-3
        # 两模是直积热态 ⇒ operator-space OSEE 为 0
        @test osee(s, [1, 3]) ≈ 0.0 atol=1e-8
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
    end

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
end
