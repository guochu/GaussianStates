# ============================================================================
# bosonic.jl —— 玻色高斯态
#
# 两个分支（与费米侧完全对称，构造时按谱自动判定）：
#   Gibbs   —— bdg 为 2n×2n 实对称正定（正则坐标序 R = (x₁,p₁,x₂,p₂,…)，
#              ρ ∝ exp(−½Rᵀ·bdg·R)；复系数的 Bogoliubov 块经 bosonic_bdg_matrix
#              归一为实 K）。entropy + osee。
#   纯配对 —— bdg 为实对称、谱 ⊂ (−1,1) 的非正定矩阵（典型形式
#              W = [[0, M],[Mᵀ, 0]]），表示纯高斯**向量**
#              |Ψ⟩ = exp(½ c†·W·c†)|0⟩（IF/影响泛函系数态，ρ 非密度矩阵）。
#              只 osee（Γ 限制熵）；其协方差由 G/K 矩阵函数闭式给出（见下）。
# Γ 与 Γd 惰性计算并缓存；纯配对分支 Γd === Γ。
# ============================================================================

"""
    BosonicGaussianState(bdg::Matrix{T}) where {T<:Number}

玻色高斯态，`bdg` 为 2n×2n **实对称**矩阵，按谱自动判定分支：

1. **正定**（Gibbs 形态，`isbdg = true`）：态为高斯密度矩阵
   （正则坐标序 `R = (x₁,p₁,x₂,p₂,…)`，`[x_a,p_b] = iδ_ab`）

   ```text
   ρ = exp(−½ Rᵀ·bdg·R) / Z ,      Z = Tr exp(−½ Rᵀ·bdg·R)
   ```

   `bdg` 已吸收逆温度（一般混合态；零温/纯态对应 `bdg → ∞` 极限）。
   协方差 `Γ_ab = ½⟨{R_a,R_b}⟩`（惰性计算，可经 `s.Γ` 访问）。实现上令
   `A = bdg^{1/2}`、`K = AΩA`，则 `Γ = A⁻¹ g(−iK/2) A⁻¹`，`g(z) = z·coth(z)`
   （`−iK/2` 的本征值为 Williamson 频率 `±ω_k/2`）。校验：
   `bdg = ωI ⇒ Γ = ½coth(ω/2)·I`。

2. **非正定且谱 ⊂ (−1,1)**（纯配对形态，`isbdg = false`）：`bdg = W` 表示
   doubled 空间 2n 个模上的纯高斯**向量**

   ```text
   |Ψ⟩ = exp(½ c†·W·c†)|0⟩ ,      W = Wᵀ 实对称，‖W‖ < 1
   ```

   （如影响泛函系数态 `exp(Σ M_pq c†_p d†_q)|0⟩` 取 `W = [[0,M],[Mᵀ,0]]`，
   见 TEMPO/osee 报告 §5.2）。ρ = |Ψ⟩⟨Ψ| 不是 Gibbs 密度矩阵（纯高斯向量的
   生成元必带相位/反对称部分，任何有限正定 K 都给不出），故不走分支 1。
   协方差由矩阵函数闭式给出（`W = QΛQᵀ`）：

   ```text
   G = (I − W²)⁻¹ = ⟨c c†⟩ ,   K = G·W = ⟨c c⟩
   σ_xx = G + K − ½I ,   σ_pp = G − K − ½I ,   σ_xp = 0
   ```

   （谱上 `G_kk = 1/(1−λ_k²)`、`K_kk = λ_k/(1−λ_k²)`；σ 正定且辛本征值
   全为 ½，即纯态。等价于 ⊗ 双模压缩对。）

`osee` 对两个分支都有定义（Gibbs：doubled 纯化协方差；纯配对：向量协方差
限制熵）；`entropy` 只对 Gibbs 分支（`isbdg = true`）。字段 `Γ`/`Γd` 初始为
`nothing`，惰性缓存。真正复对称的 K（非厄米算符）与复 W 仍未支持——其协方差
是复的，快速算法失效（见文档理论与算法页 §1）。
"""
mutable struct BosonicGaussianState{T<:Number} <: AbstractGaussianState
    bdg::Matrix{T}
    isbdg::Bool                            # true ⟺ Gibbs 密度矩阵；false ⟺ 纯配对向量
    Γ::Union{Nothing,Matrix{Float64}}     # σ_ab = ½⟨{R_a,R_b}⟩（惰性缓存）
    Γd::Union{Nothing,Matrix{Float64}}   # doubled 协方差（osee 用；纯配对分支 = Γ）
    function BosonicGaussianState(bdg::Matrix{T}) where {T<:Number}
        isbdg = _bosonic_validate(bdg)                  # 全部校验在构造时完成
        return new{T}(bdg, isbdg, nothing, nothing)
    end
end

# Γ 的惰性计算与缓存
function _gamma(s::BosonicGaussianState)
    s.Γ === nothing || return s.Γ
    W = _realify("bosonic bdg", s.bdg)
    s.Γ = s.isbdg ? _bosonic_gamma_from_bdg(W) : _bosonic_gamma_pairing(W)
    return s.Γ
end

# Γd 的惰性计算与缓存（纯配对分支：与 Γ 同一对象）
function _gammad(s::BosonicGaussianState)
    s.Γd !== nothing && return s.Γd
    s.isbdg && return s.Γd = _bosonic_doubled_covariance(_realify("bosonic bdg", s.bdg))
    return s.Γd = _gamma(s)
end

"""
    bosonic_bdg_matrix(h::AbstractMatrix, g::AbstractMatrix=zero(h))

由 Bogoliubov 块（湮灭算符形式）组装玻色 BdG 矩阵，返回**实** (x,p) 基矩阵：

```text
H = Σ_ij h_ij a†_i a_j + ½ Σ_ij (g_ij a†_i a†_j + h.c.)
H = ½ Rᵀ K R + const ,     R = (x₁,p₁,x₂,p₂,…)
```

- `h`：**复厄米**（粒子-空穴块）；
- `g`：**任意复矩阵**（配对块；只有其对称部分 `g_s = (g+gᵀ)/2` 有效）。

转换公式（已对稠密 Fock 算符逐元素仲裁，残差仅 Fock 截断边界态）：

```text
Kxx = Re(h) + Re(g_s) ,    Kpp = Re(h) − Re(g_s) ,
Kxp = 2 (Im(g_s) − Im(h))   （Im(h) 反对称 → 磁型反对称块；Im(g_s) 对称）
```

任何厄米二次型 H（含复系数）在 (x,p) 基下的 K 都是实对称——这是 R 算符
实矩阵表示的直接结果（H† 对应 K̄），因此复输入经本函数归一到实 K，
无信息损失。需要直接传入 (x,p) 基实对称正定矩阵时，把它直接交给
[`BosonicGaussianState`](@ref)。注意纯配对主导的输入可使 K 不正定
（对应算符谱无下界），此时构造态会报错。
"""
function bosonic_bdg_matrix(h::AbstractMatrix, g::AbstractMatrix=zero(h))
    n = size(h, 1)
    size(h, 2) == n || throw(ArgumentError("h 必须是方阵"))
    size(g) == (n, n) || throw(ArgumentError("g 必须与 h 同尺寸（$(n)×$(n)）"))
    tol = 1e-10 * max(1.0, opnorm(h, Inf))
    opnorm(h - h', Inf) ≤ tol ||
        throw(ArgumentError("玻色 BdG 矩阵要求 h 厄米（h† = h）"))
    gs = (g + transpose(g)) ./ 2
    Kxx = real.(h) .+ real.(gs)
    Kpp = real.(h) .- real.(gs)
    Kxp = 2 .* (imag.(gs) .- imag.(h))
    return [Kxx Kxp; transpose(Kxp) Kpp]
end

# 构造时的全部校验与分支判定：方阵、偶数维、实对称；
# 正定 → Gibbs（true）；谱 ⊂ (−1,1) 的非正定 → 纯配对（false）；否则报错
function _bosonic_validate(bdg::AbstractMatrix)
    n = size(bdg, 1)
    size(bdg, 2) == n || throw(ArgumentError("bdg 必须是方阵"))
    (n ≥ 2 && iseven(n)) || throw(ArgumentError("bdg 必须是偶数维 2n×2n 矩阵"))
    H = _realify("bosonic bdg", bdg)
    tol = 1e-9 * max(1.0, opnorm(H, Inf))
    opnorm(H - transpose(H), Inf) ≤ tol ||
        throw(ArgumentError("玻色 bdg 必须是实对称矩阵"))
    λ = eigvals(Symmetric(H))
    if minimum(λ) > 0
        return true
    elseif maximum(abs.(λ)) < 1 - 1e-12
        return false
    else
        throw(ArgumentError(
            "玻色 bdg 无法解释：既非正定（Gibbs 形态 ρ ∝ exp(−½RᵀKR)），" *
            "也非谱严格落在 (−1,1) 内（纯配对形态 exp(½c†Wc†)|0⟩）。" *
            "纯配对输入需要 |λ_max(W)| < 1（可归一化条件 I − W² > 0）"))
    end
end

# ========================================================================== #
# bdg → Γ
# ========================================================================== #

# ρ ∝ exp(−½ Rᵀ h R)，R = (x₁,p₁,…)。令 A = h^{1/2}（对称正定）、K = A Ω A
# （实反对称，本征值 ±iω_k 给出 Williamson 频率）。标准结果：
#   Γ = A⁻¹ g(−iK/2) A⁻¹,   g(z) = z·coth(z)
# （−iK/2 是 Hermitian、本征值 ±ω_k/2；g 为偶函数，故作用在谱上为实数）
# 校验：h=ωI ⇒ K=ωΩ ⇒ Γ = ½coth(ω/2)·I，与热振子解析协方差一致。
function _bosonic_gamma_from_bdg(bdg::AbstractMatrix)
    n = size(bdg, 1)
    H = _realify("bosonic bdg", bdg)
    e = eigen(Symmetric(H))
    sq = sqrt.(e.values)
    A = e.vectors * Diagonal(sq) * e.vectors'
    Ainv = e.vectors * Diagonal(1 ./ sq) * e.vectors'
    K = A * _omega(n ÷ 2) * A
    Kh = Hermitian(-1im .* K ./ 2)
    ev = eigen(Kh)
    g(z) = z / tanh(z)                                # z·coth(z)
    fK = ev.vectors * Diagonal(g.(ev.values)) * ev.vectors'
    Γ = Ainv * real.(fK) * Ainv
    return (Γ .+ Γ') ./ 2
end

# 纯配对向量 |Ψ⟩ = exp(½ c†Wc†)|0⟩（W 实对称，‖W‖<1）的协方差。
# W = QΛQᵀ，谱上 G_k = 1/(1−λ_k²)（⟨cc†⟩）、K_k = λ_k/(1−λ_k²)（⟨cc⟩）：
#   σ_xx = G + K − ½I ,  σ_pp = G − K − ½I ,  σ_xp = 0（模式指标上的矩阵函数）
# σ 正定且辛本征值全 ½（纯态）；对 W = [[0,M],[Mᵀ,0]] 等价于 ⊗ 双模压缩对
# （s_k = 奇异值(M)，tanh r_k = s_k）。G/K 公式与 TEMPO/osee 报告 §5.2 一致。
function _bosonic_gamma_pairing(W::Matrix{Float64})
    N = size(W, 1)                                   # 2n 个 doubled 模
    e = eigen(Symmetric(W))
    λ = e.values
    gλ = 1 ./ (1 .- λ .^ 2)
    G = e.vectors * Diagonal(gλ) * e.vectors'
    K = e.vectors * Diagonal(λ .* gλ) * e.vectors'
    xx = real.(G) .+ real.(K) .- 0.5 .* Matrix{Float64}(I, N, N)
    pp = real.(G) .- real.(K) .- 0.5 .* Matrix{Float64}(I, N, N)
    Γ = zeros(2N, 2N)                                # (x₁,p₁,…,x_N,p_N)
    for i in 1:N, j in 1:N
        Γ[2i-1, 2j-1] = xx[i, j]
        Γ[2i, 2j] = pp[i, j]
    end
    return (Γ .+ Γ') ./ 2
end

# Williamson 分解：返回 (ω, S)，S 实辛矩阵（SᵀΩS = Ω）且
#   Sᵀ h S = diag(ω₁,ω₁,ω₂,ω₂,…)
# 构造：w_k = iK = iAΩA 的 +ω_k 本征矢（K 实反对称 ⇒ iK 厄米），
# u_k = A⁻¹w_k 是 Ωh 的本征矢（Ωh u = iω u）。取 q = √2 Re u、p = −√2 Im u，
# 再按 qᵀΩp = 1 归一化（该构造满足 hq = ωΩp、hp = −ωΩq，从而 SᵀhS = ωI）。
function _williamson(h::Matrix{Float64})
    n2 = size(h, 1)
    m = n2 ÷ 2
    Ω = _omega(m)
    A = sqrt(Symmetric(h))                            # h^{1/2}
    E = eigen(Hermitian(1im .* (A * Ω * A)))
    perm = sortperm(E.values, rev=true)[1:m]          # +ω 半谱
    ω = E.values[perm]
    S = zeros(n2, n2)
    for k in 1:m
        u = A \ E.vectors[:, perm[k]]
        q = sqrt(2) .* real.(u)
        p = -sqrt(2) .* imag.(u)
        μ = dot(q, Ω * p)
        if μ < 0
            p .= -p
            μ = -μ
        end
        μ > 1e-12 || throw(ArgumentError("Williamson 分解退化（h 近奇异）"))
        S[:, 2k-1] .= q ./ sqrt(μ)
        S[:, 2k] .= p ./ sqrt(μ)
    end
    return ω, S
end

# |ρ⟩⟩ = (ρ⊗I)|I⟩ 的 doubled 协方差（4n×4n，坐标序 [R^A; R^B]）。
# 对称纯化形式 |ρ⟩⟩ = (ρ^{1/2}⊗ρ^{1/2})|I⟩ 与 |I⟩ 的 (V⊗Vᵀ) 结构给出：
# 在 Williamson 基 (q_k,p_k) 下 doubled 态 = ⊗_k 双模压缩对（tanh r_k = e^{−ω_k}），
# 但 copy-B 是**转置表示**（pᵀ = −p），对应 B 拷贝坐标的 p 反号 Z = diag(1,−1,…,1,−1)：
#   单侧块 X = S (⊕_k coth(ω_k)/2 · I₂) Sᵀ            （= ρ² 的协方差）
#   交叉块 = S (⊕_k csch(ω_k)/2 · I₂) Sᵀ · Z           （Williamson 基下全 + 对角）
#   B 拷贝同侧块 = Z · X · Z
# 结果是 doubled 空间的**纯**高斯态（辛本征值全为 ½；块结构经 dense |ρ⟩⟩ 逐元素仲裁）。
function _bosonic_doubled_covariance(h::Matrix{Float64})
    m = size(h, 1) ÷ 2
    ω, S = _williamson(h)
    X₀ = zeros(2m, 2m)
    Y₀ = zeros(2m, 2m)
    for k in 1:m
        X₀[2k-1, 2k-1] = X₀[2k, 2k] = 0.5 * coth(ω[k])
        Y₀[2k-1, 2k-1] = Y₀[2k, 2k] = 0.5 / sinh(ω[k])
    end
    X = S * X₀ * transpose(S)
    Y = S * Y₀ * transpose(S)
    Z = Matrix{Float64}(I, 2m, 2m)
    for k in 1:m
        Z[2k, 2k] = -1.0
    end
    σd = zeros(4m, 4m)
    σd[1:2m, 1:2m] = X
    σd[2m+1:4m, 2m+1:4m] = Z * X * Z
    σd[1:2m, 2m+1:4m] = Y * Z
    σd[2m+1:4m, 1:2m] = Z * Y
    return (σd + σd') ./ 2
end

# ========================================================================== #
# osee（协方差快速算法）
# ========================================================================== #

"""
    osee(s::BosonicGaussianState, sites::AbstractVector{<:Integer}; α=1) -> Float64

玻色高斯态的 OSEE（operator-space entanglement entropy），复杂度
O(模式数³)，不构造稠密矩阵，也无 Fock 截断误差。

- **Gibbs 分支**（`isbdg = true`）：把 ρ 向量化为 doubled (ket,bra) 空间的
  纯高斯态 `|ρ⟩⟩ = (ρ⊗I)|I⟩`（Williamson 分解 + 双模压缩对解析构造，惰性缓存）。
  `sites` 是 **doubled 纯态空间**的模式指标：`1..n` 为 ket 拷贝、`n+1..2n`
  为 bra 拷贝；物理 site p 的割为 `[p, n+p]`。
- **纯配对分支**（`isbdg = false`）：|Ψ⟩ = exp(½c†Wc†)|0⟩ 本身就是 doubled
  空间的纯向量，`sites` 直接是其 2n 个模的指标（IF 系数态的 site p =
  模式对 `[p, n+p]`），取 Γ 限制的辛本征值熵。

`α = 1` 为 von Neumann 熵，`α ≠ 1` 为 Rényi 熵。
"""
function osee(s::BosonicGaussianState, sites::AbstractVector{<:Integer}; α::Real=1)
    α > 0 || throw(ArgumentError("Rényi 阶数 α 必须为正"))
    allunique(sites) || throw(ArgumentError("sites 含重复指标"))
    Γd = _gammad(s)
    m = size(Γd, 1) ÷ 2
    all(1 .≤ sites .≤ m) || throw(BoundsError(1:m, sites))
    isempty(sites) && return 0.0
    idx = _siteindices(sort(collect(Int, sites)))
    Ωd = _omega(m)
    return _entropy_bosonic(Γd[idx, idx], Ωd[idx, idx], α)
end

# ========================================================================== #
# 稠密构造（仅供 densitymatrix 与测试对照；osee 永不使用）
# ========================================================================== #

function _single_mode_a(nbas::Int)
    d = nbas + 1
    a = zeros(Float64, d, d)
    for n in 1:nbas
        a[n, n + 1] = sqrt(n)
    end
    return a
end

function _embed_operator(op::AbstractMatrix, m::Int, nm::Int, d::Int)
    # 模式 1 = 最低张量因子（与 JW 的 mode i ↔ bit i−1 约定一致）
    ops = [i == m ? op : Matrix{Float64}(I, d, d) for i in 1:nm]
    return reduce(kron, reverse(ops))
end

function _bosonic_dense_rho(bdg::AbstractMatrix; nbas::Int=16)
    nm = size(bdg, 1) ÷ 2
    d = nbas + 1
    a = _single_mode_a(nbas)
    ad = transpose(a)
    xm = ComplexF64.((a .+ ad) ./ sqrt(2))
    pm = ComplexF64.(-1im .* (a .- ad) ./ sqrt(2))
    R = Vector{Matrix{ComplexF64}}(undef, 2nm)
    for m in 1:nm
        R[2m-1] = _embed_operator(xm, m, nm, d)
        R[2m]   = _embed_operator(pm, m, nm, d)
    end
    dim = d^nm
    H = zeros(ComplexF64, dim, dim)
    for i in 1:2nm, j in 1:2nm
        iszero(bdg[i, j]) && continue
        H .+= (bdg[i, j] / 2) .* (R[i] * R[j])
    end
    H = (H + H') ./ 2
    E = eigen(Hermitian(H))
    w = exp.(-E.values); w ./= sum(w)
    return E.vectors * Diagonal(w) * E.vectors', d
end

"""
    densitymatrix(s::BosonicGaussianState; d=5) -> Matrix{ComplexF64}

稠密密度矩阵（Fock 基截断，调试/对照专用）：每模保留 `d` 个 Fock 基矢
（|0⟩…|d−1⟩）。

- Gibbs 分支：返回 (d^n × d^n) 的 ρ = exp(−½Rᵀ·bdg·R)/Z；
- 纯配对分支：返回 (d^{2n} × d^{2n}) 的 |Ψ⟩⟨Ψ|（|Ψ⟩ = exp(½c†Wc†)|0⟩，
  doubled 2n 个模上的截断向量）。

注意这是**截断近似**：结果随 `d` 收敛（权重衰减慢时需更大的 `d`）；
精确量请用 `entropy` / `osee`（协方差算法，无截断）。
"""
function densitymatrix(s::BosonicGaussianState; d::Int=5)
    d ≥ 2 || throw(ArgumentError("Fock 截断维数 d 必须 ≥ 2"))
    W = _realify("bosonic bdg", s.bdg)
    if s.isbdg
        ρ, _ = _bosonic_dense_rho(W; nbas=d - 1)
        return ρ
    end
    # 纯配对：|Ψ⟩ = exp(½ Σ W_ij c†_i c†_j)|0⟩（截断生成元的矩阵指数）
    N = size(W, 1)                                   # 2n 个模
    a = _single_mode_a(d - 1)
    ad = transpose(a)
    dim = d^N
    Gd = zeros(ComplexF64, dim, dim)
    for i in 1:N, j in 1:N
        iszero(W[i, j]) && continue
        Gd .+= (W[i, j] / 2) .* (_embed_operator(ComplexF64.(ad), i, N, d) *
                                 _embed_operator(ComplexF64.(ad), j, N, d))
    end
    e0 = zeros(ComplexF64, dim); e0[1] = 1
    ψ = exp(Gd) * e0
    ψ ./= norm(ψ)
    return ψ * ψ'
end
