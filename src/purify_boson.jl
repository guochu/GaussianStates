# ============================================================================
# purify_boson.jl —— 玻色 Gibbs 态 / purification
#
# Gibbs 协方差：伴随生成元 𝒜 = [[h, A], [-Ā, -hᵀ]]（[H,Ψ] = -𝒜Ψ，厄米类
#   B=A†）本征值成 ± 对；负支 α 的 Bogoliubov 模热权重 ν = ½coth(-α/2)，
#   C = Σ ν (T₋T₋† + T₊T₊†)（σ₃-正交基，对简并谱按子空间 σ₃-Gram 组装）。
#   失稳/无界由 σ₃-Gram 符号检查与 α<0 要求拒绝（谱全实本身不保证稳定）。
#
# purify：湮灭子 V = E − exp(𝒜)·Pmᵀ（一般 𝒜 = [[h,Aᵀ],[−Bᵀ,−hᵀ]]；
#   镜像 M(aᵢ)=+dᵢ†、M(aᵢ†)=+dᵢ；A、B 复对称 ⟹ 𝒜 ∈ sp ⟹ V 迷向
#   ⟹ Choi 态高斯）；协方差由 kernel(Σ − (i/2)Ω) ∋ rowspace(V) 的线性
#   系统补全，对 n ≥ 1、任意（非）厄米配对成立。n=1 与稠密 Fock Choi
#   对拍至机器精度，n=2 与独立 Williamson 构造逐元素一致。不可归一化
#   时补全结果不正定，抛 ArgumentError。
# ============================================================================

# 𝒜 的本征值按实部分组；返回 [(α, V)]，V 列张成该本征值子空间
function _eig_groups(𝒜::AbstractMatrix; tol=1e-8)
    F = eigen(𝒜)
    scale = max(1.0, opnorm(𝒜, 1))
    maximum(abs.(imag.(F.values))) < 1e-9 * scale ||
        throw(ArgumentError("玻色 Gibbs 态不存在：伴随生成元谱非实（H 无下界/不稳定）"))
    vals = real.(F.values)
    order = sortperm(vals)
    groups = Tuple{Float64,Matrix{ComplexF64}}[]
    i = 1
    m = length(vals)
    while i ≤ m
        j = i
        while j < m && abs(vals[order[j+1]] - vals[order[i]]) < tol * scale
            j += 1
        end
        α = sum(vals[order[i:j]]) / (j - i + 1)
        V = F.vectors[:, order[i:j]]
        push!(groups, (α, V))
        i = j + 1
    end
    return groups
end

function _boson_gibbs_C(h::AbstractMatrix, A::AbstractMatrix)
    n = size(h, 1)
    𝒜 = [h A; -conj(A) -transpose(h)]
    σ = [Matrix{ComplexF64}(I, n, n) zeros(ComplexF64, n, n);
         zeros(ComplexF64, n, n) -Matrix{ComplexF64}(I, n, n)]
    groups = _eig_groups(𝒜)
    length(groups) == 2n || throw(ArgumentError("玻色 Gibbs 态：本征值分组异常"))
    C = zeros(ComplexF64, 2n, 2n)
    used = falses(length(groups))
    for (gi, (α, V)) in enumerate(groups)
        used[gi] && continue
        α < -1e-10 * max(1.0, opnorm(𝒜, 1)) ||
            throw(ArgumentError(α > 1e-10 * max(1.0, opnorm(𝒜, 1)) ?
                "玻色 Gibbs 态不可归一化（𝒜 有正本征值 $(α)：e^H 无界；热态应传 H = -β·H_sys）" :
                "玻色 Gibbs 态存在零模（𝒜 本征值 0），协方差发散；请加质量/截断"))
        gj = findfirst(k -> !used[k] && abs(groups[k][1] + α) < 1e-8 * max(1.0, opnorm(𝒜, 1)), 1:length(groups))
        gj === nothing && throw(ArgumentError("玻色 Gibbs 态：本征值 ± 配对失败"))
        used[gj] = used[gi] = true
        β, Ṽ = groups[gj]
        # σ₃-Gram：负支子空间须正定、正支子空间须负定（失稳/无界时翻转）
        G = V' * σ * V
        maximum(real.(eigvals(Hermitian(G)))) > 1e-10 ||
            throw(ArgumentError("玻色 Gibbs 态不可归一化（σ₃ 符号结构异常：e^H 无界/失稳）"))
        Gb = Ṽ' * σ * Ṽ
        minimum(real.(eigvals(Hermitian(Gb)))) < -1e-10 ||
            throw(ArgumentError("玻色 Gibbs 态不可归一化（σ₃ 符号结构异常：e^H 无界/失稳）"))
        ν = 0.5 / tanh(-α / 2)
        C .+= ν .* (V * (Hermitian(G) \ V') + Ṽ * ((-Hermitian(Gb)) \ Ṽ'))
    end
    return (C .+ C') ./ 2
end

"""
    GaussianState(gs::BosonicQuadraticGibbsState) -> BosonicGaussianState

厄米 `e^H` 的 Gibbs 态（n 模，混合）。Bogoliubov 组装协方差，Σ = to_xp(C)。
"""
function GaussianState(gs::BosonicQuadraticGibbsState)
    return BosonicGaussianState(to_xp(_boson_gibbs_C(gs.op.h, gs.op.A)))
end

"""
    purify(e::BosonicQuadraticOperatorExp) -> BosonicGaussianState

一般（可非厄米）`e^H` 的 Choi 矢量化态 `|ρ⟩⟩ = (ρ⊗I)|I⟩_B`：2n 模纯态，
ket 模 `1..n`、bra/净化模 `n+1..2n`，复杂度 O(n³)。对任意 n、任意
（非）厄米配对恒为高斯。

## 数学构造

**矢量化与 EPR 参考态。** `|I⟩_B = exp(Σᵢ aᵢ†dᵢ†)|0⟩`（单模即
`Σₖ|k,k⟩`）本身**不可归一化**，但镜像恒等式是它承受的算符作用恒等式，
逐字成立：

```
aᵢ|I⟩_B = dᵢ†|I⟩_B ,    aᵢ†|I⟩_B = dᵢ|I⟩_B
```

（玻色无费米符号；费米侧 `c†` 的镜像带负号。）线性镜像
`M(aᵢ) = +dᵢ†`、`M(aᵢ†) = +dᵢ`。`(ρ⊗I)|I⟩_B` 的可归一化由 `e^H`
的衰减补上，其充要体现是下述协方差补全的正定性。

**湮灭子。** 玻色生成元带符号约定 `[H,Ψ] = −𝒜Ψ`，

```
𝒜 = [[h, Aᵀ], [−Bᵀ, −hᵀ]] ,    e^{−H}Ψe^H = exp(𝒜)Ψ ≡ WΨ .
```

与费米完全同构的链式 `Ψ_r|ρ⟩⟩ = e^H·O_r|I⟩_B = M(O_r)|ρ⟩⟩`
（`O_r = ΣₖW_{rk}Ψₖ`，末步用 ancilla 与 `e^H` 对易、镜像恒等式）给出
2n 个湮灭子，按 `(a,a†,d,d†)` 序排成 `V = E − W·Pmᵀ`。

**V 迷向 ⟹ 恒高斯。** A、B 复对称（构造器强制）⟹ `𝒜 ∈ sp(2n)`（块形
`[[K,L],[M,−Kᵀ]]`，L、M 对称）⟹ `W` 辛。镜像把 Nambu 交换形式
`J_{ab} = [Ψ_a,Ψ_b]` 变号（`(a,a†)` 换成 `(d†,d)`），于是

```
[V_r, V_s] = J_{rs} − (WJWᵀ)_{rs} = 0 .
```

2n 个两两对易的 Nambu 线性算符湮灭的态是（可能奇异的）高斯态；
可归一化时即正规高斯纯态。

**协方差补全。** 湮灭子 `A = w·R`：`R` 为倍增 (x,p) 四极算符
（`R = (x₁,p₁,…,xₙ,pₙ | x'₁,p'₁,…)`），`w` 为 `V` 的四极协变式行。
由算符代数 `½{A, R_a} − ½[A, R_a] = R_aA`、对易关系
`[R_a, R_b] = iΩ_ab`（`Ω` 取倍增 4n×4n）及 `⟨φ|R_aA|φ⟩ = 0`，得纯态
约束的左核形式

```
w(Σ − (i/2)Ω) = 0 ,    w̄(Σ + (i/2)Ω) = 0 .
```

湮灭子行 `w` 与其复共轭合起来给 4n 个独立约束，解线性系统
`[w; w̄]·Σt = [(i/2)wΩ; −(i/2)w̄Ω]` 唯一确定 `Σ`。解为实对称；
实对称**正定** ⟺ `(ρ⊗I)|I⟩_B` 可归一化。不正定（`e^H` 含增长方向、
压缩参量 `|tanh r| → 1` 等失稳/无界情形）或约束奇异时抛
`ArgumentError`。

Gibbs 态的两个入口：
[`GaussianState(::BosonicQuadraticGibbsState)`](@ref)（混合态）与
[`purify(::BosonicQuadraticGibbsState)`](@ref)（TFD）。

# 例

```julia
n = 2
A0 = 0.5 .* randn(ComplexF64, n, n)
A = (A0 + transpose(A0)) / 2                     # 复对称配对
opb  = exp(BosonicQuadraticOperator(-3.0 * Matrix(1.0I, n, n), A, A))
psib = purify(opb)                               # 2n 模纯 Choi 态
@assert ispure(psib)
```
"""
function purify(e::BosonicQuadraticOperatorExp)
    h, A, B = e.op.h, e.op.A, e.op.B
    n = size(h, 1)
    𝒜 = [h transpose(A); -transpose(B) -transpose(h)]
    W = exp(𝒜)
    Pm = zeros(ComplexF64, 4n, 2n)
    for i in 1:n
        Pm[3n+i, i] = 1.0            # M(aᵢ)  = +dᵢ†
        Pm[2n+i, n+i] = 1.0          # M(aᵢ†) = +dᵢ
    end
    E = zeros(ComplexF64, 2n, 4n)
    E[1:2n, 1:2n] = Matrix{ComplexF64}(I, 2n, 2n)
    V = E - W * transpose(Pm)
    # Nambu 系数 → 四极协变式（交错 x,p；ket 模 1..n 前、bra 模 n+1..2n 后）
    Wq = zeros(ComplexF64, 2n, 4n)
    for j in 1:n, r in 1:2n
        Wq[r, 2j-1]     = (V[r, j]   + V[r, n+j])   / sqrt(2)
        Wq[r, 2j]       = im * (V[r, j]   - V[r, n+j])   / sqrt(2)
        Wq[r, 2n+2j-1]  = (V[r, 2n+j] + V[r, 3n+j]) / sqrt(2)
        Wq[r, 2n+2j]    = im * (V[r, 2n+j] - V[r, 3n+j]) / sqrt(2)
    end
    Ω = _omega(2n)
    G = [Wq; conj(Wq)]
    rhs = [(im / 2) * Wq * Ω; -(im / 2) * conj(Wq) * Ω]
    Σt = try
        G \ rhs
    catch err
        isa(err, SingularException) && throw(ArgumentError(
            "玻色一般 purify：湮灭子协变式线性相关（Choi 态不可归一化？）"))
        rethrow()
    end
    Σ = real.((Σt .+ Σt') ./ 2)
    return BosonicGaussianState(Σ)
end

"""
    purify(gs::BosonicQuadraticGibbsState) -> BosonicGaussianState

厄米 `e^H` Gibbs 态 `ρ = e^H/Z` 的 thermofield double（TFD）：`√ρ ∝ e^{H/2}`
的 Choi 矢量化。内部把 `gs.op/2` 经 `.B = A†` 合成转成一般指数，走
[`purify(::BosonicQuadraticOperatorExp)`](@ref) 的同一条 O(n³) 路径；
2n 模纯态。能量本征基下 `|TFD⟩ ∝ Σₖ e^{εₖ/2}|k⟩⊗|k⟩`（传
`H = −βH_sys` 时即通常的 `e^{−βεₖ/2}` 权重），且

```
partialtr(purify(gs), 1:n) == GaussianState(gs)
```

（测试固化）。

**两种矢量化语义，勿混**：`purify(gs)` 的纠缠 = Gibbs **态**的热纠缠
（= `entropy(GaussianState(gs))`）；Gibbs **密度矩阵在算符空间**的
OSEE 要用完整 `e^H` 的 Choi：
`purify(exp(BosonicQuadraticOperator(gs.op.h, gs.op.A, gs.op.B)))`。
可归一化的拒绝机制与一般情形相同：协方差补全结果不正定即抛
`ArgumentError`（见该方法的数学说明）。
"""
purify(gs::BosonicQuadraticGibbsState) = begin
    H2 = gs.op / 2
    purify(BosonicQuadraticOperatorExp(BosonicQuadraticOperator(
        H2.h, H2.A, H2.B; c0=H2.c0)))
end
