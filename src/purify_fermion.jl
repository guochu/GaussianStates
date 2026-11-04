# ============================================================================
# purify_fermion.jl —— 费米 purifications
#
# 一般非厄米 e^H：Choi/IF 态 |ρ⟩⟩ = (ρ⊗I)|I⟩_G，|I⟩_G = exp(Σ c†d†)|0⟩。
#   O(n³) 解析构造：e^{-H}Ψe^{H} 的伴随作用矩阵是 W = exp(bdg(op))；
#   |I⟩_G 的镜像恒等式 cᵢ|I⟩=dᵢ†|I⟩、cᵢ†|I⟩=−dᵢ|I⟩ 给出镜像 Pm；
#   湮灭子 V = E − W·Pmᵀ（2n×4n），协方差 = 湮灭子行空间投影子的余
#   （4n×4n）。与稠密零空间版逐元素一致（n=1..4 验证到机器精度）。
# 厄米 e^H：Gibbs 态 C = (I + e^X)^{-1}；TFD = purify(一般 exp 的 e^{H/2})
# ============================================================================

function _mirror_matrix(n::Int)
    # EPR 镜像 M：cᵢ ↦ dᵢ†，cᵢ† ↦ −dᵢ。
    # Pm 的第 k 列 = M(Ψ_k) 在 (c, c†, d, d†) 序下的坐标（2n 个 Nambu 元 → 4n）。
    Pm = zeros(ComplexF64, 4n, 2n)
    for i in 1:n
        Pm[3n+i, i] = 1.0        # M(cᵢ)  = +dᵢ†
        Pm[2n+i, n+i] = -1.0     # M(cᵢ†) = −dᵢ
    end
    return Pm
end

# (c,c†,d,d†) → (c,d,c†,d†) 的重排（to_majorana 的输入序）
function _nambu_reorder(n::Int)
    Po = zeros(ComplexF64, 4n, 4n)
    for i in 1:n
        Po[i, i] = 1.0
        Po[n+i, 2n+i] = 1.0
        Po[2n+i, n+i] = 1.0
        Po[3n+i, 3n+i] = 1.0
    end
    return Po
end

"""
    purify(e::FermionicQuadraticOperatorExp) -> FermionicGaussianState

一般（可非厄米）`e^H` 的 Choi 矢量化态 `|ρ⟩⟩ = (ρ⊗I)|I⟩_G`：2n 模纯态，
ket 模 `1..n`、bra/净化模 `n+1..2n`。复杂度 O(n³)。这是 IF/GMPS 系数态
与算符空间 OSEE 所用的矢量化。

## 数学构造

**矢量化与 EPR 参考态。** 倍增 Fock 空间上取

```
|I⟩_G = exp(Σᵢ cᵢ†dᵢ†)|0⟩ = ⊗ᵢ(|00⟩ + |11⟩)
```

（费米求和有限、已归一）。`|ρ⟩⟩` 对 bra 模偏迹还原 ρ，差一个标量；
协方差与归一化无关，故 ρ 不必归一化。

**镜像恒等式。** 作为对 `|I⟩_G` 作用的算符恒等式：

```
cᵢ|I⟩_G = dᵢ†|I⟩_G ,    cᵢ†|I⟩_G = −dᵢ|I⟩_G
```

由此定义线性镜像 `M(cᵢ) = +dᵢ†`、`M(cᵢ†) = −dᵢ`（只定义在 ket 算符的
线性张成上，不是代数同态）。

**湮灭子。** 记 Nambu 向量 `Ψ = (c, c†)`；伴随作用矩阵就是 BdG 指数，
`e^{-H}Ψe^H = exp(bdg(H))Ψ ≡ WΨ`。对 `W` 第 r 行给 `O_r = ΣₖW_{rk}Ψₖ`，
链式

```
Ψ_r|ρ⟩⟩ = e^H·(O_r|I⟩_G) = e^H·(M(O_r)|I⟩_G) = M(O_r)|ρ⟩⟩
```

（第一步 `Xe^H = e^H·e^{-H}Xe^H`，第二步镜像恒等式，第三步 ancilla
算符与 `e^H` 对易）给出 2n 个独立湮灭子，按 `(c,c†,d,d†)` 序排成

```
V = E − W·Pmᵀ
```

`E = [I₂ₙ | 0]` 取系统 Nambu 基，`Pm` 的第 k 列是 `M(Ψₖ)` 的坐标。
纯高斯态的湮灭子空间恰为 2n 维，故 `V` 行满秩。

**协方差 = 湮灭子行空间投影子。** 设 `A = QΦ` 湮灭态，`Φ` 为
`(c,c†,d,d†)` 的 4n 维 Nambu 向量（`{Φ_a, Φ_b†} = δ_ab`）。由
`⟨φ|A_k†A_l|φ⟩ = 0` 与 `Φ_a†Φ_b = δ_ab − Φ_bΦ_a†` 得
`Q̄(I − C)Qᵀ = 0`（`C` 为 Nambu 协方差 `⟨ΦΦ†⟩`）；`Q` 行满秩给
`(I − C)Qᵀ = 0`，而 `C` 厄米、纯态秩 2n，故 `C` 是湮灭子系数行空间上
的正交投影子。数值上取 `V†` 的 QR 正交基 `B`、`C = BB†`（避免
`(VV†)⁻¹` 的平方条件数），经 `(c,c†,d,d†) → (c,d,c†,d†)` 重排后按
`Ψ = (c,d)` 转 Majorana。

厄米 Gibbs 态的两个入口：
[`GaussianState(::FermionicQuadraticGibbsState)`](@ref)（混合态）与
[`purify(::FermionicQuadraticGibbsState)`](@ref)（TFD）。

# 例

```julia
n = 4
op  = exp(FermionicQuadraticOperator(h, A, B))   # 一般非厄米 e^H
psi = purify(op)                                 # |ρ⟩⟩，2n 模纯态
@assert ispure(psi)
S = entropy(psi, [1, n + 1])                     # 站点 1 的 OSEE
```
"""
function purify(e::FermionicQuadraticOperatorExp)
    n = nmodes(e.op)
    W = exp(bdg(e.op))
    Pm = _mirror_matrix(n)
    E = zeros(ComplexF64, 2n, 4n)
    E[1:2n, 1:2n] = Matrix{ComplexF64}(I, 2n, 2n)
    V = E - W * transpose(Pm)
    # 湮灭子行空间投影子：用 QR 正交基构造，避免 (VV†)^{-1} 的平方条件数
    B = Matrix(qr(adjoint(V)).Q)         # 4n×2n，B†B = I，列空间 = 行空间(V)
    C = B * adjoint(B)
    Po = _nambu_reorder(n)
    Γ = to_majorana(Po * C * adjoint(Po))
    return FermionicGaussianState(Γ)
end

"""
    covariance(gs::FermionicQuadraticGibbsState) -> Matrix{Float64}

厄米 `e^H` Gibbs 态的实 Majorana 协方差（n 模，混合）：本征基下
Nambu 协方差 `C = (I + e^X)^{-1}`（`X = bdg(op)`，Nambu 表象）经
`to_majorana` 转到 **Majorana 表象**输出（与
`covariance(::FermionicGaussianState)` 同表象；非 `bdg` 的 Nambu 表象）。
不做物理性校验（校验在 `FermionicGaussianState` 构造器里）。
"""
function covariance(gs::FermionicQuadraticGibbsState)
    X = bdg(gs.op)
    F = eigen(Hermitian(X))
    C = F.vectors * Diagonal(1 ./ (1 .+ exp.(F.values))) * adjoint(F.vectors)
    return to_majorana(C)
end

"""
    FermionicGaussianState(gs::FermionicQuadraticGibbsState)

厄米 `e^H` 的 Gibbs 态（n 模，混合）；等价于 `GaussianState(gs)`。
"""
FermionicGaussianState(gs::FermionicQuadraticGibbsState) =
    FermionicGaussianState(covariance(gs))

"""
    GaussianState(gs::FermionicQuadraticGibbsState) -> FermionicGaussianState

厄米 `e^H` 的 Gibbs 态（n 模，混合）。
"""
GaussianState(gs::FermionicQuadraticGibbsState) = FermionicGaussianState(gs)

"""
    purify(gs::FermionicQuadraticGibbsState) -> FermionicGaussianState

厄米 `e^H` Gibbs 态 `ρ = e^H/Z` 的 thermofield double（TFD）：`√ρ ∝ e^{H/2}`
的 Choi 矢量化。内部把 `gs.op/2` 经 `.B = A†` 合成转成一般指数，走
[`purify(::FermionicQuadraticOperatorExp)`](@ref) 的同一条 O(n³) 路径；
2n 模纯态。

能量本征基（`H|k⟩ = εₖ|k⟩`）下 `|TFD⟩ ∝ Σₖ e^{εₖ/2}|k⟩⊗|k⟩`（传
`H = −βH_sys` 时即通常的 `e^{−βεₖ/2}` 权重），系统偏迹还原 Gibbs 态：

```
partialtr(purify(gs), 1:n) == GaussianState(gs)
```

（测试固化）。

**两种矢量化语义，勿混**：

- `purify(gs)`（本方法，TFD）：纠缠 = Gibbs **态**的热纠缠，
  `entropy(partialtr(purify(gs), 1:n)) == entropy(GaussianState(gs))`。
- `purify(exp(FermionicQuadraticOperator(gs.op.h, gs.op.A, gs.op.B)))`：
  完整 `e^H` 的 Choi 态。纠缠 = Gibbs **密度矩阵在算符空间**的 OSEE
  （IF 系数态语义）。

厄米算子的 `.B` 经 `getproperty` 合成为 `A†`，第二种写法可直接构造。
"""
purify(gs::FermionicQuadraticGibbsState) = begin
    H2 = gs.op / 2                   # √ρ = e^{H/2}，经一般 exp 走 Choi 路径
    purify(FermionicQuadraticOperatorExp(FermionicQuadraticOperator(
        H2.h, H2.A, H2.B; c0=H2.c0)))
end
