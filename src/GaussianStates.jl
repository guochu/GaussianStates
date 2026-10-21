# ============================================================================
# GaussianStates.jl
#
# 独立的高斯态纯数学工具包（无任何杂质物理依赖）：
#   - 玻色 / 费米高斯态的表示（BdG 矩阵 bdg 与协方差 Γ）
#   - 子系统 von Neumann / Rényi 熵与 OSEE（osee，全部为协方差快速算法）
#   - 稠密密度矩阵（densitymatrix，调试/对照用；osee 永不回退到稠密实现）
#
# 文件组织：
#   common.jl     抽象类型、访问器、指标工具、熵内核（费米/玻色通用）
#   fermionic.jl  费米高斯态：类型、构造、BdG 矩阵、协方差（含 doubled）、
#                 osee、densitymatrix
#   bosonic.jl    玻色高斯态：类型、构造、BdG 矩阵、Williamson 分解、
#                 doubled 协方差、osee、densitymatrix
#   entropy.jl    entropy（偏迹子系统熵）泛型接口
#
# 术语（统一约定）：
#   BdG 矩阵按 n×n 分块，h = 左上块，g = 右上块。它是二次型哈密顿量的系数矩阵
#   （已吸收逆温度），态为 ρ ∝ exp(−½ ψ†·bdg·ψ)。
#   由于 Nambu 自旋量 ψ=(c,c†) 的反对易关系使右下块可并入左上块（只差常数），
#   物理上只有**两个独立块**，标准规范取：
#       费米（厄米）  ：⎡ h   A ⎤       A = g − gᵀ（g 只有反对称部分有效）
#                     ⎣ A† −hᵀ⎦
#       玻色（Bogoliubov 块 → 实 (x,p) K）：
#       H = Σ h a†a + ½(a†ga† + h.c.)，h 复厄米、g 复任意；
#       Kxx = Re(h)+Re(g_s)，Kpp = Re(h)−Re(g_s)，Kxp = 2(Im(g_s)−Im(h))。
#       （任何厄米二次型在 (x,p) 基下 K 恒为实对称——R 算符是实矩阵。）
#   组装用 `fermionic_bdg_matrix(h[, g])` / `bosonic_bdg_matrix(h[, g])`。
#   费米与玻色都另支持**纯配对**输入（费米：复反对称 W；玻色：实对称、
#   谱 ⊂ (−1,1) 的非正定 W），表示 doubled 空间的纯高斯向量
#   |Ψ⟩ = exp(½c†Wc†)|0⟩（OSEE/影响泛函系数态用），构造时按谱/对称性判别
#   一次并记录在字段 isbdg（true = 密度矩阵形态，entropy 仅此形态有定义）。
#   协方差经 `s.Γ` 字段访问（covariance 不导出，惰性计算后缓存于字段）。
#
# OSEE 的快速算法（对**任何**分支都不构造稠密矩阵）：
#   把 ρ 向量化为 doubled (ket,bra) 空间的纯态 |ρ⟩⟩ = (ρ⊗I)|I⟩。对高斯 ρ 它是
#   **纯高斯态**，其 doubled 协方差 Γd 可由 BdG 数据解析组装：
#     - 费米 Gibbs：BdG 本征分解 M u_k = λ_k u_k（λ>0 半谱），伴模 f_k† ↔ 列
#       S₀·conj(u_k)（粒子–空穴对称 M S₀ = −S₀ M̄，fermionic_bdg_matrix 保证）。
#       |ρ⟩⟩ = Π_k (|00⟩−t_k|11⟩)_{(f_k^A, f̄*_k^B)}·(Bogoliubov 旋转)，t_k = e^{−λ_k}。
#     - 玻色：Williamson 分解，|ρ⟩⟩ = Π_k 双模压缩对（tanh r_k = e^{−ω_k}），
#       copy-B 为转置表示（pᵀ = −p），交叉块带 p 反号 Z 修正。
#   OSEE = Γd 限制到给定模式子集的标准协方差熵。
#
# 数值公式来源（均已用稠密态逐元素仲裁验证，见 GTEMPO/osee/osee_report.md
# 与本包 test/runtests.jl 的稠密对照测试）：
#   - 费米纯 ā–a 配对：SVD 闭式（osee_matsubara.jl 的 gauss_correlations）
#   - 费米一般复反对称：混合共轭 Stein 方程（osee_bcs.jl 的 gauss_correlations_W）
#   - Majorana 协方差组装（osee_matsubara.jl 的 majorana_covariance）
#
# 设计约定见 GTEMPO/osee/quapi_osee_api_plan.md。
# ============================================================================

"""
    GaussianStates

高斯态纯数学工具包。

- [`FermionicGaussianState`](@ref)：BdG 矩阵 `bdg` 是二次型哈密顿量的系数
  （`ρ ∝ exp(−½ψ†·bdg·ψ)`，一般混合态）；也支持复反对称的纯配对输入
  `|Ψ⟩ = exp(½c†·bdg·c†)|0⟩`（OSEE 用）。协方差 `Γ` 为 Majorana 基下的实
  反对称矩阵（`Γ_ab = i⟨γ_aγ_b⟩`，`γ_{2m−1} = c_m + c†_m`，
  `γ_{2m} = i(c†_m − c_m)`）。
- [`BosonicGaussianState`](@ref)：BdG 输入为 2n×2n 实对称矩阵（正则坐标序
  `R = (x₁,p₁,x₂,p₂,…)`）：正定 ⟺ Gibbs 密度矩阵形态（`ρ ∝ exp(−½ Rᵀ·bdg·R)`，
  协方差 `Γ` 为实对称矩阵，`σ_ab = ½⟨{R_a,R_b}⟩`，`Ω = ⊕ₖ [[0,1],[−1,0]]`）；
  非正定且谱 ⊂ (−1,1) ⟺ 纯配对向量 exp(½c†Wc†)|0⟩（IF 系数态）。
  复系数的 Bogoliubov 块经 [`bosonic_bdg_matrix`](@ref) 归一为实矩阵。
  协方差不经导出函数访问——惰性计算后直接读字段 `s.Γ`。
- [`osee`](@ref)：operator-space von Neumann（`α=1`）/ Rényi（`α≠1`）熵，
  全部由协方差矩阵快速计算（O(模式数³)），不构造稠密矩阵。对任何形态的输入
  （密度矩阵或 exp(非厄米二次型) 算符）都有定义。
- [`entropy`](@ref)：子系统（偏迹）熵，同样由协方差快速计算；只对密度矩阵
  形态（`isbdg = true`）有定义，否则报错。
- [`densitymatrix`](@ref)：稠密密度矩阵（调试/对照专用）。

熵公式（`Γ` 限制到 `left` 的行列指标）：

| 类型 | Γ | 谱 | von Neumann 熵 |
|---|---|---|---|
| 费米 | 实反对称 | `eigvals(iΓ_A) = ±ν`，\\|ν\\| ≤ 1 | `Σₖ h((1+νₖ)/2)` |
| 玻色 | 实对称 | `eigvals(Ω_A Γ_A) = ±iν`，ν ≥ ½ | `Σₖ [(ν₊½)ln(ν₊½) − (ν₋½)ln(ν₋½)]` |
"""
module GaussianStates

using LinearAlgebra

export AbstractGaussianState, BosonicGaussianState, FermionicGaussianState,
       osee, entropy, densitymatrix, nmodes,
       bosonic_bdg_matrix, fermionic_bdg_matrix

include("common.jl")
include("fermionic.jl")
include("bosonic.jl")
include("entropy.jl")

end # module
