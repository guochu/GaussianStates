# v7 实现状态

## 2026-10-29 会话（三）：接口收紧重构

- **算子类型参数化**：`AbstractQuadraticOperator{T<:Number}` 及全部
  继承类（`*Exp` / `*GibbsState` 同步参数化）；构造器自动提升元素类型，
  实数输入走实数路径（`{Float64}`）。`Fermionic/BosonicGaussianState`
  保持非参数（协方差恒为实 `Float64`）。
- **删除访问器函数**：`constant` / `_h` / `op` / `blocks` 全部移除，
  直接 `.c0` / `.h` / `.A` / `.B` 访问字段；厄米算子重载
  `Base.getproperty` 使 `.B` 合成为 `A†`（含 `propertynames`），
  与一般算子接口对齐——`FermionicQuadraticOperator(g.h, g.A, g.B)`
  可直接把 Gibbs 态转一般 exp。
- **删除 `asoperator`**：`purify(gs)` 内联 `gs.op/2` 经 `.B` 合成构造
  一般 exp；算符空间 OSEE 的写法改为
  `purify(exp(FermionicQuadraticOperator(gs.op.h, gs.op.A, gs.op.B)))`。
- **删除 `_gaussian_state`**：物理性校验移入
  `FermionicGaussianState(Γ)` / `BosonicGaussianState(Σ)` 的 inner
  constructor（所有构造路径都过校验）；`Val(:raw)` 后门移除。
- **`bdg` 与 `c₀`**：确认无 bug——`c₀` 是单位算符项，`e^{c₀}` 只给
  `e^H` 乘标量，归一化后对协方差/purify/熵无影响；只有
  `densitymatrix`（Fock 矩阵）需要并已计入。docstring 已写明。
- 测试/文档同步（`docs` 重建 0 错误）；**99/99 全绿**，TFD 广谱
  （稳定算符 n=1..8 × 4 seeds）通过。debug/ 下旧仲裁脚本引用已删
  函数，属会话历史记录，未更新。
- 追加（同日）：`fockmatrix`（原 densitymatrix）截断关键字 `nbas → d`；
  构造器对 A、B 的对称性检查（`_check_sym`/`_check_antisym`）改为
  `symmetrize`/`antisymmetrize` 自动投影（任意输入合法，Fock 代数下
  严格等价，测试用 fockmatrix + purify 协方差对拍固化）。

## 2026-10-30 会话（五）：quadraticoperator 反推（协方差 → bdg）

- **费米**（`src/inverse.jl`，已验证）：G=⟨c†c⟩、F=⟨cc⟩ 由 Γ 的直接
  Majorana 公式给出，C = [[I−Gᵀ, F],[−F̄, G]]，**X = log(C⁻¹−I)**（符号
  由 A≠0 稠密 Fock 仲裁钉死）；h=(X₁₁−X₂₂ᵀ)/2、A=antisym(X₁₂)；数值
  厄米化保复厄米 h 的虚部（曾误用 real.() 抹掉 ±0.82i 的虚部）。
  n=1..3 精确恢复 (h,A) + 往返；n=64 恢复误差 4e-14、6.6 ms。
  纯态（|ν|=1）抛 ArgumentError。测试 +8。
- **玻色：未实现**（调用抛 ErrorException），三条路线均不封闭：
  ① Nambu C⁻¹−I 谱双侧跨越；② Williamson 组装 G=S⁻¹(⊕ε)S⁻ᵀ 对本征向量
  相位敏感（n=1 数值证实：e^{iθ} 旋转给不同 G，θ=0 的 LAPACK 默认相位
  恰好正确但无规范依据）；③ ΩG=g(ΩΣ) 不成立（[ΩΣ,ΩG]≠0，本征向量不同）。
  正确构造需辛极分解/复结构路线，作为后续工作包。

## 2026-10-29 会话（四）：boundary-driven OSEE 端到端示例

`docs/examples/boundary_driven_osee.jl`：n=6 链（随机在位+hopping）+ 两端
零温损耗 Lindblad（L = √γ c），混合直积初态；快速路径（双线性仿射 EOM
精确解 Γ(t) → 模算符重建 → purify+entropy）与稠密路径（𝓛 超算符指数 →
FSWAP 交错矢量化 → 偏迹熵）**全程对齐：|ΔOSEE| ≤ 1e-12，|ΔΓ| ≤ 3e-15，
ρ(t) 全程在 e^quadratic 族内 ~1e-16**。过程中钉死的约定（易错点）：

- **费米模重排必须走 FSWAP 网络**（(−1)^{n₁n₂} 符号），朴素张量轴置换错；
  站点割（左 ket+bra 模）在交错序 (c₁d₁c₂d₂…) 下才是连续前缀，
  此约定下 graded |I⟩_G 的 OSEE 才与协方差模式熵相等。
- **⟨c†ᵢcⱼ⟩ = f(h)ᵀ（转置）**：模算符重建 h = W̄εWᵀ（W 为 G 的本征
  向量）；实对称时转置不可见，复厄米（电流相干）时必错。
- G = ⟨c†c⟩ 从 Γ 的直接 Majorana 公式：
  `¼[2δᵢⱼ − i(Γ₂ᵢ₋₁,₂ⱼ₋₁ + Γ₂ᵢ,₂ⱼ) + Γ₂ᵢ₋₁,₂ⱼ + Γ₂ⱼ₋₁,₂ᵢ]`。
- 双线性 EOM 的系数用稠密矩阵一次性数值提取（𝓛† 闭合残差 ~1e-17），
  免手推；增广仿射矩阵右下角必须是 0 不是 1（d/dt 1 = 0）。
- 动力学选型：线性 Lindblad（c、c† 型 jump）+ 二次 H 保持 e^quadratic
  族（本例全时验证）；数守恒型驱动（dephasing n̂）会把态锁在固定 N 扇区，
  混合后离开族——不要用。
- 传播子 e^{𝓛t} 的 OSEE 曾尝试后撤出：奇宇称 jump 的 kron(L̄,L) 项在双模
  JW 基带弦算符（≥四阶），𝓛 不是二次型 doubled 算符，logm(Φ) 亦非单个
  二次型；e^{𝓛t} 是高斯信道（物理 Nambu 线性流闭合已验证），正确的快
  路径应从物理流 T_t + |I⟩_G 镜像构造 Choi 湮灭子（未完成）。

## 2026-10-29 会话（二）：legacy 删除 + M4 文档 + §6.4 性质测试

- `src/legacy/`、`test/legacy/` 已删除（无任何引用，v1 完整备份在
  `backup/GaussianStates`）。
- **M4 文档完成**：`docs/src/{index,manual,api}.md` 按 v7 全部重写
  （快速上手含一般非厄米/Gibbs/TFD/玻色四条工作流；manual 含 block
  约定、c₀、purify 湮灭子构造、Gibbs/TFD、熵公式、复杂度与验证摘要）。
  类型与 `blocks`/`bdg`/`densitymatrix`/`asoperator`/`covariance`/`ispure`
  等补齐 docstring，`Documenter` 构建通过（`build/index.html`）。
- **§6.4 性质测试完成**：整纯态熵 0（含 Rényi）、`S(∅)=0`、纯态互补
  `S(A)=S(Ā)`（费米/玻色）、`c₀` 无关、Rényi `α→1` 连续与单调、
  费米 `α=2` vs 稠密偏迹、玻色 `α=2` 解析；错误路径（尺寸/对称性/
  厄米性校验、非物理协方差、越界/重复 modes、非法 α）。
- 测试 **95/95** 全绿。

## 2026-10-29 会话（一）：M1/M3 核心缺口全部关闭

### 已实现并通过测试（63/63）

- **费米一般 `purify`（O(n³)，M1 完成）**：
  湮灭子 `V = E − exp(bdg)·Pmᵀ`（伴随作用矩阵就是 `exp(bdg(op))`，
  EPR 镜像 `M(cᵢ)=+dᵢ†、M(cᵢ†)=−dᵢ`），协方差 = 湮灭子行空间投影子
  （QR 正交基构造，避免 `(VV†)⁻¹` 平方条件数），再 Po-重排 + to_majorana。
  n=1..4 与稠密零空间逐元素一致（1e-15）；n=64 purify ≈ 72 ms、n=128 ≈ 112 ms。
  **旧记录里钉死的 M = [[−h,B],[A,hᵀ]] 是错的**——正确的就是 bdg 矩阵本身。
- **费米 TFD**：走一般路径 `purify(asoperator(exp(op/2)))`，自动 O(n³)；
  `partialtr(psi,1:n) == GaussianState(gs)` 精确（n=1..3 测试）。
- **玻色 Gibbs（A≠0 压缩热态修复）**：旧公式 `½coth(−bdg/2)` 只在 A=0 正确。
  正确构造：伴随生成元 `𝒜 = [[h,A],[−Ā,−hᵀ]]`（[H,Ψ]=−𝒜Ψ）本征值 ± 配对，
  负支热权重 `ν = ½coth(−α/2)`，按本征值分组做 σ₃-Gram 组装
  `C = Σ ν(T₋T₋† + T₊T₊†)`（简并稳健）。失稳/无界由 σ₃ 符号检查 +
  α<0 拒绝（**谱全实不保证稳定**；判据是 σ₃ 符号结构 / −X_c ≻ 0）。
  n=1 多组 + n=2 随机与稠密 Fock 一致（热态热尾巴决定 nbas 截断误差）。
- **玻色一般 `purify`（M3 完成，任意 n、任意配对）**：
  湮灭子 `V = E − exp(𝒜)·Pmᵀ`，𝒜 = [[h,Aᵀ],[−Bᵀ,−hᵀ]]，玻色镜像无符号
  `M(aᵢ)=+dᵢ†、M(aᵢ†)=+dᵢ`；**A、B 复对称 ⟹ 𝒜 ∈ sp ⟹ V 迷向 ⟹
  Choi 态恒高斯**（之前"多模配对非高斯"的论断是推导失误，已被
  稠密 Wick 检验与 n=1 逐元素对拍推翻）。协方差由
  `kernel(Σ−(i/2)Ω) ∋ rowspace(V)` 的线性系统 `[Wq; W̄q]Σ = [∓(i/2)WqΩ…]`
  补全（四极协变式，W̄(Σ+iΩ/2)=0）。n=1 数守恒/配对/复 h 与稠密 Choi
  逐元素一致（1e-15）；n=2 与独立 Williamson 构造一致（0.0）。
- **玻色 TFD**：`purify(asoperator(exp(op/2)))`（与费米同一路径、正则
  `(ρ^{1/2}⊗I)|I⟩` 约定）。n=1 A=0/A=0.2 与稠密 TFD 一致（截断极限）；
  真稳定随机算符 n=1..8 × 4 seeds `ispure` + `partialtr==Gibbs` 全通过。
- **bug 修复：`_boson_annihilator` 转置了**（`a[k+1,k]` 应为 `a[k,k+1]`）：
  A=B=0 时只差常数（单模热态测试侥幸通过），配对项全错。修复后
  `densitymatrix`（玻色）才可用作仲裁。
- `_gaussian_state`（玻色）加正定检查：不可归一化（如 h>0）会得到
  负定"协方差"，旧的 abs 谱检查骗不过它。
- `partialtr`、`entropy(gs[,modes])`、Rényi、`densitymatrix` 不变，
  测试扩到 63 项。

### 性能

费米：purify n=64 72ms / n=128 112ms（QR 版；`(VV†)\V` 版在 n=64 即超
1e-8 容差）。玻色：n=8/16 Gibbs/TFD/purify 均毫秒级。

### 已知边界 / 遗留（按优先级）

1. 玻色 `densitymatrix` 的 Fock 截断：强压缩/热态需要大 nbas，无误差
   估计（后续：nbas 自适应或错误提示）。

### 语义锚点（调试时的仲裁约定）

- 费米 Choi：|I⟩_G = exp(Σc†d†)|0⟩，ket 模低位（`_dense_vectorize`）；
  `_nambu_ops` 序 (c,c†,d,d†)；Po 重排 → (c,d,c†,d†) → to_majorana。
- 玻色 Choi 稠密仲裁：`vec(g)` 按 (ket₁..n, bra₁..n) kron 序；x=(a+a†)/√2。
- 玻色 TFD 正则约定：x-x 交叉 +g、p-p 交叉 −g（数守恒 t=e^h 时
  cross = (Re t, Im t; Im t, −Re t)/(1−|t|²)）。
- 稠密 Fock 仲裁的截断规律：配对元 ∝√(mn) 随能级增长，强压缩时
  收敛慢（nbas 20→36 误差 0.48→0.06 的例子在 debug/dbg_seed13.jl）。
