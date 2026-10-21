# ============================================================================
# GaussianStates.jl 测试（驱动）
#
# 两条费米分支各自用**稠密密度矩阵**独立对照：
#   Gibbs：ρ = exp(−½ψ†·bdg·ψ)/Z 的 Nambu-Fock 稠密表示 → 偏迹 → 熵
#   纯配对：|Ψ⟩ = exp(½c†Wc†)|0⟩ 的 JW 稠密表示 → ρ=ψψ† → 偏迹 → 熵
# 玻色同样用 Fock 截断密度矩阵对照。全部用随机 h、g。
#
# 按主题拆分（各文件在全局作用域求值，共享此处定义的 `rng`，保持与拆分前
# 完全相同的随机数消耗顺序；嵌套 @testset 经动态栈登记为子项）：
#   bdg_assembly.jl     BdG 矩阵组装 / 玻色块转换仲裁 / 构造校验
#   fermionic_gibbs.jl  费米 Gibbs：稠密仲裁、非规范规范化、温度极限、复输入
#   fermionic_pure.jl   费米纯配对：稠密仲裁、单模解析、双配对可加
#   bosonic.jl          玻色：稠密仲裁、ωI 热态、双模可加
#   entropy.jl          entropy/osee 接口、二分割互补、协方差惰性缓存
#   densitymatrix.jl    densitymatrix 三方对拍、benchmark
# ============================================================================

using GaussianStates
using GaussianStates: covariance                    # 不导出；测试经显式导入使用
using LinearAlgebra
using Random
using Test

include("helpers.jl")

const rng = MersenneTwister(20240915)

@testset "GaussianStates" begin
    include("bdg_assembly.jl")
    include("fermionic_gibbs.jl")
    include("fermionic_pure.jl")
    include("bosonic.jl")
    include("entropy.jl")
    include("densitymatrix.jl")
end
