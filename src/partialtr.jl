# ============================================================================
# partialtr.jl —— 协方差子矩阵偏迹
# ============================================================================

"""
    partialtr(s::GaussianState, modes) -> GaussianState

对物理模式子集 `modes` 取偏迹；返回只含协方差子矩阵的 `GaussianState`。
费米按代数（模式）约化；玻色为通常偏迹。
"""
function partialtr(s::FermionicGaussianState, modes::AbstractVector{<:Integer})
    N = nmodes(s)
    _check_modes(modes, N)
    idx = _modeindices(modes)
    return FermionicGaussianState(s.Γ[idx, idx])
end
function partialtr(s::BosonicGaussianState, modes::AbstractVector{<:Integer})
    N = nmodes(s)
    _check_modes(modes, N)
    idx = _modeindices(modes)
    return BosonicGaussianState(s.Σ[idx, idx])
end
partialtr(s::AbstractGaussianState, mode::Integer) = partialtr(s, [mode])
