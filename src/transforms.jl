# ============================================================================
# transforms.jl —— Bogoliubov (Nambu) 协方差 ↔ Majorana / (x,p) 协方差
#
# 非导出函数；只在 purify / GaussianState(Gibbs) 内部使用。
# ============================================================================

# Nambu 协方差 C_ab = ⟨Ψ_a Ψ_b†⟩, Ψ=(c,c†)
# Majorana Γ_ab = i⟨γ_aγ_b⟩, γ_{2m-1}=c_m+c_m†, γ_{2m}=i(c_m†-c_m)
function to_majorana(C::AbstractMatrix)
    C = Matrix{ComplexF64}(C)
    N2, N2b = size(C)
    N2 == N2b || throw(ArgumentError("Nambu 协方差必须是方阵"))
    iseven(N2) || throw(ArgumentError("Nambu 协方差维数必须是偶数"))
    N = N2 ÷ 2

    P = zeros(ComplexF64, N2, N2)
    for i in 1:N
        P[i, N+i] = 1.0
        P[N+i, i] = 1.0
    end

    Co = zeros(ComplexF64, N2, N2)
    for m in 1:N
        Co[2m-1, m] = 1
        Co[2m-1, N+m] = 1
        Co[2m, N+m] = im
        Co[2m, m] = -im
    end

    Γ = real.(im .* (Co * (C * P) * transpose(Co)))
    Γ = (Γ .- transpose(Γ)) ./ 2
    return Γ
end

# Majorana 协方差 Γ_ab = i⟨γ_aγ_b⟩（实反对称）→ Nambu 协方差 C_ab = ⟨Ψ_aΨ_b†⟩, Ψ=(c,c†)
# 反解 to_majorana：Co·(C·P)·Coᵀ = ⟨γ_aγ_b⟩，其对称部分被 {γ_a,γ_b}=2δ_ab 钉死为 I，
# 即 ⟨γ_aγ_b⟩ = I − iΓ，逆映射无信息损失
function to_nambu(Γ::AbstractMatrix)
    Γ = Matrix{Float64}(Γ)
    N2, N2b = size(Γ)
    N2 == N2b || throw(ArgumentError("Majorana 协方差必须是方阵"))
    iseven(N2) || throw(ArgumentError("Majorana 协方差维数必须是偶数"))
    N = N2 ÷ 2

    P = zeros(ComplexF64, N2, N2)
    for i in 1:N
        P[i, N+i] = 1.0
        P[N+i, i] = 1.0
    end

    Co = zeros(ComplexF64, N2, N2)
    for m in 1:N
        Co[2m-1, m] = 1
        Co[2m-1, N+m] = 1
        Co[2m, N+m] = im
        Co[2m, m] = -im
    end

    C = (Co \ (I - im * Γ)) / transpose(Co)
    return C * P
end

# Bogoliubov 协方差 C_ab = ½⟨{Ψ_a,Ψ_b†}⟩, Ψ=(a,a†)
# (x,p) 协方差 Σ_ab = ½⟨{R_a,R_b}⟩, R=(x₁,p₁,…)
function to_xp(C::AbstractMatrix)
    C = Matrix{ComplexF64}(C)
    N2, N2b = size(C)
    N2 == N2b || throw(ArgumentError("Bogoliubov 协方差必须是方阵"))
    iseven(N2) || throw(ArgumentError("Bogoliubov 协方差维数必须是偶数"))
    N = N2 ÷ 2

    W = zeros(ComplexF64, N2, N2)
    s = 1 / sqrt(2)
    for m in 1:N
        W[2m-1, m] = s; W[2m-1, N+m] = s
        W[2m, m] = -im * s; W[2m, N+m] = im * s
    end
    Σ = real.(W * C * adjoint(W))
    return (Σ .+ transpose(Σ)) ./ 2
end

# (x,p) 协方差 Σ_ab = ½⟨{R_a,R_b}⟩（实对称）→ Bogoliubov 协方差 C_ab = ½⟨{Ψ_a,Ψ_b†}⟩, Ψ=(a,a†)
# 反解 to_xp：W 是 Nambu ↔ (x,p) 的幺正正则变换，C = W†·Σ·W
function to_bogoliubov(Σ::AbstractMatrix)
    Σ = Matrix{Float64}(Σ)
    N2, N2b = size(Σ)
    N2 == N2b || throw(ArgumentError("(x,p) 协方差必须是方阵"))
    iseven(N2) || throw(ArgumentError("(x,p) 协方差维数必须是偶数"))
    N = N2 ÷ 2

    W = zeros(ComplexF64, N2, N2)
    s = 1 / sqrt(2)
    for m in 1:N
        W[2m-1, m] = s; W[2m-1, N+m] = s
        W[2m, m] = -im * s; W[2m, N+m] = im * s
    end
    return adjoint(W) * Σ * W
end
