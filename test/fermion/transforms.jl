# 费米子：Nambu ↔ Majorana 协方差变换（稠密 Fock 仲裁 + 往返）
using Test, LinearAlgebra, Random
include(joinpath(dirname(@__FILE__), "..", "helpers.jl"))

@testset "Fermionic" begin
    @testset "Nambu ↔ Majorana" begin
        # 稠密 Fock 独立构造两侧：C = ⟨Ψ_aΨ_b†⟩ 与 Γ = i⟨γ_aγ_b⟩ 互为正反变换
        for (n, seed) in ((1, 1), (2, 3), (3, 5))
            rho = fockmatrix(exp(random_op(n, seed; herm=true)))
            rho = rho / tr(rho)
            c = [GS._jw_annihilator(n, i) for i in 1:n]
            ops = [c; adjoint.(c)]                    # Ψ = (c, c†)
            C = zeros(ComplexF64, 2n, 2n)
            for a in 1:2n, b in 1:2n
                C[a, b] = tr(rho * ops[a] * adjoint(ops[b]))
            end
            Γ = cov_from_rho(rho, n)
            @test maximum(abs.(GS.to_majorana(C) .- Γ)) < 1e-10
            @test maximum(abs.(GS.to_nambu(Γ) .- C)) < 1e-10
        end
        # 往返：Gibbs 态 Majorana 协方差
        for (n, seed) in ((1, 2), (2, 7), (3, 11))
            Γ = covariance(GaussianState(exp(random_op(n, seed; herm=true))))
            @test maximum(abs.(GS.to_majorana(GS.to_nambu(Γ)) .- Γ)) < 1e-12
        end
    end
end
