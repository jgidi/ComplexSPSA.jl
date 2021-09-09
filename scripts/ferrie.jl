using ProgressMeter, LinearAlgebra, Statistics

include("../src/ComplexSPSA.jl")
using .ComplexSPSA

Nruns = 100
Nit = 500
Nvars = 10
Nmeasures = 10^3

using StaticArrays

function random_unitary(d)
    q, r = qr(randn(d, d))
    U = q * Diagonal(sign.(diag(r)))
    return U
end

# Pauli Z, Up and Down
Sz = SMatrix{2,2}([1.0  0.0; 0.0 -1.0])
Su = SMatrix{2,2}([0.0  1.0; 0.0  0.0])
Sd = SMatrix{2,2}([0.0  0.0; 1.0  0.0])

# Control gate. Managed through the pulse parameters ω
Uc(ω) = prod( exp(-im*(Sz + v*Su + conj(v)*Sd)) for v in Iterators.reverse(ω) )

# Target gate
Ut = random_unitary(2)

fidelity(U1, U2) = 0.25abs2(tr(U1' * U2))

# # W/out experimental noise
infidelity(ω) = 1 - fidelity(Uc(ω), Ut)
metric(ω1, ω2) = -0.5fidelity(Uc(ω1), Uc(ω2))

# Make comparison

optimizers = [
    ComplexSPSA.SPSA_complex,
    ComplexSPSA.CSPSA,
    ComplexSPSA.SPSA2_complex,
    ComplexSPSA.CSPSA2,
    ComplexSPSA.SPSA_NG,
    ComplexSPSA.CSPSA_NG,
    ComplexSPSA.CSPSA_NG_scalar,
]

guess = rand(Complex{Float64}, Nvars, Nruns)

using Plots

p = ComplexSPSA.make_comparison_plot(infidelity, metric, guess, optimizers, Nit, Nmeasures)

plot!(p, yscale = :log10)
display(p)

p = plot!(p, dpi = 300)
savefig(p, "ferrie_Nruns$Nruns.png")
