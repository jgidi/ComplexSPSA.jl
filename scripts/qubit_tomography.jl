using ProgressMeter, LinearAlgebra, Statistics

include("../src/ComplexSPSA.jl")
using .ComplexSPSA

# Generate a random Haar ket
rand_qudit(d::Int = 2) = rand_qudit(Complex{Float64}, d)
rand_qudit(T::DataType = Complex{Float64}) = rand_qudit(T, 2)

function rand_qudit(T::DataType, d...)
    ket = randn(T, d...)

    # Return normalized
    return ket / LinearAlgebra.norm(ket)
end

# Fidelity between 2 qudits in their (complex) natural form
function fidelity(guess, state)
    prod2 = abs2(state' * guess)
    norms2 = sum(abs2, guess) * sum(abs2, state)
    return prod2 / norms2
end

Nruns = 10^3
Nit = 10^3
Nvars = 2
Nmeasures = 10^5

# Reference state
refstate = rand_qudit(Complex{Float64}, Nvars)

infidelity(z) = 1 - fidelity(z, refstate)

metric(z1, z2) = -0.5fidelity(z1, z2)


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

# p = plot!(p, dpi = 300)
# savefig(p, "tomography_Nruns$Nruns.png")
