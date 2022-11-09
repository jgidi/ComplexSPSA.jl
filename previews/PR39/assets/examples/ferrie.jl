using ProgressMeter, LinearAlgebra, Statistics, StaticArrays

using ComplexSPSA

Nvars = 10
Nruns = 100
Niters = 500
Nmeasures = 2^13 #10^3

function random_state(d = 2)
    ψ = @SVector rand(Complex{Float64}, d)

    return ψ / norm(ψ)
end

# Pauli Z, Up and Down
Sz = SMatrix{2,2}([1.0  0.0; 0.0 -1.0])
Su = SMatrix{2,2}([0.0  1.0; 0.0  0.0])
Sd = SMatrix{2,2}([0.0  0.0; 1.0  0.0])

# Reference states
U0 = random_state()             # Initial state
Ut = random_state()             # Final state

# Control gate. Managed through the pulse parameters ω
function Uc(ω)
    U = U0
    for v in ω
        U = exp(-0.5im*(Sz + v*Su + conj(v)*Sd)) * U
    end
    return U
end

fidelity(s1, s2) = abs2(s1' * s2) / abs2( norm(s1) * norm(s2)  )

# W/out experimental noise
infidelity(ω) = 1 - fidelity(Uc(ω), Ut)
metric(ω1, ω2) = -0.5fidelity(Uc(ω1), Uc(ω2))

# Experimental function to optimize
f(z) = simulate_experiment(infidelity(z), Nmeasures)

zacc = ones(ComplexF64, Nvars, Niters, Nruns, 8) # 8 optimizers
@showprogress for run in 1:Nruns
    guess = rand(ComplexF64, Nvars)

    zacc[:, :, run, 1] = SPSA_on_complex(f, guess, Niters)
    zacc[:, :, run, 2] = CSPSA(f, guess, Niters)
    zacc[:, :, run, 3] = SPSA2_on_complex(f, guess,Niters)
    zacc[:, :, run, 4] = CSPSA2(f, guess, Niters)
    zacc[:, :, run, 5] = SPSA_QN_on_complex(f, metric, guess, Niters)
    zacc[:, :, run, 6] = CSPSA_QN(f, metric, guess, Niters)
    zacc[:, :, run, 7] = SPSA_QN_scalar_on_complex(f, metric, guess, Niters)
    zacc[:, :, run, 8] = CSPSA_QN_scalar(f, metric, guess, Niters)
end

# Calculate statistics

# Apply theoretical fidelity over variables variables obtained
fz = apply_along_dim(infidelity, zacc, dim = 1)

# Mean and variance
fmean = mean(fz, dims = 3)
fmedian = median(fz, dims = 3)
fvar  =  var(fz, dims = 3)

# Get rid of singleton dimensions
fmean = squeeze(fmean)
fmedian = squeeze(fmedian)
fvar = squeeze(fvar)

using Plots

# Line colors [SPSA, CSPSA]
colors = ["blue", "red"]
# Line styles [Mean, Median]
styles =[:line, :dot]
# Line widths
lw = 2

# First order
p1 = plot(ylabel = "Infidelity", leftmargin=6Plots.mm)
# Mean
plot!(p1, 1:Niters, fmean[:, 1], ribbon=fvar[:, 1], line=(colors[1], lw), label="SPSA")
plot!(p1, 1:Niters, fmean[:, 2], ribbon=fvar[:, 2], line=(colors[2], lw), label="CSPSA")
# Median
plot!(p1, 1:Niters, fmedian[:, 1], line=(colors[1], lw, styles[2]), label=false)
plot!(p1, 1:Niters, fmedian[:, 2], line=(colors[2], lw, styles[2]), label=false)

# Second order
p2 = plot()
# Mean
plot!(p2, 1:Niters, fmean[:, 3], ribbon=fvar[:, 3], line=(colors[1], lw), label="2-SPSA")
plot!(p2, 1:Niters, fmean[:, 4], ribbon=fvar[:, 4], line=(colors[2], lw), label="2-CSPSA")
# Median
plot!(p2, 1:Niters, fmedian[:, 3], line=(colors[1], lw, styles[2]), label=false)
plot!(p2, 1:Niters, fmedian[:, 4], line=(colors[2], lw, styles[2]), label=false)

# Natural gradient
p3 = plot()
# Mean
plot!(p3, 1:Niters, fmean[:, 5], ribbon=fvar[:, 5], line=(colors[1], lw), label="QN-SPSA")
plot!(p3, 1:Niters, fmean[:, 6], ribbon=fvar[:, 6], line=(colors[2], lw), label="QN-CSPSA")
# Median
plot!(p3, 1:Niters, fmedian[:, 5], line=(colors[1], lw, styles[2]), label=false)
plot!(p3, 1:Niters, fmedian[:, 6], line=(colors[2], lw, styles[2]), label=false)

# Natural gradient - scalar
p4 = plot()
# Mean
plot!(p4, 1:Niters, fmean[:, 7], ribbon=fvar[:, 7], line=(colors[1], lw), label="QN-SPSA scalar")
plot!(p4, 1:Niters, fmean[:, 8], ribbon=fvar[:, 8], line=(colors[2], lw), label="QN-CSPSA scalar")
# Median
plot!(p4, 1:Niters, fmedian[:, 7], line=(colors[1], lw, styles[2]), label=false)
plot!(p4, 1:Niters, fmedian[:, 8], line=(colors[2], lw, styles[2]), label=false)

# Combine subplots
p = plot(p1, p2, p3, p4,
         size = (1200, 260),
         yscale = :log10,
         grid = true,
         # ylims = extrema([fmean; fmedian]),
         framestyle = :box,
         xlim = (0, Niters),
         bottommargin=7Plots.mm,
         layout = (1, 4),
         thickness_scaling=1.1,
         xlabel = "Iterations",
         fontfamily = "serif-roman",
         )

# display(plot(p))

# Save plot as PDF
# savefig("figname.pdf")
savefig("$(Nvars)vars-$(Nruns)runs-$(Niters)iters_$(Nmeasures)measures.pdf")
