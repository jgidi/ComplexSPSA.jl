using ProgressMeter, LinearAlgebra, Statistics
using ComplexSPSA

Nvars = 2
Nruns = 10^3
Niters = 2*10^2
Nmeasures = 10^3

# Generate a random Haar ket in dimension d
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

# Reference (target) state
refstate = rand_qudit(ComplexF64, Nvars)

# Theoretical functions
infidelity(z) = 1 - fidelity(z, refstate)
metric(z1, z2) = -0.5fidelity(z1, z2)

# Experimental function to simulate
f(z) = simulate_experiment(infidelity(z), Nmeasures)

labels = ["SPSA", "CSPSA", "SPSA2", "CSPSA2", "SPSA_QN", "CSPSA_QN", "CSPSA_QN_scalar"]

zacc = ones(ComplexF64, Nvars, Niters, Nruns, length(labels))
@showprogress for run in 1:Nruns
    guess = rand(ComplexF64, Nvars)
    
    zacc[:, :, run, 1] = SPSA_on_complex(f, guess, Niters)
    zacc[:, :, run, 2] = CSPSA(f, guess, Niters)
    zacc[:, :, run, 3] = SPSA2_on_complex(f, guess,Niters)
    zacc[:, :, run, 4] = CSPSA2(f, guess, Niters)
    zacc[:, :, run, 5] = SPSA_QN_on_complex(f, metric, guess, Niters)
    zacc[:, :, run, 6] = CSPSA_QN(f, metric, guess, Niters)
    zacc[:, :, run, 7] = CSPSA_QN_scalar(f, metric, guess, Niters)
end

# Calculate statistics
using Statistics: mean, var

# Apply theoretical fidelity over variables variables obtained
fz = apply_along_dim(infidelity, zacc, dim = 1)

# Mean and variance
fmean = mean(fz, dims = 3)
fvar  =  var(fz, dims = 3)

# Get rid of singleton dimensions
fmean = squeeze(fmean);
fvar = squeeze(fvar);

using Plots

# Make plot
p = plot(yscale = :log10,                  # Log scale on the y-axis
         xlabel = "Number of iterations",  # 
         ylabel = "Infidelity",            #
         legend = :outertopright,
)

# Add all optimizers to the plot
for i in eachindex(labels)
    plot!(p, 1:Niters, fmean[:, i],  # Plot mean lines
          ribbon = fvar[:, i],       # Plot variance
          label = labels[i],         # Show optimizer label
    )
end
    
display(p)

# p = plot!(p, dpi = 300)
# savefig(p, "tomography_Nruns$Nruns.png")


