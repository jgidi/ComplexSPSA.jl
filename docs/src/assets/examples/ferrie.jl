using ProgressMeter, LinearAlgebra, Statistics

using ComplexSPSA

Nvars = 10
Nruns = 100
Niters = 500
Nmeasures = Inf #10^3

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

# W/out experimental noise
infidelity(ω) = 1 - fidelity(Uc(ω), Ut)
metric(ω1, ω2) = -0.5fidelity(Uc(ω1), Uc(ω2))

# Experimental function to optimize
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
         #legend = :outertopright,
)

# Add all optimizers to the plot
for i in eachindex(labels)
    plot!(p, 1:Niters, fmean[:, i],  # Plot mean lines
          ribbon = fvar[:, i],       # Plot variance
          label = labels[i],         # Show optimizer label
    )
end
    
display(p)


