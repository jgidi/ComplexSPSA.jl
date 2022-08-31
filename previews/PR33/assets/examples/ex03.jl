# Remember to install the required libraries first, or this lines
# will throw and error.
using ComplexSPSA               #
using LinearAlgebra             # Provides 'norm'
using Statistics                # Provides 'mean' and 'var'
using PyPlot                    # uses PyPlot from python to make plots

## Define the problem

# Generate a random qubit
function rand_qubit()
    ket = randn(Complex{Float64}, 2)

    # Return normalized
    return ket / LinearAlgebra.norm(ket)
end

# Infidelity between 2 qubits, 'guess' and 'state'
function infidelity(guess, state)
    prod2 = abs2(state' * guess)
    norms2 = sum(abs2, guess) * sum(abs2, state)
    return 1 - prod2/norms2
end

# Generate a reference (target) state
refstate = rand_qubit()

# Define objective function as the infidelity of the argument
# with respect to the reference state
f(z) = infidelity(z, refstate)


## Optimize 'f' using CSPSA and SPSA

Niters = 100                    # Number of iterations
Nruns = 1000                    # Number of runs

# Make an array to store results
# Axes are: (1 : variable number), (2 : iteration),
# (3 : run number), (4 : optimizer number)
z = ones(ComplexF64, 2, Niters, Nruns, 2)
for run in 1:Nruns
    guess = rand_qubit()        # Random qubit as a guess

    z[:, :, run, 1] = CSPSA(f, guess, Niters)
    z[:, :, run, 2] = SPSA_on_complex(f, guess, Niters)
end

## Obtain the infidelity statistics per iteration

# Apply the function 'f' along the first axis
# 'apply_along_dim' is provided by ComplexSPSA
fz = apply_along_dim(f, z, dim = 1)

# Get the mean and variance along the axis of runs (3rd axis)
fmean = mean(fz, dims = 3)
fvar  =  var(fz, dims = 3)

# Remove singleton (reduced, length-1) dimensions
# 'squeeze' is provided by ComplexSPSA
fmean = squeeze(fmean)
fvar = squeeze(fvar)

# Now 'fmean' and 'fvar' have 2 axes: (1: iteration) and (2: optimizer)

# ## Make a plot, using the library PyPlot

# Logarithmic scale on y-axis
# Plot mean values
semilogy(1:Niters, fmean[:, 1], label = "CSPSA")
semilogy(1:Niters, fmean[:, 2], label = "SPSA")
# Plot variances
fill_between(1:Niters, fmean[:, 1]-fvar[:, 1], fmean[:, 1]+fvar[:, 1], alpha = 0.5)
fill_between(1:Niters, fmean[:, 2]-fvar[:, 2], fmean[:, 2]+fvar[:, 2], alpha = 0.5)

# Labels
xlabel("Interations")
ylabel("Infidelity")

# Show legend
legend(loc="best")

show()
