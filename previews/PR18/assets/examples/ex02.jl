# Remember to install the required libraries first, or this lines
# will throw and error.
using ComplexSPSA               #
using LinearAlgebra             # Provides 'norm'
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
guess = rand_qubit()                # Random qubit as a guess
zC = CSPSA(f, guess, 100)           # 100 iterations of CSPSA
zS = SPSA_on_complex(f, guess, 100) # 100 iterations of CSPSA


## Obtain the infidelities per iteration
fzC = [f(zC[:, i]) for i in axes(zC, 2)]
fzS = [f(zS[:, i]) for i in axes(zS, 2)]


## Make a plot, using the library PyPlot

# Logarithmic scale on y-axis
semilogy(1:100, fzC, label = "CSPSA")
semilogy(1:100, fzS, label = "SPSA")

# Labels
xlabel("Interations")
ylabel("Infidelity")

# Show legend
legend(loc="best")

show()
