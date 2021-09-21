"""
    SPSA(f, guess, Niters)

Wrapper around the first order [SPSA optimizer from Qiskit](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.QNSPSA.html#qiskit.algorithms.optimizers.SPSA).
"""
function SPSA(f, guess, Niters)
    # Select optimizer class from python library
    qiskit_opts = pyimport("qiskit.algorithms.optimizers")

    # Preallocate array to store optimized variables
    Nx = length(guess)
    values = Array{Float64}(undef, Nx, Niters)

    # Define optimizer rules
    optimizer = qiskit_opts.SPSA(maxiter = Niters,
                                 callback = callback_accumulator(values),
                                 )

    # Start optimization
    optimizer.optimize(Nx,
                       f,
                       initial_point = guess,
                       )
    return values
end

"""
    SPSA2(f, guess, Niters)

Wrapper around the second order [SPSA optimizer from Qiskit](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.QNSPSA.html#qiskit.algorithms.optimizers.SPSA).
"""
function SPSA2(f, guess, Niters)
    # Select optimizer class from python library
    qiskit_opts = pyimport("qiskit.algorithms.optimizers")

    # Preallocate array to store optimized variables
    Nx = length(guess)
    values = Array{Float64}(undef, Nx, Niters)

    # Define optimizer rules
    optimizer = qiskit_opts.SPSA(maxiter = Niters,
                                 callback = callback_accumulator(values),
                                 second_order = true,
                                 )

    # Start optimization
    optimizer.optimize(Nx,
                       f,
                       initial_point = guess,
                       )
    return values
end

"""
    SPSA_NG(f, fidelity, guess, Niters)

Wrapper around the [QNSPSA optimizer from Qiskit](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.QNSPSA.html#qiskit.algorithms.optimizers.QNSPSA).
"""
function SPSA_NG(f, fidelity, guess, Niters)
    # Select optimizer class from python library
    qiskit_opts = pyimport("qiskit.algorithms.optimizers")

    # Preallocate array to store optimized variables
    Nx = length(guess)
    values = Array{Float64}(undef, Nx, Niters)

    # Define optimizer rules
    optimizer = qiskit_opts.QNSPSA(maxiter = Niters,
                                   fidelity = fidelity,
                                   callback = callback_accumulator(values),
                                   )

    # Start optimization
    optimizer.optimize(Nx,
                       f,
                       initial_point = guess,
                       )
    return values
end
