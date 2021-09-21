"""
    SPSA_on_complex(f, guess::Vector{Complex{Float64}}, Niters)

Wrapper around [`SPSA`](@ref) which internally converts the
complex-valued initial point, `guess`, and objective function, `f`,
to be optimized as a real-variable problem with twice the parameters.
"""
function SPSA_on_complex(f, guess::Vector{Complex{Float64}}, Niters;
                         a = gains[:a], b = gains[:b],
                         A = gains[:A], s = gains[:s], t = gains[:t],
                         )
    # Prepare real-variable versions of the function f and the initial value
    f_real(x)  = f(reinterpret(Complex{Float64}, x))
    guess_real = reinterpret(Float64, guess)

    # Solve using real optimizer
    xacc = SPSA(f_real, guess_real, Niters,
                a = a, b = b, A = A, s = s, t = t,
                )

    # Return optimized variables as complex
    return reinterpret(Complex{Float64}, xacc)
end

"""
    SPSA2_on_complex(f, guess::Vector{Complex{Float64}}, Niters)

Wrapper around [`SPSA2`](@ref) which internally converts the
complex-valued initial point, `guess`, and objective function, `f`,
to be optimized as a real-variable problem with twice the parameters.
"""
function SPSA2_on_complex(f, guess::Vector{Complex{Float64}}, Niters;
                          b = gains[:b],
                          A = gains[:A], s = gains[:s], t = gains[:t],
                          )
    # Prepare real-variable versions of the function f and the initial value
    f_real(x)  = f(reinterpret(Complex{Float64}, x))
    guess_real = reinterpret(Float64, guess)

    # Solve using real optimizer
    xacc = SPSA2(f_real, guess_real, Niters,
                 b = b, A = A, s = s, t = t,
                 )

    # Return optimized variables as complex
    return reinterpret(Complex{Float64}, xacc)
end

"""
    SPSA_NG_on_complex(f, fidelity, guess::Vector{Complex{Float64}}, Niters)

Wrapper around [`SPSA_NG`](@ref) which internally converts the
complex-valued initial point, `guess`, the `fidelity`, and the objective function, `f`,
to be optimized as a real-variable problem with twice the parameters.
"""
function SPSA_NG_on_complex(f, fidelity, guess::Vector{Complex{Float64}}, Niters;
                            b = gains[:b],
                            A = gains[:A], s = gains[:s], t = gains[:t],
                            )
    # Prepare real-variable versions of the function f,
    # the fidelity, and the initial value
    f_real(x)  = f(reinterpret(Complex{Float64}, x))
    guess_real = reinterpret(Float64, guess)
    fid_real(x, y) = fidelity(reinterpret(Complex{Float64}, x),
                              reinterpret(Complex{Float64}, y))

    # Solve using real optimizer
    xacc = SPSA_NG(f_real, fid_real, guess_real, Niters,
                   b = b, A = A, s = s, t = t,
                   )

    # Return optimized variables as complex
    return reinterpret(Complex{Float64}, xacc)
end
