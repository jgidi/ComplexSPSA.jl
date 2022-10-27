on_complex_docs = "Takes each variable, separate them as a pair of real values, and uses the equivalent real optimizer on them."

# First order implementations

"""
    SPSA(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)

First order optimizer taking real variables.
"""
function SPSA(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    return _first_order(f, guess, Niters; kwargs...)
end

"""
    SPSA_on_complex(f::Function, guess::AbstractVector, Niters; kwargs...)

First order optimizer taking complex variables.
$(on_complex_docs)
"""
function SPSA_on_complex(f::Function, guess::AbstractVector, Niters; kwargs...)
    f_re, guess_re = complex_problem_as_real(f, ComplexF64.(guess))
    acc = _first_order(f_re, guess_re, Niters; kwargs...)

    return Matrix(reinterpret(ComplexF64, acc))
end
"""
    CSPSA(f::Function, guess::AbstractVector, Niters; kwargs...)

First order optimizer taking complex variables
"""
function CSPSA(f::Function, guess::AbstractVector, Niters; kwargs...)
    return _first_order(f, ComplexF64.(guess), Niters; kwargs...)
end

# Second order implementations

"""
    SPSA2(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)

Second order optimizer taking complex variables.
"""
function SPSA2(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    return _preconditioned(f, guess, Niters; kwargs...)
end

"""
    SPSA2_on_complex(f::Function, guess::AbstractVector, Niters; kwargs...)

Second order optimizer taking complex variables.
$(on_complex_docs)
"""
function SPSA2_on_complex(f::Function, guess::AbstractVector, Niters; kwargs...)
    f_re, guess_re = complex_problem_as_real(f, ComplexF64.(guess))
    acc = _preconditioned(f_re, guess_re, Niters; kwargs...)

    return Matrix(reinterpret(ComplexF64, acc))
end

"""
    CSPSA2(f::Function, guess::AbstractVector, Niters; kwargs...)

Second order optimizer taking complex variables.
"""
function CSPSA2(f::Function, guess::AbstractVector, Niters; kwargs...)
    return _preconditioned(f, ComplexF64.(guess), Niters; kwargs...)
end

## Scalars

scalar_docs = "Uses a scalar approximation of the Hessian estimator."

"""
    SPSA2_scalar(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)

Second order optimizer taking real variables.
$(scalar_docs)
"""
function SPSA2_scalar(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    return _preconditioned(f, guess, Niters;
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

"""
    SPSA2_on_complex(f::Function, guess::AbstractVector, Niters; kwargs...)

Second order optimizer taking complex variables.
$(scalar_docs)
$(on_complex_docs)
"""
function SPSA2_scalar_on_complex(f::Function, guess::AbstractVector, Niters; kwargs...)
    f_re, guess_re = complex_problem_as_real(f, ComplexF64.(guess))
    acc = _preconditioned(f_re, guess_re, Niters;
                          hessian_estimate=hessian_estimate_scalar,
                          kwargs...)

    return Matrix(reinterpret(ComplexF64, acc))
end

"""
    CSPSA2_scalar(f::Function, guess::AbstractVector, Niters; kwargs...)

Second order optimizer taking complex variables.
$(scalar_docs)
"""
function CSPSA2_scalar(f::Function, guess::AbstractVector, Niters; kwargs...)
    return _preconditioned(f, ComplexF64.(guess), Niters;
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

# Natural gradient
#
"""
    SPSA_QN(f::Function, fidelity::Function,
            guess::AbstractVector{<:Real}, Niters; kwargs...)

Quantum Natural optimizer taking real variables.
"""
function SPSA_QN(f::Function, fidelity::Function,
                 guess::AbstractVector{<:Real}, Niters; kwargs...)

    return _preconditioned(f, guess, Niters;
                           fidelity=fidelity,
                           kwargs...)
end

"""
    SPSA_QN_on_complex(f::Function, fidelity::Function,
                       guess::AbstractVector, Niters; kwargs...)

Quantum Natural optimizer taking complex variables.
$(on_complex_docs)
"""
function SPSA_QN_on_complex(f::Function, fidelity::Function,
                            guess::AbstractVector, Niters; kwargs...)

    f_re, guess_re = complex_problem_as_real(f, ComplexF64.(guess))
    acc = _preconditioned(f_re, guess_re, Niters;
                          fidelity=fidelity,
                          kwargs...)

    return Matrix(reinterpret(ComplexF64, acc))
end

"""
     CSPSA_QN(f::Function, fidelity::Function,
              guess::AbstractVector, Niters; kwargs...)

Quantum Natural optimizer taking complex variables.
"""
function CSPSA_QN(f::Function, fidelity::Function,
                  guess::AbstractVector, Niters; kwargs...)

    return _preconditioned(f, ComplexF64.(guess), Niters;
                           fidelity=fidelity,
                           kwargs...)
end

## Full
full_docs = "Does not consider the block-diagonal approximation for the complex Hessian estimator."

"""
    CSPSA2_full(f::Function, guess::AbstractVector, Niters; kwargs...)

Second order optimizer taking complex variables.
$(full_docs)
"""
function CSPSA2_full(f::Function, guess::AbstractVector, Niters; kwargs...)
    return _preconditioned(f, ComplexF64.(guess), Niters;
                           hessian_estimate=hessian_estimate_full,
                           apply_hessian=apply_hessian_full,
                           kwargs...)
end

"""
    CSPSA_QN_full(f::Function, fidelity::Function,
                  guess::AbstractVector, Niters; kwargs...)

Quantum Natural optimizer taking complex variables.
$(full_docs)
"""
function CSPSA_QN_full(f::Function, fidelity::Function,
                       guess::AbstractVector, Niters; kwargs...)

    return _preconditioned(f, ComplexF64.(guess), Niters;
                           fidelity=fidelity,
                           hessian_estimate=hessian_estimate_full,
                           apply_hessian=apply_hessian_full,
                           kwargs...)
end

## Scalars
"""
    SPSA_QN_scalar(f::Function, fidelity::Function,
                   guess::AbstractVector{<:Real}, Niters; kwargs...)

Quantum Natural optimizer taking real variables.
$(scalar_docs)
"""
function SPSA_QN_scalar(f::Function, fidelity::Function,
                        guess::AbstractVector{<:Real}, Niters; kwargs...)

    return _preconditioned(f, guess, Niters;
                           fidelity=fidelity,
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

"""
    SPSA_QN_scalar_on_complex(f::Function, fidelity::Function,
                              guess::AbstractVector, Niters; kwargs...)

Quantum Natural optimizer taking complex variables.
$(scalar_docs)
$(on_complex_docs)
"""
function SPSA_QN_scalar_on_complex(f::Function, fidelity::Function,
                                   guess::AbstractVector, Niters; kwargs...)

    f_re, guess_re = complex_problem_as_real(f, ComplexF64.(guess))
    acc = _preconditioned(f_re, guess_re, Niters;
                          fidelity=fidelity,
                          hessian_estimate=hessian_estimate_scalar,
                          kwargs...)

    return Matrix(reinterpret(ComplexF64, acc))
end

"""
    CSPSA_QN_scalar(f::Function, fidelity::Function,
                    guess::AbstractVector, Niters; kwargs...)

Quantum Natural optimizer taking complex variables.
$(scalar_docs)
"""
function CSPSA_QN_scalar(f::Function, fidelity::Function,
                         guess::AbstractVector, Niters; kwargs...)

    return _preconditioned(f, ComplexF64.(guess), Niters;
                           fidelity=fidelity,
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end
