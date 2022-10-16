# First order implementations

first_order_kwarg_docs = "\
sign = -1,
initial_iter = 1,
a = gains[:a], b = gains[:b],
A = gains[:A], s = gains[:s], t = gains[:t],
Ncalibrate = 0,
Nresampling = 1,
postprocess = identity,
"

"""
    SPSA(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)

"""
function SPSA(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    return _first_order(f, guess, Niters; kwargs...)
end

function SPSA_on_complex(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    f_cx, guess_cx = real_problem_as_complex(f, guess)

    return _first_order(f_cx, guess_cx, Niters; kwargs...)
end

function CSPSA(f::Function, guess::AbstractVector{<:Complex}, Niters; kwargs...)
    return _first_order(f, guess, Niters; kwargs...)
end

# Second order implementations
function SPSA2(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    return _preconditioned(f, guess, Niters; kwargs...)
end

function SPSA2_on_complex(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    f_cx, guess_cx = real_problem_as_complex(f, guess)

    return _preconditioned(f_cx, guess_cx, Niters; kwargs...)
end

function CSPSA2(f::Function, guess::AbstractVector{<:Complex}, Niters; kwargs...)
    return _preconditioned(f, guess, Niters; kwargs...)
end

## Scalars
function SPSA2_scalar(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    return _preconditioned(f, guess, Niters;
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

function SPSA2_scalar_on_complex(f::Function, guess::AbstractVector{<:Real}, Niters; kwargs...)
    f_cx, guess_cx = real_problem_as_complex(f, guess)

    return _preconditioned(f_cx, guess_cx, Niters;
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

function CSPSA2_scalar(f::Function, guess::AbstractVector{<:Complex}, Niters; kwargs...)
    return _preconditioned(f, guess, Niters;
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

# Natural gradient
function SPSA_QN(f::Function, fidelity::Function,
                 guess::AbstractVector{<:Real}, Niters; kwargs...)

    return _preconditioned(f, guess, Niters;
                           fidelity=fidelity,
                           kwargs...)
end

function SPSA_QN_on_complex(f::Function, fidelity::Function,
                            guess::AbstractVector{<:Real}, Niters; kwargs...)

    f_cx, guess_cx = real_problem_as_complex(f, guess)

    return _preconditioned(f_cx, guess_cx, Niters;
                           fidelity=fidelity,
                           kwargs...)
end

function CSPSA_QN(f::Function, fidelity::Function,
                  guess::AbstractVector{<:Complex}, Niters; kwargs...)

    return _preconditioned(f, guess, Niters;
                           fidelity=fidelity,
                           kwargs...)
end

## Scalars
function SPSA_QN(f::Function, fidelity::Function,
                 guess::AbstractVector{<:Real}, Niters; kwargs...)

    return _preconditioned(f, guess, Niters;
                           fidelity=fidelity,
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

function SPSA_QN_on_complex(f::Function, fidelity::Function,
                            guess::AbstractVector{<:Real}, Niters; kwargs...)

    f_cx, guess_cx = real_problem_as_complex(f, guess)

    return _preconditioned(f_cx, guess_cx, Niters;
                           fidelity=fidelity,
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end

function CSPSA_QN(f::Function, fidelity::Function,
                  guess::AbstractVector{<:Complex}, Niters; kwargs...)

    return _preconditioned(f, guess, Niters;
                           fidelity=fidelity,
                           hessian_estimate=hessian_estimate_scalar,
                           kwargs...)
end
