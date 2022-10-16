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
    acc_cx =  _first_order(f_cx, guess_cx, Niters; kwargs...)

    return acc_cx
end

function CSPSA(f::Function, guess::AbstractVector{<:Complex}, Niters; kwargs...)
    return _first_order(f, guess, Niters; kwargs...)
end

# Second order implementations
