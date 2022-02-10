"""
    squeeze(A :: AbstractArray)

Remove the singleton (length-1) dimensions of a multidimensional array.

Example
===

```julia-repl
julia> a = rand(10, 3, 1, 4);

julia> size(a)
(10, 3, 1, 4)

julia> size( squeeze(a) )
(10, 3, 4)
```
"""
function squeeze(A :: AbstractArray)
    keepdims = Tuple(i for i in size(A) if i != 1)
    return reshape(A, keepdims)
end

"""
    apply_along_dim(f::Function, A::AbstractArray; dim::Integer = 1)

Apply a function, `f`, over slices along a dimension, `dim`, of a multidimensional
array, `A`.

Example
===
```julia-repl
julia> a = [ 1 2; 3 4 ]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> apply_along_dim(sum, a, dim = 1)
1×2 Matrix{Int64}:
 4  6
```
"""
function apply_along_dim(f::Function, A::AbstractArray; dim::Integer = 1)

    axs = axes(A)
    d = length(axes(A))
    @assert dim<=d "The array provided has only $d dims, but you asked for dim = $dim"

    prev_axes = axs[ 1:(dim-1) ]
    post_axes = axs[ (dim+1):end ]

    prev_ind = CartesianIndices(prev_axes)
    post_ind = CartesianIndices(post_axes)

    R = [f(A[I, :, J]) for J in post_ind for I in prev_ind]

    # Return array with modified length along 'dim'
    return reshape(R, size(prev_ind)..., :, size(post_ind)...)
end
# # This version drops the singleton dimension automatically
# function apply_along_dim(f::Function, A::AbstractArray; dim::Integer = 1)
#    return [f(slice) for slice in eachslice(A, dims = dim)]
# end

"""
    simulate_experiment(refvalue, Nmeasures = Inf)

Simulates the experimental measurement with `Nmeasures` number of tries of
an observable whose theoretical value is `refvalue`.

The experiental result is simulated by sampling a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
with `Nmeasures` tries and success rate `refvalue`, and normalizing the result against the number of tries.

Notes
=====
* If the number of measurements, `Nmeasures`, is infinite, the reference value, `refvalue`, is returned exactly.
* It is assumed that `0.0 <= refvalue <= 1.0`. If not, `refvalue` will be taken as its closest boundary.
"""
function simulate_experiment(refvalue, Nmeasures = Inf)
    if isinf(Nmeasures)
        sample = refvalue
    else
        refvalue = min(max(refvalue, zero(refvalue)), one(refvalue))
        distrib = Binomial(Nmeasures, refvalue)
        sample = rand(distrib)/Nmeasures
    end

    return sample
end
