# Require the library
using ComplexSPSA

# Define some objective function.
# In this case, a simple example given by | z - i |^2,
# with its minimum at z = i
f(z) = abs2( z[1] - 1im )

# Optimize using CSPSA, starting from z0 = [0.0] and iterating 50 times.
# Remember that the function 'f' must take an array as input and output a real scalar,
# and, therefore, the seed must be also an array.
# The output 'z' is an array with 2 dimensions, where z[i, j] is the
# i-th complex variable optimized at the j-th iteration.
z = CSPSA(f, [0.0], 10)

# Evaluate the function 'f' along the first dimension of 'z'.
# The result will be an array of the objective function at each iteration
# (that is, an array with length 10).
fz = [f(z[:, i]) for i in axes(z, 2)]

# Since the function under optimization is very simple, 10 iterations
# should be enough to make last value of 'fz' approximately 0.
@show fz[end]
