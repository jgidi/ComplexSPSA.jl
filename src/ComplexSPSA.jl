module ComplexSPSA

# External dependencies
using LinearAlgebra, Statistics

# Optimzer exports

# First order
export SPSA                     # Real
export SPSA_on_complex, CSPSA   # Complex

# Second order
export SPSA2                    # Real
export SPSA2_on_complex, CSPSA2 # Complex
# Scalar
export SPSA2_scalar                           # Real
export SPSA2_scalar_on_complex, CSPSA2_scalar # Complex
# Full
export CSPSA2_full              # Only valid for Complex args

# Quantum Natural
export SPSA_QN                                    # Real
export SPSA_QN_on_complex, CSPSA_QN               # Complex
# Scalar
export SPSA_QN_scalar                             # Real
export SPSA_QN_scalar_on_complex, CSPSA_QN_scalar # Complex
# Full
export CSPSA_QN_full                              # Only valid for complex args

include("include/gains.jl")
include("include/estimators.jl")
include("include/core_optimizers.jl")
include("include/hessian_postprocess.jl")
include("include/implementations.jl")
include("include/tools.jl")

end # module
