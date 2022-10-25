module ComplexSPSA

# External dependencies
using QuantumToolkit
using LinearAlgebra, Statistics

# # Exported
# # Qiskit wrapper submodule
# export Qiskit
# export PaperPlot

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
include("include/estimates.jl")
include("include/core_optimizers.jl")
include("include/hessian_postprocess.jl")
include("include/implementations.jl")
include("include/tools.jl")

# # Qiskit wrapper submodule
# include("include/qiskit/Qiskit.jl")
# include("include/paperplot/PaperPlot.jl")

end # module
