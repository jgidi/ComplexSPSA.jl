module ComplexSPSA

# # Exported
# # Qiskit wrapper submodule
# export Qiskit
# export PaperPlot

# Optimzers
export SPSA_on_complex, CSPSA                     # First order

export SPSA2_on_complex, CSPSA2                   # Second order
export SPSA2_scalar_on_complex, CSPSA2_scalar     # Scalar Second order
export CSPSA2_full                                # Second order full

export SPSA_QN_on_complex, CSPSA_QN               # Natural gradient
export SPSA_QN_scalar_on_complex, CSPSA_QN_scalar # Scalar Natural gradient
export CSPSA_QN_full                              # Natural gradient full

# External dependencies
using QuantumToolkit
using LinearAlgebra, Statistics
export SPSA                     # Real
export SPSA2                    # Real
export SPSA_QN                                    # Real
export SPSA_QN_scalar                             # Real

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
