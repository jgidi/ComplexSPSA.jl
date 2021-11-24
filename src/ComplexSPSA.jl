module ComplexSPSA

# Exported
# Qiskit wrapper submodule
export Qiskit
# Optimzers
export SPSA_on_complex, CSPSA                     # First order
export SPSA2_on_complex, CSPSA2                   # Second order
export SPSA_QN_on_complex, CSPSA_QN               # Natural gradient
export SPSA_QN_scalar_on_complex, CSPSA_QN_scalar # Scalar Natural gradient
# Tools
export squeeze, apply_along_dim, simulate_experiment

# External dependencies
import LinearAlgebra
import Statistics
import Distributions: Binomial

include("include/gains.jl")
include("include/first_order.jl")
include("include/second_order.jl")
include("include/natural_gradient.jl")
include("include/tools.jl")

# Qiskit wrapper submodule
include("include/qiskit/Qiskit.jl")

end # module
