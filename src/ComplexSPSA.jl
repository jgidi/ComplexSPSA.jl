module ComplexSPSA

# Exported
# Qiskit wrapper submodule
export Qiskit
# First order optimzers
export SPSA_on_complex, CSPSA
# Second order optimizers
export SPSA2_on_complex, CSPSA2
# Natural gradient optimizers
export SPSA_NG_on_complex, CSPSA_NG, CSPSA_NG_scalar
# Tools
export squeeze, apply_along_dim, simulate_experiment

# External dependencies
import LinearAlgebra
import Statistics
import Distributions: Binomial

include("include/first_order.jl")
include("include/second_order.jl")
include("include/natural_gradient.jl")
include("include/tools.jl")

# Qiskit wrapper submodule
include("include/qiskit/Qiskit.jl")

end # module
