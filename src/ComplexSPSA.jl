module ComplexSPSA

export Qiskit
# Tools
export squeeze, apply_along_dim, simulate_experiment

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
