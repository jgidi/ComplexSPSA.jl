module ComplexSPSA

# Exported
# Qiskit wrapper submodule
export Qiskit
export PaperPlot

# Optimzers
export SPSA_on_complex, CSPSA                     # First order

export SPSA2_on_complex, CSPSA2                   # Second order
export SPSA2_scalar_on_complex, CSPSA2_scalar     # Scalar Second order
export MCSPSA2, CSPSA2_full                       #

export SPSA_QN_on_complex, CSPSA_QN               # Natural gradient
export SPSA_QN_scalar_on_complex, CSPSA_QN_scalar # Scalar Natural gradient

# External dependencies
import LinearAlgebra
import Statistics
using QuantumToolkit

include("include/gains.jl")
include("include/first_order.jl")
include("include/second_order.jl")
include("include/second_order_scalar.jl")
include("include/natural_gradient.jl")
include("include/natural_gradient_scalar.jl")
include("include/tools.jl")

# Qiskit wrapper submodule
include("include/qiskit/Qiskit.jl")
include("include/paperplot/PaperPlot.jl")

end # module
