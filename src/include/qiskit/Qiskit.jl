module Qiskit

using PyCall

gains = Dict(
    :a => 3,
    :b => 0.1,
    :A => 1.0,
    :s => 1.0,
    :t => 1/6,
)

include("include/real_optimizers.jl")
include("include/complex_optimizers.jl")
include("include/tools.jl")


end #module
