## Real-variable optimizers

Simple, thin wrappers around the SPSA-based optimizers implemented by [Qiskit](https://qiskit.org). This optimizers only work on real functions of real variables.

```@autodocs
Modules = [ComplexSPSA.Qiskit]
Pages = ["include/qiskit/include/real_optimizers.jl"]
```

## Complex-variable optimizers

Optimizers which separate the optimization problem of a real function of $N$ complex variables
as the equivalent problem of optimizing a real function of $2N$ real variables, and then solve it
by means of the [Real-variable optimizers](@ref).

```@autodocs
Modules = [ComplexSPSA.Qiskit]
Pages = ["include/qiskit/include/complex_optimizers.jl"]
```
