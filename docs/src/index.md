# ComplexSPSA

### Description
A package on [Julia](https://julialang.org/) for the implementation of stochastic optimizers of real functions on many complex variables, currently under development by [Jorge Gidi](https://plasmas-udec.netlify.app/en/authors/jgidi/) for the Quantum Information Group from the Universidad de Concepción, Chile.

This package is still on an early stage of development.

### Installation

While this package is available on [github](https://github.com/jgidi/ComplexSPSA.jl), it has not (yet) been registered into the Julia ecosystem. Anyways, Julia is capable of cloning, installing and keeping track of this package automatically. To install it, just open a Julia session in a terminal or a Jupyter Notebook, and run
```julia
using Pkg
pkg"add https://github.com/jgidi/ComplexSPSA.jl"
```

Assuming the process has ended successfully, ComplexSPSA.jl will be installed and you can now access its functionalities by means of
```julia
using ComplexSPSA
```

and may be updated, as any other package, with
```julia
using Pkg
pkg"update ComplexSPSA"
```

### About the algorithms
The algorithms hereafter presented are based upon the [Simultaneous Perturbation Stochastic Optimization (SPSA)](https://www.jhuapl.edu/spsa/) method for real functions of real variables.
