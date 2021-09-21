# ComplexSPSA

### Description
A package on julia for the implementation of stochastic optimizers of real functions on many complex variables, currently under development by [Jorge Gidi](https://plasmas-udec.netlify.app/en/authors/jgidi/) for the Quantum Information Group from the Universidad de Concepción, Chile.

### Private repository
The repository holding the source code of this package is private as it is on an early stage of development. However, you can ask for read-access by sending a mail to [jorgegidi@udec.cl](mailto:jorgegidi@udec.cl), indicating your [github](https://github.com) account.

### Installation
As the repository is private, you need to gain read-access to the repository on github. Then, you have to download the repository to your computer,
open a julia session within its folder, and install the package by means of 
```julia-repl
julia> using Pkg

julia> Pkg.add(path=".")
```
After the process ends, you can safely remove the repository from your computer. Then, from any location, you can access the package on julia by executing
```julia-repl
julia> using ComplexSPSA
```

### About the algorithms
The algorithms hereafter presented are based upon the [Simultaneous Perturbation Stochastic Optimization (SPSA)](https://www.jhuapl.edu/spsa/) method for real functions of real variables.
