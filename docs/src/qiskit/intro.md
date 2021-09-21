# Qiskit Wrapper

For purposes of ease of comparison, the ComplexSPSA module includes a Qiskit submodule which wraps some SPSA-based optimizers from the python library [Qiskit](https://qiskit.org), and exposes them with an interface common to the rest of the module.
The wrappers use the default options as defined by Qiskit for each optimizer.

### Index of optimizers implemented
```@index
Pages = ["optimizers.md"]
```

### Ensure julia can find Qiskit
To make sure that julia can find and use the Qiskit library, you can execute the function
```@docs
ComplexSPSA.Qiskit.pip_install_dependencies
```

which is provided by the Qiskit submodule, and may thus be executed as
```julia
using ComplexSPSA
ComplexSPSA.Qiskit.pip_install_dependencies()
```

### Using the optimizers 

To expose the methods defined on the Qiskit submodule, you can run
```julia
using ComplexSPSA.Qiskit
```
and then, use the optimizers as
```julia
x = Qiskit.SPSA(f, guess, Niters)
```
assuming you have previously defined a valid function, `f`, initial array of parameters, `guess`, and number of iterations, `Niters`.

