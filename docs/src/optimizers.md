# Optimizers

This package provides an interface to the set optimizers presented in ["Stochastic optimization algorithms for quantum applications"](https://arxiv.org/abs/2203.06044) (Gidi _et. al._, 2022).

In the publication two groups of gain parameters are mentioned:

The default goup is
```@docs
ComplexSPSA.gains
```

and the asymptotic gains are also provided as
```@docs
ComplexSPSA.gains_asymptotic
```

The optimizers are subdivided into two categories:

## First-order optimizers


### Options

Optimizers of this category accept the arguments:

- `sign`: Specifies if the objective function should be maximized (`sign=1`) or minimized (`sign=-1`). Default value is `-1`.
- `initial_iter`: Determines the initial value of the iteration index `k`.
- `a`, `b`, `A`, `s` and `t`: The gain parameters. Default values are contained in the dictionary [`ComplexSPSA.gains`](@ref).
- `learning_rate_constant`: Specifies if the learning rate should be decaying in the iteration number `a / (k + A)^s` (`learning_rate_constant=false`) or fixed to `a` across all iterations (`learning_rate_constant=true`). Default value is `false`.
- `learning_rate_Ncalibrate`: Integer indicating how many samples to evaluate from the objective function to calibrate the leraning rate `a` as proposed by [Kandala _et. al._ (2017)](https://arxiv.org/pdf/1704.05018.pdf). Default value is `0` (no calibration).
- `blocking`: Allows to accept only variable updates which improve the value of the function up to certain tolerance. Default value is `false`.
- `blocking_tol`: The tolerance used for blocking. Default value is `0.0`.
- `blocking_Ncalibrate`: Is an integer representing how many evaluations of the function on the seed value should be used to estimate its standard deviation. If `blocking_Ncalibrate > 1`, then `blocking_tol` is overriden with the value of twice the standard deviation. The default value of `blocking_Ncalibrate` is `0`.
- `Nresampling`: Integer indicating how many times to estimate the gradient to use an average at each iteration. Default value is `1`.
- `postprocess`: A function which takes the array of variables `z` at the end of each iteration, and returns a postprocessed version of it. The default value is `identity`, which returns its arguments identically.

### Optimizers

The optimizers are

```@docs
SPSA
```

```@docs
SPSA_on_complex
```

```@docs
CSPSA
```

## Preconditioned optimizers

### Options

All of the options for first-order optimizers may also be provided. Additionally, preconditioned optimizers take the following options:

- `initial_hessian`: Allows to pass a guess for the initial value of the Hessian estimator. If not given, an Identity matrix is used.
- `hessian_delay`: An integer indicating how many iterations should be performed using a first-order optimization rule (while collecting information for the Hessian estimator) before using the Hessian estimator to precondition the gradient estimator. The default value is `0`.
- `a2`: Mimics the first-order gain parameter `a` but for preconditioned iterations. Default value is `1.0`. As in the first-order case, the keyword `learning_rate_constant` may be used to control wether the learning rate should be constant or decaying on the iteration number.
- `regularization`: A real number indicating the perturbation used on the Hessian regularization. The default value is `0.001`.

### Optimizers

Two categories are mixed within the preconditioned algorithms: Second order and Quantum Natural methods.

#### Second order

Second order methods use additional measurements of the objective function to estimate its Hessian matrix. This methods are:

```@docs
SPSA2
```

```@docs
SPSA2_on_complex
```

```@docs
CSPSA2
```

```@docs
CSPSA2_full
```

```@docs
SPSA2_scalar
```
```@docs
SPSA2_scalar_on_complex
```

```@docs
CSPSA2_scalar
```

#### Quantum Natural

Quantum Natural methods, while following a first-order update rule, consider the metric of the problem to precondition the gradient estimate.
In particular, the following optimizers require the fidelity $ \mathscr{fidelity}(z, z')$ to estimate the Fubini-Study metric:


```@docs
SPSA_QN
```

```@docs
SPSA_QN_on_complex
```

```@docs
CSPSA_QN
```

```@docs
CSPSA_QN_full
```

```@docs
SPSA_QN_scalar
```

```@docs
SPSA_QN_scalar_on_complex
```

```@docs
CSPSA_QN_scalar
```
