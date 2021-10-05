# Optimizers

This package provides a set optimizers for real-valued functions of a number of complex variables, $f:\mathbb{C}^N\to\mathbb{R}$, which we subdivide into three categories

## First-order optimizers

To reach a local minimum of $f(\bm z)$ with $\bm z \in \mathbb{C}^N$, this methods start from some seed value of $\bm z = \bm z^0$ and iterate over $k$ performing a first-order update,
```math
\bm z^{k+1} = \bm z^k - a^k \bm g^{k}(\bm z^k, b^k),
```

where $\bm g^k(\bm z^k, b^k)$ is a randomly directed finite-difference estimate of the gradient of $f$, $\partial f / \partial\bm{z}$, and
```math
\begin{aligned}
a^k &= \frac{a}{(k + A + 1)^s}, \\
b^k &= \frac{b}{(k + 1)^t},
\end{aligned}
```
are convergence factors determined by the input gain parameters $a$, $b$, $A$, $s$, and $t$.

By default, standard gains are defined and used package-wide from the dictionary [`ComplexSPSA.gains`](@ref),
```@docs
ComplexSPSA.gains
```
where they can be changed on run-time, and the optimizers will use their value as defined at the moment when they are called.

### SPSA on complex

In particular, two first-order methods are implemented. The first, called [`SPSA_on_complex`](@ref), is based on the [SPSA optimization method](https://www.jhuapl.edu/spsa/) for real variables of real functions, and works by treating each complex variable as a pair of real variables. This method is commonly used for the minimization of real functions of complex variables, but may result cumbersome on domains where the complex algebra is natively involved.
```@docs
SPSA_on_complex
```

### CSPSA

A second method called [`CSPSA`](@ref) is also provided, which has the advantage of having being formulated in terms of complex variables, thus resulting more natural in natively-complex domains.
```@docs
CSPSA
```


## Second-order optimizers

```@docs
SPSA2_on_complex
```


```@docs
CSPSA2
```

## Natural gradient
