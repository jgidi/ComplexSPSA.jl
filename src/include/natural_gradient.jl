"""
    SPSA_QN_on_complex(f::Function, z₀::Vector, Niters = 200;
                       sign = -1,
                       hessian_delay = 0,
                       a = gains[:a], b = gains[:b],
                       A = gains[:A], s = gains[:s], t = gains[:t],
                       )

The Quantum Natural SPSA, presented by [Gacon _et. al_. (2021)](https://arxiv.org/abs/2103.09232), is a second-order stochastic optimization method
based on [2-SPSA](https://www.jhuapl.edu/spsa/), where the second-order correction comes from the Hessian of the Fubini-Study metric of the problem
instead of the Hessian of the function under optimization.

Note that the metric must be a function taking two input vectors, and returning minus a half of the fidelity between the states
generated each from one input,

\$ \\text{metric}(\\vec z₁, \\vec z₂) = -\\frac{1}{2} |\\langle ψ(\\vec z₁) | ψ(\\vec z₂) \\rangle|^2, \$

where \$ ψ(\\vec z) \$ is the quantum state parameterized with the variables \$ \\vec z \$.

This function performs Quantum Natural SPSA optimization of the real-valued function `f` of complex variables by treating each complex variable
as a pair of real variables, starting from the complex vector `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`, with size
`(length(z₀), Niters)`, such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Since second-order effects usually show improvements once the seed value is closer to a local minimum,
it is possible to accept a number `hessian_delay` of first-order iterations before including the application of the Hessian information.

Notes
===
* The value of `a` is only required to perform a possible number of initial first-order iterations (via `hessian_delay`), since the second-order iterations yield an optimum value for `a = 1`.
"""
function SPSA_QN_on_complex(f::Function, metric::Function, z₀::Vector, Niters = 200;
                            sign = -1,
                            hessian_delay = 0,
                            a = gains[:a], b = gains[:b],
                            A = gains[:A], s = gains[:s], t = gains[:t],
                            )

    z = z₀[:] .+ 0im
    zr = reinterpret(Float64, z)        # View of z as pairs of reals
    Nz = length(z)

    # Set of possible perturbations
    samples = Float64.((-1, 1))

    # Preallocate some quantities and views
    g  = similar(z)
    zp = similar(z)
    zm = similar(z)
    gr  = reinterpret(Float64, g)
    zpr = reinterpret(Float64, zp)
    zmr = reinterpret(Float64, zm)

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nz, Niters)

    # Initial Hessian
    Hsmooth = LinearAlgebra.I(2Nz)
    for iter in 1:Niters
        ak = 1 / (iter + A)^s
        bk = b / iter^t

        # Perturbations
        Δ1 = bk*rand(samples, 2Nz)
        Δ2 = bk*rand(samples, 2Nz)

        # First order
        # Gradient estimation as a central difference of the loss function
        @. zpr = zr + Δ1                  # Perturb variables
        @. zmr = zr - Δ1

        df = f(zp) - f(zm)
        @. gr = df / (2Δ1)

        if iter > hessian_delay
            dF = metric(z, zp) - metric(z, zm)

            # Second order
            # Hessian estimation as a forward difference of the gradient
            @. zpr = zr + Δ1 + Δ2
            @. zmr = zr - Δ1 + Δ2

            dFp = metric(z, zp) - metric(z, zm)
            H = @. (dFp - dF) / (2Δ1*Δ2')     # Estimate Hessian
            H = (H + H')/2                    # Symmetrization

            # Regularization
            H = sqrt(H*H + 1e-3LinearAlgebra.I(2Nz))

            # Smoothing
            H = (iter*Hsmooth + H) / (iter+1)
            Hsmooth = H

            # Correct gradient with the Hessian
            gr .= ( H \ gr )
        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        zacc[:, iter] = z
    end

    return zacc
end

"""
    CSPSA_QN(f::Function, metric::Function, z₀::Vector, Niters = 200;
             sign = -1,
             hessian_delay = 0,
             a = gains[:a], b = gains[:b],
             A = gains[:A], s = gains[:s], t = gains[:t],
             )

The Quantum Natural CSPSA (QN-CSPSA), is a second-order stochastic optimization method which, analogous to the [Quantum Natural SPSA by Gacon _et. al_. (2021)](https://arxiv.org/abs/2103.09232),
takes into account a stochastic approximation of the Fubiny-Study metric instead of the usual Hessian correction from [`CSPSA2`](@ref).
However, the main difference between QN-CSPSA and QN-SPSA is that the former is natively formulated in terms of complex variables, while the latter
requires real variables. Note that the metric must be a function taking two input vectors, and returning minus a half of the fidelity between the states
generated each from one input,

\$ \\text{metric}(\\vec z₁, \\vec z₂) = -\\frac{1}{2} |\\langle ψ(\\vec z₁) | ψ(\\vec z₂) \\rangle|^2, \$

where \$ ψ(\\vec z) \$ is the quantum state parameterized with the variables \$ \\vec z \$.

This function performs Quantum Natural CSPSA optimization of the real-valued function `f` of complex variables,
starting from the complex vector `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`,
with size `(length(z₀), Niters)`, such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Since second-order effects usually show improvements once the seed value is closer to a local minimum,
it is possible to accept a number `hessian_delay` of first-order iterations before including the application of the Hessian information.

Notes
===
* The value of `a` is only required to perform a possible number of initial first-order iterations (via `hessian_delay`), since the second-order iterations yield an optimum value for `a = 1`.
"""
function CSPSA_QN(f::Function, metric::Function, z₀::Vector, Niters = 200;
                  sign = -1,
                  hessian_delay = 0,
                  a = gains[:a], b = gains[:b],
                  A = gains[:A], s = gains[:s], t = gains[:t],
                  )

    z = z₀[:] .+ 0im
    Nz = length(z)

    # Set of possible perturbations
    samples = Complex{Float64}.((-1, 1, -im, im))

    # Preallocate some quantities and views
    g  = similar(z)
    zp = similar(z)
    zm = similar(z)

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nz, Niters)

    # Initial Hessian
    Hsmooth = LinearAlgebra.I(Nz)
    for iter in 1:Niters
        ak = 1 / (iter + A)^s
        bk = b / iter^t

        # Perturbations
        Δ1 = bk*rand(samples, Nz)
        Δ2 = bk*rand(samples, Nz)

        # First order
        # Gradient estimation as a central difference of the loss function
        @. zp = z + Δ1                  # Perturb variables
        @. zm = z - Δ1

        df = f(zp) - f(zm)
        @. g = df / (2conj(Δ1))

        if iter > hessian_delay
            dF = metric(z, zp) - metric(z, zm)

            # Second order
            # Hessian estimation as a forward difference of the gradient
            @. zp = z + Δ1 + Δ2
            @. zm = z - Δ1 + Δ2

            dFp = metric(z, zp) - metric(z, zm)
            H = @. (dFp - dF) / (2Δ1*Δ2')     # Estimate Hessian
            H = (H + H')/2                    # Symmetrization

            # Regularization
            H = sqrt(H*H + 1e-3LinearAlgebra.I(Nz))

            # Smoothing
            H = (iter*Hsmooth + H) / (iter+1)
            Hsmooth = H

            # Correct gradient with the Hessian
            g .= ( H \ g )
        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        zacc[:, iter] = z
    end

    return zacc
end

"""
    CSPSA_QN_scalar(f::Function, metric::Function, z₀::Vector, Niters = 200;
                    sign = -1,
                    hessian_delay = 0,
                    a = gains[:a], b = gains[:b],
                    A = gains[:A], s = gains[:s], t = gains[:t],
                    )

The Quantum Natural scalar CSPSA (QN-CSPSA scalar) is a method based upon QN-CSPSA, which avoids matrix operations by discarding the 2-dimensional
perturbation distribution of the Hessian matrix and only retaining its scalar factor. *This method is currently experimental.*

Note that the metric must be a function taking two input vectors, and returning minus a half of the fidelity between the states
generated each from one input,

\$ \\text{metric}(\\vec z₁, \\vec z₂) = -\\frac{1}{2} |\\langle ψ(\\vec z₁) | ψ(\\vec z₂) \\rangle|^2, \$

where \$ ψ(\\vec z) \$ is the quantum state parameterized with the variables \$ \\vec z \$.

This function performs scalar-appriximated Quantum Natural CSPSA optimization of the real-valued function `f` of complex variables,
starting from the complex vector `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`,
with size `(length(z₀), Niters)`, such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Since second-order effects usually show improvements once the seed value is closer to a local minimum,
it is possible to accept a number `hessian_delay` of first-order iterations before including the application of the Hessian information.

Notes
===
* The value of `a` is only required to perform a possible number of initial first-order iterations (via `hessian_delay`), since the second-order iterations yield an optimum value for `a = 1`.
"""
function CSPSA_QN_scalar(f::Function, metric::Function, z₀::Vector, Niters = 200;
                         sign = -1,
                         hessian_delay = 0,
                         a = gains[:a], b = gains[:b],
                         A = gains[:A], s = gains[:s], t = gains[:t],
                         )

    z = z₀[:] .+ 0im
    Nz = length(z)

    # Set of possible perturbations
    samples = Complex{Float64}.((-1, 1, -im, im))

    # Preallocate some quantities and views
    g  = similar(z)
    zp = similar(z)
    zm = similar(z)

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nz, Niters)

    # Initial Hessian
    Hsmooth = 1.0
    for iter in 1:Niters
        ak = 1 / (iter + A)^s
        bk = b / iter^t

        # Perturbations
        Δ1 = bk*rand(samples, Nz)
        Δ2 = bk*rand(samples, Nz)

        # First order
        # Gradient estimation as a central difference of the loss function
        @. zp = z + Δ1                  # Perturb variables
        @. zm = z - Δ1

        df = f(zp) - f(zm)
        @. g = df / (2conj(Δ1))

        if iter > hessian_delay
            dF = metric(z, zp) - metric(z, zm)

            # Second order
            # Hessian estimation as a forward difference of the gradient
            @. zp = z + Δ1 + Δ2
            @. zm = z - Δ1 + Δ2

            dFp = metric(z, zp) - metric(z, zm)
            H = abs(dFp - dF) / (2bk^2) # Estimate Hessian # NOTE Erased the factor 1/4

            # Smoothing
            H = (iter*Hsmooth + H) / (iter+1)
            Hsmooth = H

            # Correct gradient with the Hessian
            g .= ( H \ g )
        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        zacc[:, iter] = z
    end

    return zacc
end
