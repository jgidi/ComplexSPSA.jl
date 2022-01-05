"""
    SPSA2_on_complex(f::Function, z₀::Vector, Niters = 200;
                     sign = -1,
                     hessian_delay = 0,
                     a = gains[:a], b = gains[:b],
                     A = gains[:A], s = gains[:s], t = gains[:t],
                     )

The second-order SPSA, commonly referred to as `2-SPSA` method is a second-order stochastic optimization method
based on [SPSA](https://www.jhuapl.edu/spsa/), which additional to a gradient estimate performs a Hessian correction
on the update rule to optimize real-valued functions of a number of real variables.

This function performs second-order SPSA optimization of the real-valued function `f` of complex variables by treating each complex variable
as a pair of real variables, starting from the complex vector `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`, with size `(length(z₀), Niters)`,
such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Since second-order effects usually show improvements once the seed value is closer to a local minimum,
it is possible to accept a number `hessian_delay` of first-order iterations before including the application of the Hessian information.

Notes
===
* The value of `a` is only required to perform a possible number of initial first-order iterations (via `hessian_delay`), since the second-order iterations yield an optimum value for `a = 1`.
"""
function SPSA2_on_complex(f::Function, z₀::Vector, Niters = 200;
                          sign = -1,
                          hessian_delay = 0,
                          a = gains[:a], b = gains[:b],
                          A = gains[:A], s = gains[:s], t = gains[:t],
                          )

    z = z₀[:] .+ 0im
    zr = reinterpret(Float64, z)        # View of z as pairs of reals

    grad = zeros(ComplexF64, size(z₀))
    gradr = reinterpret(Float64, grad)

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

    # Gradient Accumulator
    gacc = Array{Complex{Float64}}(undef, Nz, Niters)

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
        @. zpr = zr + Δ1                  # Perturb variables as reals
        @. zmr = zr - Δ1

        df = f(zp) - f(zm)
        @. gr = df / (2Δ1)

        if iter > hessian_delay
            # Second order
            # Hessian estimation as a forward difference of the gradient
            @. zpr = zr + Δ1 + Δ2             # Perturb variables as reals
            @. zmr = zr - Δ1 + Δ2

            dfp = f(zp) - f(zm)
            H = @. (dfp - df) / (2Δ1*Δ2')     # Estimate Hessian
            H = (H + H')/2                    # Symmetrization

            # Hessian conditioning

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

        # Define Gradient
        @. gradr = ak * g

        zacc[:, iter] = z
        gacc[:, iter] = grad
    end

    return zacc, gacc
end

"""
    CSPSA2(f::Function, z₀::Vector, Niters = 200;
           sign = -1,
           hessian_delay = 0,
           a = gains[:a], b = gains[:b],
           A = gains[:A], s = gains[:s], t = gains[:t],
           )

The second-order CSPSA method, CSPSA2, is a second-order stochastic optimization method
based on [`CSPSA`](@ref), which additional to a gradient estimate performs a Hessian correction
on the update rule to optimize real-valued functions of a number of complex variables.

This function performs second-order CSPSA optimization of the real-valued function `f` of complex variables,
starting from the complex vector `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`,
with size `(length(z₀), Niters)`, such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If they are not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Since second-order effects usually show improvements once the seed value is closer to a local minimum,
it is possible to accept a number `hessian_delay` of first-order iterations before including the application of the Hessian information.

Notes
===
* The value of `a` is only required to perform a possible number of initial first-order iterations (via `hessian_delay`), since the second-order iterations yield an optimum value for `a = 1`.
"""
function CSPSA2(f::Function, z₀::Vector, Niters = 200;
                sign = -1,
                hessian_delay = 0,
                a = gains[:a], b = gains[:b],
                A = gains[:A], s = gains[:s], t = gains[:t],
                )

    z = z₀[:] .+ 0im
    Nz = length(z)

    grad = zeros(ComplexF64, size(z₀))

    # Set of possible perturbations
    samples = Complex{Float64}.((-1, 1, -im, im))

    # Preallocate some quantities and views
    g  = similar(z)
    zp = similar(z)
    zm = similar(z)

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nz, Niters)

    # Gradient Accumulator
    gacc = Array{Complex{Float64}}(undef, Nz, Niters)

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
            # Second order
            # Hessian estimation as a forward difference of the gradient
            @. zp = z + Δ1 + Δ2             # Perturb variables as reals
            @. zm = z - Δ1 + Δ2

            dfp = f(zp) - f(zm)
            H = @. (dfp - df) / (2Δ1*Δ2')     # Estimate Hessian
            H = (H + H')/2                    # Symmetrization

            # Hessian conditioning

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

        # Define Gradient
        @. grad = ak * g

        zacc[:, iter] = z
        gacc[:, iter] = grad
    end

    return zacc, gacc
end
