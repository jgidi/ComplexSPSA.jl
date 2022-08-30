"""
    SPSA2_on_complex(f::Function, z₀::Vector, Niters = 200;
                     sign = -1,
                     hessian_delay = 0,
                     initial_iteration = 1,
                     constant_learning_rate = false,
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
                          initial_iteration = 1,
                          constant_learning_rate = false,
                          a = gains[:a], b = gains[:b],
                          A = gains[:A], s = gains[:s], t = gains[:t],
                          postprocess = x->x,
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
    for index in 1:Niters
        # Number of iteration
        iter = index + initial_iteration - 1

        # Estimation parameters
        ak = constant_learning_rate ? 1.0 : 1.0 / (iter + A)^s
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
        H = (index*Hsmooth + H) / (index+1)
        Hsmooth = H

        if index > hessian_delay
            # Correct gradient with the Hessian
            gr .= ( H \ gr )
        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        # Apply postprocessing to z
        z .= postprocess(z)

        zacc[:, index] = z
    end

    return zacc
end

"""
    CSPSA2(f::Function, z₀::Vector, Niters = 200;
           sign = -1,
           hessian_delay = 0,
           initial_iteration = 1,
           constant_learning_rate = false,
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
                initial_iteration = 1,
                constant_learning_rate = false,
                a = gains[:a], b = gains[:b],
                A = gains[:A], s = gains[:s], t = gains[:t],
                postprocess = x->x,
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
    for index in 1:Niters
        # Number of iteration
        iter = index + initial_iteration - 1

        # Estimation parameters
        ak = constant_learning_rate ? 1.0 : 1.0 / (iter + A)^s
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

        # Second order
        # Hessian estimation as a forward difference of the gradient
        @. zp = z + Δ1 + Δ2             # Perturb variables as reals
        @. zm = z - Δ1 + Δ2

        dfp = f(zp) - f(zm)
        H = @. (dfp - df) / conj(2Δ1 * Δ2')     # Estimate Hessian
        H = (H + H')/2                    # Symmetrization

        # Hessian conditioning

        # Regularization
        H = sqrt(H*H + 1e-3LinearAlgebra.I(Nz))

        # Smoothing
        H = (index*Hsmooth + H) / (index+1)
        Hsmooth = H

        if index > hessian_delay
            # Correct gradient with the Hessian
            g .= ( H \ g )
        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        # Apply postprocessing to z
        z .= postprocess(z)

        zacc[:, index] = z
    end

    return zacc
end


function CSPSA2_full(f::Function, z₀::Vector, Niters = 200;
                     sign = -1,
                     hessian_delay = 0,
                     initial_iteration = 1,
                     constant_learning_rate = false,
                     a = gains[:a], b = gains[:b],
                     A = gains[:A], s = gains[:s], t = gains[:t],
                     postprocess = x->x,
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
    Hsmooth = LinearAlgebra.I(2Nz)
    for index in 1:Niters
        # Number of iteration
        iter = index + initial_iteration - 1

        # Estimation parameters
        ak = constant_learning_rate ? 1.0 : 1.0 / (iter + A)^s
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

        # Second order
        # Hessian estimation as a forward difference of the gradient
        @. zp = z + Δ1 + Δ2             # Perturb variables as reals
        @. zm = z - Δ1 + Δ2

        dfp = f(zp) - f(zm)
        d2f = (dfp - df )
        # Estimate Hessian
        Δc1 = vcat(Δ1, conj(Δ1))
        Δc2 = vcat(Δ2, conj(Δ2))
        H  = @. d2f / conj(2Δc1 * Δc2')

        # Symmetrization
        H  = (H + H')/2

        # Hessian conditioning

        # Regularization
        H  = sqrt(H*H + 1e-3LinearAlgebra.I(2Nz))

        # Smoothing
        H = (index*Hsmooth + H) / (index+1)
        Hsmooth = H
        # H2 = (index*Hsmooth2 + H2) / (index+1)
        # Hsmooth2 = H2

        if index > hessian_delay
            # Correct gradient with the Hessian
            g2 = vcat(g, conj(g))
            g .= ( H \ g2 )[1:Nz]

        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        # Apply postprocessing to z
        z .= postprocess(z)

        zacc[:, index] = z
    end

    return zacc
end

# M2SPSA (Xun Zhu, 2021)
# Matrix Conditioning and Adaptive Simultaneous Perturbation Stochastic Approximation Method
function MCSPSA2(f::Function, z₀::Vector, Niters = 200;
                 sign = -1,
                 hessian_delay = 0,
                 initial_iteration = 1,
                 constant_learning_rate = false,
                 a = gains[:a], b = gains[:b],
                 A = gains[:A], s = gains[:s], t = gains[:t],
                 postprocess = x->x,
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
    for index in 1:Niters
        # Number of iteration
        iter = index + initial_iteration - 1

        # Estimation parameters
        ak = constant_learning_rate ? 1.0 : 1.0 / (iter + A)^s
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

        # Second order
        # Hessian estimation as a forward difference of the gradient
        @. zp = z + Δ1 + Δ2             # Perturb variables as reals
        @. zm = z - Δ1 + Δ2

        dfp = f(zp) - f(zm)
        H = @. (dfp - df) / conj(2Δ1 * Δ2')     # Estimate Hessian
        H = (H + H')/2                    # Symmetrization

        # Hessian conditioning

        # Regularization
        H = sqrt(H*H + 1e-3LinearAlgebra.I(Nz))

        # Smoothing
        H = (index*Hsmooth + H) / (index+1)
        Hsmooth = H

        vals = LinearAlgebra.eigvals(H)
        vals = sort(vals, rev=true)
        firstneg = findfirst(x->x<0, vals)

        if !isnothing(firstneg)     # Only if there are negative eigenvalues
            q = firstneg - 1        # Position of the smallest positive eigenvalue

            vareps = ( vals[q-1] / vals[1] )^(q-2)

            for i in q:Nz
                vals[i] = vareps * vals[i-1]
            end
        end

        h = exp.(sum(log, vals)/Nz)

        if index > hessian_delay
            # Correct gradient with the Hessian
            g .= ( h \ g )
        else
            ak = ak * a
        end

        # Update variable in-place
        @. z += sign * ak * g

        # Apply postprocessing to z
        z .= postprocess(z)

        zacc[:, index] = z
    end

    return zacc
end
