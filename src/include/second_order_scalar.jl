function SPSA2_scalar_on_complex(f::Function, z₀::Vector, Niters = 200;
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
    Hsmooth = one(Float64)
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
        d2f = abs(dfp - df) / (2bk^2)

        # Smoothing
        Hsmooth = (index*Hsmooth + d2f) / (index+1)

        if index > hessian_delay
            # Correct gradient with the Hessian
            gr .= ( Hsmooth \ gr )
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

function CSPSA2_scalar(f::Function, z₀::Vector, Niters = 200;
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
    Hsmooth = one(z[1])
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
        d2f = abs(dfp - df) / (2bk^2)

        Hsmooth = (index*Hsmooth + d2f) / (index + 1)


        if index > hessian_delay
            # Correct gradient with the Hessian
            g .= ( Hsmooth \ g )

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
