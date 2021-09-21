function SPSA_NG_on_complex(f::Function, metric::Function, z₀::Vector, Niters = 200;
                            sign = -1,
                            hessian_delay = 0,
                            b = gains[:b],
                            A = gains[:A], s = gains[:s], t = gains[:t],
                            )

    z = copy(z₀)
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

            # Hessian conditioning
            # TODO In Qiskit they save the smoothed version and then regularize
            # https://qiskit.org/documentation/_modules/qiskit/algorithms/optimizers/spsa.html#SPSA

            # Regularization
            H = sqrt(H*H + 1e-3LinearAlgebra.I(2Nz))

            # Smoothing
            H = (iter*Hsmooth + H) / (iter+1)
            Hsmooth = H

            # Correct gradient with the Hessian
            # TODO: try
            # ldiv!(cholesky(H), g)
            # or
            # pinv(H) * g
            #LinearAlgebra.ldiv!(LinearAlgebra.cholesky(H), gr)
            gr .= ( H \ gr )
        end

        # Update variable in-place
        @. z += sign * ak * g

        zacc[:, iter] = z
    end

    return zacc
end

function CSPSA_NG(f::Function, metric::Function, z₀::Vector, Niters = 200;
                  sign = -1,
                  hessian_delay = 0,
                  b = gains[:b],
                  A = gains[:A], s = gains[:s], t = gains[:t],
                  )

    z = copy(z₀)
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

            # Hessian conditioning
            # TODO In Qiskit they save the smoothed version and then regularize
            # https://qiskit.org/documentation/_modules/qiskit/algorithms/optimizers/spsa.html#SPSA

            # Regularization
            H = sqrt(H*H + 1e-3LinearAlgebra.I(Nz))

            # Smoothing
            H = (iter*Hsmooth + H) / (iter+1)
            Hsmooth = H

            # Correct gradient with the Hessian
            # TODO: try
            # ldiv!(cholesky(H), g)
            # or
            # pinv(H) * g
            #LinearAlgebra.ldiv!(LinearAlgebra.cholesky(H), gr)
            g .= ( H \ g )
        end

        # Update variable in-place
        @. z += sign * ak * g

        zacc[:, iter] = z
    end

    return zacc
end

function CSPSA_NG_scalar(f::Function, metric::Function, z₀::Vector, Niters = 200;
                         sign = -1,
                         hessian_delay = 0,
                         b = gains[:b],
                         A = gains[:A], s = gains[:s], t = gains[:t],
                         )

    z = copy(z₀)
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

            # Hessian conditioning
            # TODO In Qiskit they save the smoothed version and then regularize
            # https://qiskit.org/documentation/_modules/qiskit/algorithms/optimizers/spsa.html#SPSA

            # Smoothing
            H = (iter*Hsmooth + H) / (iter+1)
            Hsmooth = H

            # Correct gradient with the Hessian
            # TODO: try
            # ldiv!(cholesky(H), g)
            # or
            # pinv(H) * g
            #LinearAlgebra.ldiv!(LinearAlgebra.cholesky(H), gr)
            g .= ( H \ g )
        end

        # Update variable in-place
        @. z += sign * ak * g

        zacc[:, iter] = z
    end

    return zacc
end
