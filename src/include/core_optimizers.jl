function _first_order(f::Function, guess::AbstractVector, Niters;
                      sign = -1,
                      initial_iter = 1,
                      a = gains[:a], b = gains[:b],
                      A = gains[:A], s = gains[:s], t = gains[:t],
                      Ncalibrate = 0,
                      Nresampling = 1,
                      postprocess = identity,
                      )

    T = promote_type(Float64, eltype(guess))
    z = T.(guess)
    Nvars = length(z)

    # Per-iteration accumulator for z
    acc = Array{T}(undef, Nvars, Niters+1)
    acc[:, 1] = z

    if Ncalibrate > 0
        bk = decaying_pert_magnitude(b, t, initial_iter)
        a = calibrate_gain_a(f, z, a, bk, Ncalibrate)
    end

    for iter in 1:Niters
        k = iter + initial_iter - 1

        ak = decaying_learning_rate(a, A, s, k)
        bk = decaying_pert_magnitude(b, t, k)

        # Estimates of the gradient and Hessian
        g = estimate_g(f, z, bk, Nresampling)

        # update variable
        @. z += sign * ak * g

        # postprocessing
        z .= postprocess(z)

        # Copy current values to the accumulator
        acc[:, iter+1] = z
    end

    return acc
end

function _preconditioned(f::Function, guess::AbstractVector, Niters;
                         fidelity = nothing,
                         sign = -1,
                         initial_iter = 1,
                         a2 = 1.0,
                         a = gains[:a], b = gains[:b],
                         A = gains[:A], s = gains[:s], t = gains[:t],
                         Ncalibrate = 0,
                         Nresampling = 1,
                         postprocess = identity,
                         constant_learning_rate = false,
                         hessian_delay = 0,
                         initial_hessian = nothing,
                         regularization = 1e-3,
                         apply_hessian = apply_hessian,
                         hessian_estimate = hessian_estimate_standard,
                         hessian_postprocess = hessian_postprocess_gidi,
                         )

    T = promote_type(Float64, eltype(guess))
    z = T.(guess)
    Nvars = length(z)

    # Per-iteration accumulator for z
    acc = Array{T}(undef, Nvars, Niters+1)
    acc[:, 1] = z

    # Initial Hessian
    if isnothing(initial_hessian)
        H0 = Matrix{T}(I, Nvars, Nvars)
    end

    # Obtain an initial Hessian estimate by measurement
    if Ncalibrate > 0
        bk = decaying_pert_magnitude(b, t, initial_iter)
        g, H0 = estimate_gH(f, fidelity, z, bk, bk, Ncalibrate)
    end

    for iter in 1:Niters
        k = iter + initial_iter - 1

        ak = constant_learning_rate ? 1.0 : decaying_learning_rate(1.0, A, s, k)
        bk = decaying_pert_magnitude(b, t, k)

        # Estimates of the gradient and Hessian
        g, H = estimate_gH(f, fidelity, z, bk, bk, Nresampling)

        # Second order usually requires fixing the Hessian
        H0 = hessian_postprocess(H, H0, iter,
                                 regularization=regularization)

        # Only apply Hessian after hessian_delay
        if iter > hessian_delay
            # Preconditioned iteration
            g = apply_hessian(g, H0)
            ak *= a2
        else
            # First order iteration
            ak *= a
        end

        # Iteration
        @. z += sign * ak * g

        # Apply posibble inter-iteration postprocessing
        z .= postprocess(z)

        # Copy current values to the accumulator
        acc[:, iter+1] = z
    end

    return acc
end