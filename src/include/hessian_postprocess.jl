# Hermitizations
hermitize(H) = 0.5(H + H')

# Regularization
## Standard
regularize(H::AbstractMatrix, eps) = sqrt(H'H + eps*I(size(H, 1)))
## Scalar
regularize(h::Real, eps) = sqrt(abs2(h) + eps)


# Inertia collection
collect_inertia(H, H0, iter) = (iter+H0 + H) / (iter + 1)

# Hessian construction
hessian_estimate_standard(h, Δ1, Δ2) = h * Δ1*Δ2'
hessian_estimate_scalar(h, Δ1, Δ2) = h

# Hessian postprocessing
function hessian_postprocess_gidi(H, H0, iter,
                                  regularization=1e-3)
    H = hermitize(H)
    H = regularize(H, regularization)
    H = collect_inertia(H, H0, iter)

    return H
end

function hessian_postprocess_spall(H, H0, iter,
                                   regularization=1e-3)
    H = hermitize(H)
    H = collect_inertia(H, H0, iter)
    H = regularize(H, regularization)

    return H
end

# Hessian application to gradient
apply_hessian(g, H0) = H0 \ g
