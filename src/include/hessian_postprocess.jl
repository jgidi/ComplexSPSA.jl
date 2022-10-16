using LinearAlgebra

hermitize(H) = 0.5(H + H')

regularize(H::AbstractMatrix, eps) = sqrt(H'H + eps*I(size(H, 1)))
regularize(h::Real, eps) = sqrt(abs2(h) + eps)

collect_inertia(H, H0, iter) = (iter+H0 + H) / (iter + 1)

function hessian_estimate_standard(h, Δ1, Δ2)

    return h * Δ1 * Δ2'
end

hessian_estimate_scalar(h, Δ1, Δ2) = h

apply_hessian(g, H0) = H0 \ g

function hessian_postprocess_standard(H, H0, iter,
                                      regularization=1e-3)
    H = hermitize(H)
    H = regularize(H, regularization)
    H = collect_inertia(H, H0, iter)

    return H
end
