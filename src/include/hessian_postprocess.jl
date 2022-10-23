# Hermitizations
hermitize(H) = 0.5(H + H')

# Regularization
## Standard
regularize(H::AbstractMatrix, eps) = sqrt(H'H) + UniformScaling(eps)
regularize_insquare(H::AbstractMatrix, eps) = sqrt(H'H + UniformScaling(eps))
## Scalar
regularize(h::Real, eps) = abs(h) + eps


# Inertia collection
collect_inertia(H, H0, iter) = (iter*H0 + H) / (iter + 1)

# Hessian construction
hessian_estimate_standard(h, Δ1, Δ2) = @. h * Δ1 * Δ2'
hessian_estimate_scalar(h, Δ1, Δ2) = h
hessian_estimate_full(h, Δ1, Δ2) = h .* vcat(Δ1, conj(Δ1)) .* vcat(Δ2, conj(Δ2))'

# Hessian postprocessing
function hessian_postprocess_gidi(H, H0, iter;
                                  regularization=1e-3)
    H = hermitize(H)
    H = regularize_insquare(H, regularization)
    H = collect_inertia(H, H0, iter)

    return H
end

function hessian_postprocess_spall(H, H0, iter;
                                   regularization=1e-3)
    H = hermitize(H)
    H = collect_inertia(H, H0, iter)
    H = regularize(H, regularization)

    return H
end

# Hessian application to gradient
apply_hessian(g, H0) = H0 \ g
apply_hessian_full(g, H0) = (H0 \ vcat(g, conj(g)))[1:length(g)]
