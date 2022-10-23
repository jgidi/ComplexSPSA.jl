perturbation_samples(::Type{<:Real}) = Float64.((-1, 1))
perturbation_samples(::Type{<:Complex}) = ComplexF64.((-1, 1, -im, im))

# First order
function estimate_g(f, z, bk,
                    Nresampling::Integer = 1,
                    )

    Nvars, T = length(z), eltype(z)
    g = zeros(T, Nvars)
    samples = perturbation_samples(T)

    for _ in 1:Nresampling
        Δ = rand(samples, Nvars)

        df = f(@. z + bk*Δ)/bk - f(@. z - bk*Δ)/bk

        @. g += 0.5df * Δ
    end
    g ./= Nresampling

    return g
end

# Second order
function estimate_gH(f::Function, fidelity::Nothing, z,
                     bk1::Real, bk2::Real, Nresampling::Integer,
                     hessian_estimate=hessian_estimate_standard,
                     )

    Nvars, T = length(z), eltype(z)

    g = zeros(T, Nvars)
    H = zero(T)
    samples = perturbation_samples(T)

    for _ in 1:Nresampling
        Δ1 = rand(samples, Nvars)
        Δ2 = rand(samples, Nvars)

        # First order
        zp = @. z + bk1*Δ1
        zm = @. z - bk1*Δ1
        df = f(zp)/bk1 - f(zm)/bk1

        @. g += 0.5df * Δ1

        # Preconditioning
        b2 = bk1 * bk2

        dfp =  f(@. zp + bk2*Δ2)/b2
        dfp -= f(@. zm + bk2*Δ2)/b2

        d2f = 0.5(dfp - df/bk2)

        H = H .+ hessian_estimate(d2f, Δ1, Δ2)
    end
    g ./= Nresampling
    H /= Nresampling

    return g, H
end

# Natural gradient
function estimate_gH(f::Function, fidelity::Function, z,
                     bk1::Real, bk2::Real, Nresampling::Integer,
                     hessian_estimate=hessian_estimate_standard,
                     )

    Nvars, T = length(z), eltype(z)

    g = zeros(T, Nvars)
    H = zero(T)
    samples = perturbation_samples(T)

    for _ in 1:Nresampling
        Δ1 = rand(samples, Nvars)
        Δ2 = rand(samples, Nvars)

        # First order
        zp = @. z + bk1*Δ1
        zm = @. z - bk1*Δ1
        df = f(zp)/bk1 - f(zm)/bk1

        @. g += 0.5df * Δ1

        # Preconditioning
        b2 = bk1 * bk2

        df  = fidelity(z, zp)/b2 - fidelity(z, zm)/b2

        dfp =  fidelity(z, @. zp + bk2*Δ2)/b2
        dfp -= fidelity(z, @. zm + bk2*Δ2)/b2

        d2f = -0.25(dfp - df)

        H = H .+ hessian_estimate(d2f, Δ1, Δ2)
    end
    g ./= Nresampling
    H /= Nresampling

    return g, H
end
