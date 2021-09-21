function SPSA_on_complex(f::Function, z₀::Vector, Niters = 200;
                         sign = -1,
                         Ncalibrate = 0,
                         a = gains[:a], b = gains[:b],
                         A = gains[:A], s = gains[:s], t = gains[:t],
                         )

    z = z₀[:]
    zr = reinterpret(Float64, z)        # View of z as pairs of reals

    Nvars = length(z₀)

    # Set of possible perturbations
    samples = Float64.((-1, 1))

    # Calibrate gain a
    if Ncalibrate > 0
        ac = calibrate_gain_a(f, z, a, b, samples, Ncalibrate)
        ac > 1e-10 ? (a = ac) : nothing
    end

    # Preallocate quantities
    zp = similar(z)                            # z+
    zm = similar(z)                            # z-
    zpr = reinterpret(Float64, zp)             # z+ (real view)
    zmr = reinterpret(Float64, zm)             # z- (real view)

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nvars, Niters)

    for iter in 1:Niters
        ak = a / (iter + A)^s
        bk = b / iter^t

        # Vector of 2N real perturbations
        Δ = rand(samples, 2Nvars)

        # Perturb variables as real
        @. zpr = zr + bk*Δ
        @. zmr = zr - bk*Δ

        df = f(zp) - f(zm)

        # Update variable in-place
        @. zr += 0.5sign * ak * Δ * df / bk

        # Copy arguments as complex to the accumulator
        zacc[:, iter] = z
    end

    return zacc
end

function CSPSA(f::Function, z₀::Vector, Niters = 200;
               sign = -1,
               Ncalibrate = 0,
               a = gains[:a], b = gains[:b],
               A = gains[:A], s = gains[:s], t = gains[:t],
               )

    z = z₀[:]
    Nvars = length(z)

    # Set of possible perturbations
    samples = Complex{Float64}.((-1, 1, -im, im))

    # Calibrate gain a
    if Ncalibrate > 0
        ac = calibrate_gain_a(f, z, a, b, samples, Ncalibrate)
        ac > 1e-10 ? (a = ac) : nothing
    end

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nvars, Niters)

    for iter in 1:Niters
        ak = a / (iter + A)^s
        bk = b / iter^t

        # Vector of complex perturbations
        Δ = rand(samples, Nvars)

        df = f(@. z + bk*Δ) - f(@. z - bk*Δ)

        # Update variable in-place
        @. z += 0.5sign * ak * Δ * df / bk

        # Copy arguments as complex to the accumulator
        zacc[:, iter] = z
    end

    return zacc
end
