"""
    SPSA_on_complex(f::Function, z₀::Vector, Niters = 200;
                    sign = -1,
                    Ncalibrate = 0,
                    a = gains[:a], b = gains[:b],
                    A = gains[:A], s = gains[:s], t = gains[:t],
                    )

The [Simultaneous Perturbation Stochastic Approximation (SPSA)](https://www.jhuapl.edu/spsa/)
optimizer is a stochastic method for optimizing real functions of a number of real variables.

This function performs SPSA optimization of the real-valued function `f` of complex variables by treating each complex varible
as a pair of real variables, starting from the complex vector `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`, with size `(length(z₀), Niters)`,
such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If they are not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Automatic calibration for the input parameter `a` can be accomplished taking `Ncalibrate` samples using the method defined on [Kandala _et. al._ (2017), Sec. 11](https://arxiv.org/pdf/1704.05018v2.pdf#section*.11).
By default, the calibration is disabled (`Ncalibrate = 0`).
"""
function SPSA_on_complex(f::Function, z₀::Vector, Niters = 200;
                         sign = -1,
                         Ncalibrate = 0,
                         a = gains[:a], b = gains[:b],
                         A = gains[:A], s = gains[:s], t = gains[:t],
                         )

    z = z₀[:] .+ 0im
    zr = reinterpret(Float64, z)        # View of z as pairs of reals

    g = zeros(ComplexF64, size(z₀))
    gr = reinterpret(Float64, g)

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

    # Gradient Accumulator
    gacc = Array{Complex{Float64}}(undef, Nvars, Niters)

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

        # Define Gradient
        @. gr = 0.5sign * ak * Δ * df / bk

        # Copy arguments as complex to the accumulator
        zacc[:, iter] = z
        gacc[:, iter] = g
    end

    return zacc, gacc
end

"""
    CSPSA(f::Function, z₀::Vector, Niters = 200;
          sign = -1,
          Ncalibrate = 0,
          a = gains[:a], b = gains[:b],
          A = gains[:A], s = gains[:s], t = gains[:t],
          )

The [Complex Simultaneous Perturbation Stochastic Approximation (CSPSA)](https://www.nature.com/articles/s41598-019-52289-0)
optimizer is a method for optimizing real functions of a number of complex variables, and corresponds to a complex generalization
of the [SPSA method](https://www.jhuapl.edu/spsa/).

This function performs CSPSA optimization of the real-valued function `f`, starting from a vector
of complex variables `z₀` and iterating `Niter` times. Then, returns a complex matrix, `zacc`, with size `(length(z₀), Niters)`,
such that `zacc[i, j]` corresponds to the value of the `i`-th complex variable on the `j`-th iteration.

The input parameters `a`, `b`, `A`, `s`, and `t` can be provided as keyword arguments of the function.
If they are not provided explicitly, they are selected at runtime from the [`ComplexSPSA.gains`](@ref) dictionary.

Automatic calibration for the input parameter `a` can be accomplished taking `Ncalibrate` samples using the method defined on [Kandala _et. al._ (2017), Sec. 11](https://arxiv.org/pdf/1704.05018v2.pdf#section*.11).
By default, the calibration is disabled (`Ncalibrate = 0`).
"""
function CSPSA(f::Function, z₀::Vector, Niters = 200;
               sign = -1,
               Ncalibrate = 0,
               a = gains[:a], b = gains[:b],
               A = gains[:A], s = gains[:s], t = gains[:t],
               )

    z = z₀[:] .+ 0im
    Nvars = length(z)

    g = zeros(ComplexF64, size(z₀))

    # Set of possible perturbations
    samples = Complex{Float64}.((-1, 1, -im, im))

    # Calibrate gain a
    if Ncalibrate > 0
        ac = calibrate_gain_a(f, z, a, b, samples, Ncalibrate)
        ac > 1e-10 ? (a = ac) : nothing
    end

    # Accumulator
    zacc = Array{Complex{Float64}}(undef, Nvars, Niters)

    # Gradient Accumulator
    gacc = Array{Complex{Float64}}(undef, Nvars, Niters)

    for iter in 1:Niters
        ak = a / (iter + A)^s
        bk = b / iter^t

        # Vector of complex perturbations
        Δ = rand(samples, Nvars)

        df = f(@. z + bk*Δ) - f(@. z - bk*Δ)

        # Update variable in-place
        @. z += 0.5sign * ak * Δ * df / bk

        # Define Gradient
        @. g = 0.5sign * ak * Δ * df / bk

        # Copy arguments as complex to the accumulator
        zacc[:, iter] = z
        gacc[:, iter] = g
    end

    return zacc, gacc
end
