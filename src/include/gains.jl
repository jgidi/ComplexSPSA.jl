"""
    gains_asymptotic = Dict(
        :a => 3.0,
        :b => 0.1,
        :A => 0.0,
        :s => 1.0,
        :t => 0.166,
    )

Dictionary containing the set of asymptotic gain parameters. By default, the standard
set of gain parameters are used.
"""
gains_asymptotic = Dict(
    :a => 3.0,
    :b => 0.1,
    :A => 0.0,
    :s => 1.0,
    :t => 0.166,
)

"""
    gains = Dict(
        :a => 3.0,
        :b => 0.1,
        :A => 0.0,
        :s => 0.602,
        :t => 0.101,
    )


Contains the gain parameters used for the optimizers defined within the `ComplexSPSA` module.
By default, the standard gains are used.
"""
gains = Dict(
    :a => 3.0,
    :b => 0.1,
    :A => 0.0,
    :s => 0.602,
    :t => 0.101,
)

function calibrate_gain_a(f, z, target_a, b, perturbations, Ncalibrate)
    avg = 0.0
    for _ in 1:Ncalibrate
        Δ = rand(perturbations, length(z))
        df = f(@. z + b*Δ) - f(@. z - b*Δ)

        avg += abs(df)
    end

    # Average gradient magnitude
    avg /= (2*b*Ncalibrate)

    # Calibrated value of a
    return target_a / avg
end
