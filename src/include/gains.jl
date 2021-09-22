"""
    gains::Dict

Contains the gain parameters used for the optimizers defined within the `ComplexSPSA` module.
"""
gains = Dict(
    :a => 3,
    :b => 0.1,
    :A => 1.0,
    :s => 1.0,
    :t => 1/6,
)
# gains = Dict(
#     :a => 2pi/10,
#     :b => 0.2,
#     :A => 0.0,
#     :s => 0.602,
#     :t => 0.101,
# )

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
