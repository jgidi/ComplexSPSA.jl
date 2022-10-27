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


decaying_learning_rate(a, A, s, k) =  a / (k + A)^s
decaying_pert_magnitude(b, t, k) = b / k^t

function calibrate_gain_a(f, z, a_target, bk, Ncalibrate;
                          modelspace=false)

    T = eltype(z)
    avg = zero(T)
    samples = perturbation_samples(T)
    for _ in 1:Ncalibrate
        Δ = rand(samples, length(z))
        df = f(@. z + bk*Δ) - f(@. z - bk*Δ)

        avg += abs(df)
    end

    # Average gradient magnitude
    avg /= 2bk * Ncalibrate

    # Calibrated value of a
    if modelspace
        a_new = a_target / avg^2
    else
        a_new = a_target / avg
    end

    # Check calibration
    if a_new < 1e-10
        println("Calibration for \'a\' failed. Using target value, ", a_target)
        a_new = a_target
    end

    return a_new
end
