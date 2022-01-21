using ProgressMeter, LinearAlgebra, Statistics
using ComplexSPSA

Nvars = 2
Nruns = 10^4
Niters = 10^2
Nmeasures = 10^4

# Generate a random normalized qubit
function rand_state()
    ket = randn(ComplexF64, 2)
    return ket / sqrt(sum(abs2, ket))
end

# Fidelity between 2 qubits 'guess' and 'state'
function fidelity(guess, state)
    prod2 = abs2(state' * guess)
    norms2 = sum(abs2, guess) * sum(abs2, state)
    return prod2 / norms2
end

# Metric as experiment simulated from a binomial distribution
# In practice, fidelity may return 1+1e-16, which errors on the binomial distribution.
# This is solved with the 'min' function to put an upper bound of 1.
metric(z1, z2) = -0.5*simulate_experiment(min(fidelity(z1, z2), 1), Nmeasures)

# Post-iteration normalization
normalize(z) = z / sqrt(sum(abs2, z))

# Accumulators and main loop
fz = Array{Float64}(undef, Niters, Nruns, 8)             # Eval. Infidelity
zacc = Array{ComplexF64}(undef, Nvars, Niters, Nruns, 8) # Variable
@showprogress for run in 1:Nruns
    # Generate random pair of states
    guess, target = rand_state(), rand_state()

    # Infidelity with respect to the current target state
    infidelity(guess) = 1 - fidelity(guess, target)              # Theoretical infidelity
    f(guess) = simulate_experiment(infidelity(guess), Nmeasures) # Experimental infidelity

    # Optimize. All optimizers with standard gains except for CSPSA with asymptotic gains
    zacc[:, :, run, 1] = SPSA_on_complex(f, guess, Niters, postprocess=normalize)
    zacc[:, :, run, 2] = CSPSA(f, guess, Niters, postprocess=normalize; ComplexSPSA.gains_asymptotic...)
    zacc[:, :, run, 3] = SPSA2_on_complex(f, guess, Niters, postprocess=normalize)
    zacc[:, :, run, 4] = CSPSA2(f, guess, Niters, postprocess=normalize)
    zacc[:, :, run, 5] = SPSA_QN_on_complex(f, metric, guess, Niters, postprocess=normalize)
    zacc[:, :, run, 6] = CSPSA_QN(f, metric, guess, Niters, postprocess=normalize)
    zacc[:, :, run, 7] = SPSA_QN_scalar_on_complex(f, metric, guess, Niters, postprocess=normalize)
    zacc[:, :, run, 8] = CSPSA_QN_scalar(f, metric, guess, Niters, postprocess=normalize)

    # Compute theoretical infidelity with respect to the target state
    fz[:, run, :] = mapslices(infidelity, zacc[:, :, run, :], dims=1)
end

# Calculate statistics
using Statistics: mean, median, var, quantile
get_quantile(A::AbstractArray, p; dims) = mapslices(x -> quantile(x, p), A, dims=dims)

# Mean and variance
fmean = mean(fz, dims=2)        # Mean
fvar  = var(fz, dims=2)         # Variance

# Median and inter-quartile range
fmedian = median(fz, dims=2)           # Meadian
fq25  = get_quantile(fz, 0.25, dims=2) # 1st quantile
fq75  = get_quantile(fz, 0.75, dims=2) # 3rd quantile

# Get rid of singleton dimensions
fmean = squeeze(fmean);
fmedian = squeeze(fmedian);
fvar = squeeze(fvar);
fq25 = squeeze(fq25);
fq75 = squeeze(fq75);


# ----- Make plots

using Plots

# Line colors [SPSA, CSPSA]
colors = ["blue", "red"]
# Line styles [Mean, Median]
styles = [:line, :dot]
# Line widths
lw = 2;

# First order
p1 = plot(ylabel = "Infidelity", leftmargin = 6Plots.mm)
# Mean
plot!(p1, 1:Niters, fmean[:, 1], ribbon = fvar[:, 1], line = (colors[1], lw), label = "SPSA")
plot!(p1, 1:Niters, fmean[:, 2], ribbon = fvar[:, 2], line = (colors[2], lw), label = "CSPSA");

# Second order
p2 = plot()
# Mean
plot!(p2, 1:Niters, fmean[:, 3], ribbon = fvar[:, 3], line = (colors[1], lw), label = "2-SPSA")
plot!(p2, 1:Niters, fmean[:, 4], ribbon = fvar[:, 4], line = (colors[2], lw), label = "2-CSPSA");

# Natural gradient
p3 = plot()
# Mean
plot!(p3, 1:Niters, fmean[:, 5], ribbon = fvar[:, 5], line = (colors[1], lw), label = "QN-SPSA")
plot!(p3, 1:Niters, fmean[:, 6], ribbon = fvar[:, 6], line = (colors[2], lw), label = "QN-CSPSA");

# Natural gradient - scalar
p4 = plot()
# Mean
plot!(p4, 1:Niters, fmean[:, 7], ribbon = fvar[:, 7], line = (colors[1], lw), label = "QN-SPSA scalar")
plot!(p4, 1:Niters, fmean[:, 8], ribbon = fvar[:, 8], line = (colors[2], lw), label = "QN-CSPSA scalar");

# Combine subplots
plot_mean = plot(p1, p2, p3, p4,
                 size = (1200, 260),
                 yscale = :log10,
                 grid = true,
                 framestyle = :box,
                 xlim = (1, Niters),
                 bottommargin = 7Plots.mm,
                 layout = (1, 4),
                 thickness_scaling = 1.1,
                 xlabel = "Iterations",
                 fontfamily = "serif-roman",
                 )

savefig(plot_mean,
        "mean_$(Nruns)runs-$(Niters)iters_$(Nmeasures)measures.pdf"
        )

# First order
p1 = plot(ylabel = "Infidelity", leftmargin = 6Plots.mm)
# Median
plot!(p1, 1:Niters, fmedian[:, 1], ribbon = (fq25[:, 1], fq75[:, 1]), line = (colors[1], lw), label = "SPSA")
plot!(p1, 1:Niters, fmedian[:, 2], ribbon = (fq25[:, 2], fq75[:, 2]), line = (colors[2], lw), label = "CSPSA");

# Second order
p2 = plot()
# Median
plot!(p2, 1:Niters, fmedian[:, 3], ribbon = (fq25[:, 3], fq75[:, 3]), line = (colors[1], lw), label = "2-SPSA")
plot!(p2, 1:Niters, fmedian[:, 4], ribbon = (fq25[:, 4], fq75[:, 4]), line = (colors[2], lw), label = "2-CSPSA");

# Natural gradient
p3 = plot()
# Median
plot!(p3, 1:Niters, fmedian[:, 5], ribbon = (fq25[:, 5], fq75[:, 5]), line = (colors[1], lw), label = "QN-SPSA")
plot!(p3, 1:Niters, fmedian[:, 6], ribbon = (fq25[:, 6], fq75[:, 6]), line = (colors[2], lw), label = "QN-CSPSA");

# Natural gradient - scalar
p4 = plot()
# Median
plot!(p4, 1:Niters, fmedian[:, 7], ribbon = (fq25[:, 7], fq75[:, 7]), line = (colors[1], lw), label = "QN-SPSA scalar")
plot!(p4, 1:Niters, fmedian[:, 8], ribbon = (fq25[:, 8], fq75[:, 8]), line = (colors[2], lw), label = "QN-CSPSA scalar");

# Combine subplots
plot_median = plot(p1, p2, p3, p4,
                   size = (1200, 260),
                   yscale = :log10,
                   grid = true,
                   framestyle = :box,
                   xlim = (1, Niters),
                   bottommargin = 7Plots.mm,
                   layout = (1, 4),
                   thickness_scaling = 1.1,
                   xlabel = "Iterations",
                   fontfamily = "serif-roman",
                   )

savefig(plot_median,
        "median_$(Nruns)runs-$(Niters)iters_$(Nmeasures)measures.pdf"
        )

plot_both = plot(plot_mean, plot_median,
                 xscale = :log10,
                 layout = (2, 1),
                 size = (1200, 520),
                 )

savefig(plot_both,
        "both_$(Nruns)runs-$(Niters)iters_$(Nmeasures)measures.pdf"
        )
