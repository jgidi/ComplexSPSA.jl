function squeeze(A :: AbstractArray)
    keepdims = Tuple(i for i in size(A) if i != 1)
    return reshape(A, keepdims)
end

function apply_along_dim(f::Function, A::AbstractArray; dim::Integer = 1)

    axs = axes(A)
    d = length(axes(A))
    @assert dim<=d "The array provided has only $d dims, but you asked for dim = $dim"

    prev_axes = axs[ 1:(dim-1) ]
    post_axes = axs[ (dim+1):end ]

    prev_ind = CartesianIndices(prev_axes)
    post_ind = CartesianIndices(post_axes)

    R = [f(A[I, :, J]) for J in post_ind for I in prev_ind]

    # Return array with modified length along 'dim'
    return reshape(R, size(prev_ind)..., :, size(post_ind)...)
end

function simulate_experiment(refvalue, Nmeasures)
    if Nmeasures < 0
        sample = refvalue
    else
        distrib = Binomial(Nmeasures, refvalue)
        sample = rand(distrib)/Nmeasures
    end

    return sample
end

using Plots, ProgressMeter, Statistics
function make_comparison_plot(f, metric, guess, optimizers, Nit, Nmeasures = -1;
                              kw... )

    Nvars, Nruns = size(guess)

    fexp(z) = simulate_experiment(f(z), Nmeasures)

    zacc = Array{Complex{Float64}}(undef, Nvars, Nit, Nruns, length(optimizers))
    @showprogress for run = 1:Nruns
        # Start from some random qubit
        initial = guess[:, run]

        # Save the return values from various optimizers
        for l in 1:length(optimizers)
            zacc[:, :, run, l] = optimizers[l](fexp, initial, Nit, metric = metric; kw...)
        end
    end

    k = 1:Nit

    fz = apply_along_dim(f, zacc, dim = 1)
    fmean = mean(fz, dims = 3)
    fvar = var(fz, dims = 3)
    fmean = squeeze(fmean)
    fvar = squeeze(fvar)

    p = plot(
        title = "Avg over $Nruns runs",
        xlabel = "Iteration",
        ylabel = string(f),
        # legend = :bottomleft,
        frame = :box,
    )


    fnames = string.(optimizers)
    for i in eachindex(fnames)
        plot!(
            p,
            k,
            fmean[:, i],
            ribbon = fvar[:, i],
            yscale = :log10,
            label = fnames[i],
            xlims = (k[1], k[end]),
            #   ylims = (minimum(fmean), 1),
        )
    end

    return p
end
