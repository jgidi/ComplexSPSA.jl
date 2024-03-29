# Estimate standard deviation
estimate_std(f, z, N) = std(f(z) for _ in 1:N)


function real_problem_as_complex(fun_re, guess_re::AbstractVector{T}) where T <: Real

    fun_cx(x) = fun_re(reinterpret(Complex{T}, x))
    guess_cx = copy(reinterpret(T, guess_re))

    return fun_cx, guess_cx
end


function complex_problem_as_real(fun_cx, guess_cx::AbstractVector{Complex{T}}) where T

    fun_re(x) = fun_cx(reinterpret(Complex{T}, x))
    guess_re = copy(reinterpret(T, guess_cx))

    return fun_re, guess_re
end
