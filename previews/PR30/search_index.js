var documenterSearchIndex = {"docs":
[{"location":"qiskit/optimizers/#Real-variable-optimizers","page":"Optimizers","title":"Real-variable optimizers","text":"","category":"section"},{"location":"qiskit/optimizers/","page":"Optimizers","title":"Optimizers","text":"Simple, thin wrappers around the SPSA-based optimizers implemented by Qiskit. This optimizers only work on real functions of real variables.","category":"page"},{"location":"qiskit/optimizers/","page":"Optimizers","title":"Optimizers","text":"Modules = [ComplexSPSA.Qiskit]\nPages = [\"include/qiskit/include/real_optimizers.jl\"]","category":"page"},{"location":"qiskit/optimizers/#ComplexSPSA.Qiskit.SPSA-Tuple{Any, Any, Any}","page":"Optimizers","title":"ComplexSPSA.Qiskit.SPSA","text":"SPSA(f, guess, Niters;\n     a = Qiskit.gains[:a], b = Qiskit.gains[:b],\n     A = Qiskit.gains[:A], s = Qiskit.gains[:s], t = Qiskit.gains[:t] )\n\nWrapper around the first order SPSA optimizer from Qiskit.\n\n\n\n\n\n","category":"method"},{"location":"qiskit/optimizers/#ComplexSPSA.Qiskit.SPSA2-Tuple{Any, Any, Any}","page":"Optimizers","title":"ComplexSPSA.Qiskit.SPSA2","text":"SPSA2(f, guess, Niters;\n      b = Qiskit.gains[:b], A = Qiskit.gains[:A],\n      s = Qiskit.gains[:s], t = Qiskit.gains[:t] )\n\nWrapper around the second order SPSA optimizer from Qiskit.\n\n\n\n\n\n","category":"method"},{"location":"qiskit/optimizers/#ComplexSPSA.Qiskit.SPSA_NG-NTuple{4, Any}","page":"Optimizers","title":"ComplexSPSA.Qiskit.SPSA_NG","text":"SPSA_NG(f, fidelity, guess, Niters;\n        b = Qiskit.gains[:b], A = Qiskit.gains[:A],\n        s = Qiskit.gains[:s], t = Qiskit.gains[:t] )\n\nWrapper around the QNSPSA optimizer from Qiskit.\n\n\n\n\n\n","category":"method"},{"location":"qiskit/optimizers/#Complex-variable-optimizers","page":"Optimizers","title":"Complex-variable optimizers","text":"","category":"section"},{"location":"qiskit/optimizers/","page":"Optimizers","title":"Optimizers","text":"Optimizers which separate the optimization problem of a real function of N complex variables as the equivalent problem of optimizing a real function of 2N real variables, and then solve it by means of the Real-variable optimizers.","category":"page"},{"location":"qiskit/optimizers/","page":"Optimizers","title":"Optimizers","text":"Modules = [ComplexSPSA.Qiskit]\nPages = [\"include/qiskit/include/complex_optimizers.jl\"]","category":"page"},{"location":"qiskit/optimizers/#ComplexSPSA.Qiskit.SPSA2_on_complex-Tuple{Any, Vector{ComplexF64}, Any}","page":"Optimizers","title":"ComplexSPSA.Qiskit.SPSA2_on_complex","text":"SPSA2_on_complex(f, guess::Vector{Complex{Float64}}, Niters;\n                 b = Qiskit.gains[:b], A = Qiskit.gains[:A],\n                 s = Qiskit.gains[:s], t = Qiskit.gains[:t] )\n\nWrapper around SPSA2 which internally converts the complex-valued initial point, guess, and objective function, f, to be optimized as a real-variable problem with twice the parameters.\n\n\n\n\n\n","category":"method"},{"location":"qiskit/optimizers/#ComplexSPSA.Qiskit.SPSA_NG_on_complex-Tuple{Any, Any, Vector{ComplexF64}, Any}","page":"Optimizers","title":"ComplexSPSA.Qiskit.SPSA_NG_on_complex","text":"SPSA_NG_on_complex(f, fidelity, guess::Vector{Complex{Float64}}, Niters;\n                   b = Qiskit.gains[:b], A = Qiskit.gains[:A],\n                   s = Qiskit.gains[:s], t = Qiskit.gains[:t] )\n\nWrapper around SPSA_NG which internally converts the complex-valued initial point, guess, the fidelity, and the objective function, f, to be optimized as a real-variable problem with twice the parameters.\n\n\n\n\n\n","category":"method"},{"location":"qiskit/optimizers/#ComplexSPSA.Qiskit.SPSA_on_complex-Tuple{Any, Vector{ComplexF64}, Any}","page":"Optimizers","title":"ComplexSPSA.Qiskit.SPSA_on_complex","text":"SPSA_on_complex(f, guess::Vector{Complex{Float64}}, Niters;\n                a = Qiskit.gains[:a], b = Qiskit.gains[:b],\n                A = Qiskit.gains[:A], s = Qiskit.gains[:s], t = Qiskit.gains[:t] )\n\nWrapper around SPSA which internally converts the complex-valued initial point, guess, and objective function, f, to be optimized as a real-variable problem with twice the parameters.\n\n\n\n\n\n","category":"method"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#Example-1-CSPSA-optimization-of-a-simple-function","page":"Examples","title":"Example 1 - CSPSA optimization of a simple function","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Download file","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Markdown\nMarkdown.parse(\"\"\"\n```julia\n$(read(\"assets/examples/ex01.jl\", String))\n```\n\"\"\")","category":"page"},{"location":"examples/#Example-2-CSPSA-vs-SPSA-on-qubit-tomography-(single-run)","page":"Examples","title":"Example 2 - CSPSA vs SPSA on qubit tomography (single run)","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Download file","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Markdown\nMarkdown.parse(\"\"\"\n```julia\n$(read(\"assets/examples/ex02.jl\", String))\n```\n\"\"\")","category":"page"},{"location":"examples/#Example-3-CSPSA-vs-SPSA-on-qubit-tomography-(multiple-runs)","page":"Examples","title":"Example 3 - CSPSA vs SPSA on qubit tomography (multiple runs)","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Download file","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Markdown\nMarkdown.parse(\"\"\"\n```julia\n$(read(\"assets/examples/ex03.jl\", String))\n```\n\"\"\")","category":"page"},{"location":"examples/#Example-4","page":"Examples","title":"Example 4","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Download file","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Markdown\nMarkdown.parse(\"\"\"\n```julia\n$(read(\"assets/examples/qubit_tomography.jl\", String))\n```\n\"\"\")","category":"page"},{"location":"examples/#Example-5","page":"Examples","title":"Example 5","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Download file","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Markdown\nMarkdown.parse(\"\"\"\n```julia\n$(read(\"assets/examples/ferrie.jl\", String))\n```\n\"\"\")","category":"page"},{"location":"qiskit/intro/#Qiskit-Wrapper","page":"Introduction","title":"Qiskit Wrapper","text":"","category":"section"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"For purposes of ease of comparison, the ComplexSPSA module includes a Qiskit submodule which wraps some SPSA-based optimizers from the python library Qiskit, and exposes them with an interface common to the rest of the module. The wrappers use the default options as defined by Qiskit for each optimizer.","category":"page"},{"location":"qiskit/intro/#Index-of-optimizers-implemented","page":"Introduction","title":"Index of optimizers implemented","text":"","category":"section"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"Pages = [\"optimizers.md\"]","category":"page"},{"location":"qiskit/intro/#Ensure-julia-can-find-Qiskit","page":"Introduction","title":"Ensure julia can find Qiskit","text":"","category":"section"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"To make sure that julia can find and use the Qiskit library, you can execute the function","category":"page"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"ComplexSPSA.Qiskit.pip_install_dependencies","category":"page"},{"location":"qiskit/intro/#ComplexSPSA.Qiskit.pip_install_dependencies","page":"Introduction","title":"ComplexSPSA.Qiskit.pip_install_dependencies","text":"pip_install_dependencies()\n\nEnsure the Qiskit library is installed on the python distribution reached by julia.\n\nNotes\n\nAssumes the pip tool is present on the system.\n\n\n\n\n\n","category":"function"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"which is provided by the Qiskit submodule, and may thus be executed as","category":"page"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"using ComplexSPSA\nComplexSPSA.Qiskit.pip_install_dependencies()","category":"page"},{"location":"qiskit/intro/#Using-the-optimizers","page":"Introduction","title":"Using the optimizers","text":"","category":"section"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"To expose the methods defined on the Qiskit submodule, you can run","category":"page"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"using ComplexSPSA.Qiskit","category":"page"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"and then, use the optimizers as","category":"page"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"x = Qiskit.SPSA(f, guess, Niters)","category":"page"},{"location":"qiskit/intro/","page":"Introduction","title":"Introduction","text":"assuming you have previously defined a valid function, f, initial array of parameters, guess, and number of iterations, Niters.","category":"page"},{"location":"#ComplexSPSA","page":"Introduction","title":"ComplexSPSA","text":"","category":"section"},{"location":"#Description","page":"Introduction","title":"Description","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"A package on Julia for the implementation of stochastic optimizers of real functions on many complex variables, currently under development by Jorge Gidi for the Quantum Information Group from the Universidad de Concepción, Chile.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"This package is still on an early stage of development.","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"While this package is available on github, it has not (yet) been registered into the Julia ecosystem. Anyways, Julia is capable of cloning, installing and keeping track of this package automatically. To install it, just open a Julia session in a terminal or a Jupyter Notebook, and run","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using Pkg\npkg\"add https://github.com/jgidi/ComplexSPSA.jl\"","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Assuming the process has ended successfully, ComplexSPSA.jl will be installed and you can now access its functionalities by means of","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using ComplexSPSA","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"and may be updated, as any other package, with","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using Pkg\npkg\"update ComplexSPSA\"","category":"page"},{"location":"#About-the-algorithms","page":"Introduction","title":"About the algorithms","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The algorithms hereafter presented are based upon the Simultaneous Perturbation Stochastic Optimization (SPSA) method for real functions of real variables.","category":"page"},{"location":"tools/#Tools","page":"Other tools","title":"Tools","text":"","category":"section"},{"location":"tools/","page":"Other tools","title":"Other tools","text":"Some convenience functions are exposed to the user to solve simple, usual tasks that may appear commonly on the use-cases of this module. Those tools are listed below. ","category":"page"},{"location":"tools/#Index","page":"Other tools","title":"Index","text":"","category":"section"},{"location":"tools/","page":"Other tools","title":"Other tools","text":"Pages = [\"tools.md\"]","category":"page"},{"location":"tools/#Documentation","page":"Other tools","title":"Documentation","text":"","category":"section"},{"location":"tools/","page":"Other tools","title":"Other tools","text":"Modules = [ComplexSPSA]\nPages = [\"include/tools.jl\"]","category":"page"},{"location":"tools/#ComplexSPSA.apply_along_dim-Tuple{Function, AbstractArray}","page":"Other tools","title":"ComplexSPSA.apply_along_dim","text":"apply_along_dim(f::Function, A::AbstractArray; dim::Integer = 1)\n\nApply a function, f, over slices along a dimension, dim, of a multidimensional array, A.\n\nExample\n\njulia> a = [ 1 2; 3 4 ]\n2×2 Matrix{Int64}:\n 1  2\n 3  4\n\njulia> apply_along_dim(sum, a, dim = 1)\n1×2 Matrix{Int64}:\n 4  6\n\n\n\n\n\n","category":"method"},{"location":"tools/#ComplexSPSA.simulate_experiment","page":"Other tools","title":"ComplexSPSA.simulate_experiment","text":"simulate_experiment(refvalue, Nmeasures = Inf)\n\nSimulates the experimental measurement with Nmeasures number of tries of an observable whose theoretical value is refvalue.\n\nThe experiental result is simulated by sampling a binomial distribution with Nmeasures tries and success rate refvalue, and normalizing the result against the number of tries.\n\nNotes\n\nIf the number of measurements, Nmeasures, is infinite, the reference value, refvalue, is returned exactly.\nIt is assumed that 0.0 <= refvalue <= 1.0. If not, refvalue will be taken as its closest boundary.\n\n\n\n\n\n","category":"function"},{"location":"tools/#ComplexSPSA.squeeze-Tuple{AbstractArray}","page":"Other tools","title":"ComplexSPSA.squeeze","text":"squeeze(A :: AbstractArray)\n\nRemove the singleton (length-1) dimensions of a multidimensional array.\n\nExample\n\njulia> a = rand(10, 3, 1, 4);\n\njulia> size(a)\n(10, 3, 1, 4)\n\njulia> size( squeeze(a) )\n(10, 3, 4)\n\n\n\n\n\n","category":"method"},{"location":"optimizers/#Optimizers","page":"Optimizers","title":"Optimizers","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"This package provides a set optimizers for real-valued functions of a number of complex variables, fmathbbC^NtomathbbR, which we subdivide into three categories","category":"page"},{"location":"optimizers/#First-order-optimizers","page":"Optimizers","title":"First-order optimizers","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"To reach a local minimum of f(bm z) with bm z in mathbbC^N, this methods start from some seed value of bm z = bm z^0 and iterate over k performing a first-order update,","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"bm z^k+1 = bm z^k - a^k bm g^k(bm z^k b^k)","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"where bm g^k(bm z^k b^k) is a randomly directed finite-difference estimate of the gradient of f, partial f  partialbmz, and","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"beginaligned\na^k = fraca(k + A + 1)^s \nb^k = fracb(k + 1)^t\nendaligned","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"are convergence factors determined by the input gain parameters a, b, A, s, and t.","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"By default, standard gains are defined and used package-wide from the dictionary ComplexSPSA.gains,","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"ComplexSPSA.gains","category":"page"},{"location":"optimizers/#ComplexSPSA.gains","page":"Optimizers","title":"ComplexSPSA.gains","text":"gains = Dict(\n    :a => 3.0,\n    :b => 0.1,\n    :A => 0.0,\n    :s => 0.602,\n    :t => 0.101,\n)\n\nContains the gain parameters used for the optimizers defined within the ComplexSPSA module. By default, the standard gains are used.\n\n\n\n\n\n","category":"constant"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"where they can be changed on run-time, and the optimizers will use their value as defined at the moment when they are called.","category":"page"},{"location":"optimizers/#SPSA-on-complex","page":"Optimizers","title":"SPSA on complex","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"In particular, two first-order methods are implemented. The first, called SPSA_on_complex, is based on the SPSA optimization method for real variables of real functions, and works by treating each complex variable as a pair of real variables. This method is commonly used for the minimization of real functions of complex variables, but may result cumbersome on domains where the complex algebra is natively involved.","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"SPSA_on_complex","category":"page"},{"location":"optimizers/#ComplexSPSA.SPSA_on_complex","page":"Optimizers","title":"ComplexSPSA.SPSA_on_complex","text":"SPSA_on_complex(f::Function, z₀::Vector, Niters = 200;\n                sign = -1,\n                Ncalibrate = 0,\n                initial_iteration = 1,\n                constant_learning_rate = false,\n                a = gains[:a], b = gains[:b],\n                A = gains[:A], s = gains[:s], t = gains[:t],\n                )\n\nThe Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer is a stochastic method for optimizing real functions of a number of real variables.\n\nThis function performs SPSA optimization of the real-valued function f of complex variables by treating each complex varible as a pair of real variables, starting from the complex vector z₀ and iterating Niter times. Then, returns a complex matrix, zacc, with size (length(z₀), Niters), such that zacc[i, j] corresponds to the value of the i-th complex variable on the j-th iteration.\n\nThe input parameters a, b, A, s, and t can be provided as keyword arguments of the function. If they are not provided explicitly, they are selected at runtime from the ComplexSPSA.gains dictionary.\n\nAutomatic calibration for the input parameter a can be accomplished taking Ncalibrate samples using the method defined on Kandala et. al. (2017), Sec. 11. By default, the calibration is disabled (Ncalibrate = 0).\n\n\n\n\n\n","category":"function"},{"location":"optimizers/#CSPSA","page":"Optimizers","title":"CSPSA","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"A second method called CSPSA is also provided, which has the advantage of having being formulated in terms of complex variables, thus resulting more natural in natively-complex domains.","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"CSPSA","category":"page"},{"location":"optimizers/#ComplexSPSA.CSPSA","page":"Optimizers","title":"ComplexSPSA.CSPSA","text":"CSPSA(f::Function, z₀::Vector, Niters = 200;\n      sign = -1,\n      Ncalibrate = 0,\n      initial_iteration = 1,\n      constant_learning_rate = false,\n      a = gains[:a], b = gains[:b],\n      A = gains[:A], s = gains[:s], t = gains[:t],\n      )\n\nThe Complex Simultaneous Perturbation Stochastic Approximation (CSPSA) optimizer is a method for optimizing real functions of a number of complex variables, and corresponds to a complex generalization of the SPSA method.\n\nThis function performs CSPSA optimization of the real-valued function f, starting from a vector of complex variables z₀ and iterating Niter times. Then, returns a complex matrix, zacc, with size (length(z₀), Niters), such that zacc[i, j] corresponds to the value of the i-th complex variable on the j-th iteration.\n\nThe input parameters a, b, A, s, and t can be provided as keyword arguments of the function. If they are not provided explicitly, they are selected at runtime from the ComplexSPSA.gains dictionary.\n\nAutomatic calibration for the input parameter a can be accomplished taking Ncalibrate samples using the method defined on Kandala et. al. (2017), Sec. 11. By default, the calibration is disabled (Ncalibrate = 0).\n\n\n\n\n\n","category":"function"},{"location":"optimizers/#Second-order-optimizers","page":"Optimizers","title":"Second-order optimizers","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"SPSA2_on_complex","category":"page"},{"location":"optimizers/#ComplexSPSA.SPSA2_on_complex","page":"Optimizers","title":"ComplexSPSA.SPSA2_on_complex","text":"SPSA2_on_complex(f::Function, z₀::Vector, Niters = 200;\n                 sign = -1,\n                 hessian_delay = 0,\n                 initial_iteration = 1,\n                 constant_learning_rate = false,\n                 a = gains[:a], b = gains[:b],\n                 A = gains[:A], s = gains[:s], t = gains[:t],\n                 )\n\nThe second-order SPSA, commonly referred to as 2-SPSA method is a second-order stochastic optimization method based on SPSA, which additional to a gradient estimate performs a Hessian correction on the update rule to optimize real-valued functions of a number of real variables.\n\nThis function performs second-order SPSA optimization of the real-valued function f of complex variables by treating each complex variable as a pair of real variables, starting from the complex vector z₀ and iterating Niter times. Then, returns a complex matrix, zacc, with size (length(z₀), Niters), such that zacc[i, j] corresponds to the value of the i-th complex variable on the j-th iteration.\n\nThe input parameters a, b, A, s, and t can be provided as keyword arguments of the function. If not provided explicitly, they are selected at runtime from the ComplexSPSA.gains dictionary.\n\nSince second-order effects usually show improvements once the seed value is closer to a local minimum, it is possible to accept a number hessian_delay of first-order iterations before including the application of the Hessian information.\n\nNotes\n\nThe value of a is only required to perform a possible number of initial first-order iterations (via hessian_delay), since the second-order iterations yield an optimum value for a = 1.\n\n\n\n\n\n","category":"function"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"CSPSA2","category":"page"},{"location":"optimizers/#ComplexSPSA.CSPSA2","page":"Optimizers","title":"ComplexSPSA.CSPSA2","text":"CSPSA2(f::Function, z₀::Vector, Niters = 200;\n       sign = -1,\n       hessian_delay = 0,\n       initial_iteration = 1,\n       constant_learning_rate = false,\n       a = gains[:a], b = gains[:b],\n       A = gains[:A], s = gains[:s], t = gains[:t],\n       )\n\nThe second-order CSPSA method, CSPSA2, is a second-order stochastic optimization method based on CSPSA, which additional to a gradient estimate performs a Hessian correction on the update rule to optimize real-valued functions of a number of complex variables.\n\nThis function performs second-order CSPSA optimization of the real-valued function f of complex variables, starting from the complex vector z₀ and iterating Niter times. Then, returns a complex matrix, zacc, with size (length(z₀), Niters), such that zacc[i, j] corresponds to the value of the i-th complex variable on the j-th iteration.\n\nThe input parameters a, b, A, s, and t can be provided as keyword arguments of the function. If they are not provided explicitly, they are selected at runtime from the ComplexSPSA.gains dictionary.\n\nSince second-order effects usually show improvements once the seed value is closer to a local minimum, it is possible to accept a number hessian_delay of first-order iterations before including the application of the Hessian information.\n\nNotes\n\nThe value of a is only required to perform a possible number of initial first-order iterations (via hessian_delay), since the second-order iterations yield an optimum value for a = 1.\n\n\n\n\n\n","category":"function"},{"location":"optimizers/#Natural-gradient","page":"Optimizers","title":"Natural gradient","text":"","category":"section"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"SPSA_QN_on_complex","category":"page"},{"location":"optimizers/#ComplexSPSA.SPSA_QN_on_complex","page":"Optimizers","title":"ComplexSPSA.SPSA_QN_on_complex","text":"SPSA_QN_on_complex(f::Function, z₀::Vector, Niters = 200;\n                   sign = -1,\n                   hessian_delay = 0,\n                   initial_iteration = 1,\n                   constant_learning_rate = false,\n                   a = gains[:a], b = gains[:b],\n                   A = gains[:A], s = gains[:s], t = gains[:t],\n                   )\n\nThe Quantum Natural SPSA, presented by Gacon et. al. (2021), is a second-order stochastic optimization method based on 2-SPSA, where the second-order correction comes from the Hessian of the Fubini-Study metric of the problem instead of the Hessian of the function under optimization.\n\nNote that the metric must be a function taking two input vectors, and returning minus a half of the fidelity between the states generated each from one input,\n\n$ \\text{metric}(\\vec z₁, \\vec z₂) = -\\frac{1}{2} |\\langle ψ(\\vec z₁) | ψ(\\vec z₂) \\rangle|^2, $\n\nwhere $ ψ(\\vec z) $ is the quantum state parameterized with the variables $ \\vec z $.\n\nThis function performs Quantum Natural SPSA optimization of the real-valued function f of complex variables by treating each complex variable as a pair of real variables, starting from the complex vector z₀ and iterating Niter times. Then, returns a complex matrix, zacc, with size (length(z₀), Niters), such that zacc[i, j] corresponds to the value of the i-th complex variable on the j-th iteration.\n\nThe input parameters a, b, A, s, and t can be provided as keyword arguments of the function. If not provided explicitly, they are selected at runtime from the ComplexSPSA.gains dictionary.\n\nSince second-order effects usually show improvements once the seed value is closer to a local minimum, it is possible to accept a number hessian_delay of first-order iterations before including the application of the Hessian information.\n\nNotes\n\nThe value of a is only required to perform a possible number of initial first-order iterations (via hessian_delay), since the second-order iterations yield an optimum value for a = 1.\n\n\n\n\n\n","category":"function"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"CSPSA_QN","category":"page"},{"location":"optimizers/#ComplexSPSA.CSPSA_QN","page":"Optimizers","title":"ComplexSPSA.CSPSA_QN","text":"CSPSA_QN(f::Function, metric::Function, z₀::Vector, Niters = 200;\n         sign = -1,\n         hessian_delay = 0,\n         initial_iteration = 1,\n         constant_learning_rate = false,\n         a = gains[:a], b = gains[:b],\n         A = gains[:A], s = gains[:s], t = gains[:t],\n         )\n\nThe Quantum Natural CSPSA (QN-CSPSA), is a second-order stochastic optimization method which, analogous to the Quantum Natural SPSA by Gacon et. al. (2021), takes into account a stochastic approximation of the Fubiny-Study metric instead of the usual Hessian correction from CSPSA2. However, the main difference between QN-CSPSA and QN-SPSA is that the former is natively formulated in terms of complex variables, while the latter requires real variables. Note that the metric must be a function taking two input vectors, and returning minus a half of the fidelity between the states generated each from one input,\n\n$ \\text{metric}(\\vec z₁, \\vec z₂) = -\\frac{1}{2} |\\langle ψ(\\vec z₁) | ψ(\\vec z₂) \\rangle|^2, $\n\nwhere $ ψ(\\vec z) $ is the quantum state parameterized with the variables $ \\vec z $.\n\nThis function performs Quantum Natural CSPSA optimization of the real-valued function f of complex variables, starting from the complex vector z₀ and iterating Niter times. Then, returns a complex matrix, zacc, with size (length(z₀), Niters), such that zacc[i, j] corresponds to the value of the i-th complex variable on the j-th iteration.\n\nThe input parameters a, b, A, s, and t can be provided as keyword arguments of the function. If not provided explicitly, they are selected at runtime from the ComplexSPSA.gains dictionary.\n\nSince second-order effects usually show improvements once the seed value is closer to a local minimum, it is possible to accept a number hessian_delay of first-order iterations before including the application of the Hessian information.\n\nNotes\n\nThe value of a is only required to perform a possible number of initial first-order iterations (via hessian_delay), since the second-order iterations yield an optimum value for a = 1.\n\n\n\n\n\n","category":"function"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"SPSA_QN_scalar_on_complex","category":"page"},{"location":"optimizers/","page":"Optimizers","title":"Optimizers","text":"CSPSA_QN_scalar","category":"page"}]
}
