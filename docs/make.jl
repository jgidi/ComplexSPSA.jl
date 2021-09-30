# Allow julia to find the module by adding it to LOAD_PATH
push!(LOAD_PATH,"../src/")

using ComplexSPSA
using Documenter

Documenter.makedocs(
    source = "src",
    build = "build",
    clean = true,
    doctest = true,
    modules = Module[ComplexSPSA, ComplexSPSA.Qiskit],
    highlightsig = true,
    sitename = "ComplexSPSA.jl",
    expandfirst = [],
    pages = [
        "Introduction" => "index.md",
        "Optimizers" => "optimizers.md",
        "Examples" => "examples.md",
        "Qiskit Wrapper" => [
            "Introduction" => "qiskit/intro.md",
            "Optimizers"   => "qiskit/optimizers.md",
        ],
        "Other tools" => "tools.md",
    ],
)

Documenter.deploydocs(
    ;
    repo = "github.com/jgidi/ComplexSPSA.jl",
    branch = "doc-pages",
    push_preview = true,
)
