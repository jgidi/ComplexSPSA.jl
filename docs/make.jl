# Allow julia to find the module by adding it to LOAD_PATH
push!(LOAD_PATH,"../src/")

using ComplexSPSA
using Documenter

Documenter.makedocs(
    source = "src",
    build = "build",
    clean = true,
    doctest = true,
    modules = Module[ComplexSPSA],
    repo = "github.com/jgidi/ComplexSPSA.jl.git",
    highlightsig = true,
    sitename = "ComplexSPSA.jl",
    expandfirst = [],
    pages = [
        "Home" => "index.md",
    ]
)

Documenter.deploydocs(
    ;
    repo="github.com/jgidi/ComplexSPSA.jl",
)
