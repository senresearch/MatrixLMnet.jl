using Documenter
using MatrixLMnet

const src = "https://github.com/senresearch/MatrixLMnet.jl"

makedocs(
         sitename = "MatrixLMnet",
         authors = "Jane W. Liang, Saunak Sen",
         format = Documenter.HTML(),
         modules  = [MatrixLMnet],
         pages=[
            "Home" => "index.md",
            "Getting Started" => "MLMnet_Simulation.md",
            #"More examples" => "moreExamples.md",
            "Types and Functions" => "functions.md"
               ])
deploydocs(;
    repo= src,
    devbranch= "main",
    devurl = "stable"
)