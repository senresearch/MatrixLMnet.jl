push!(LOAD_PATH,"../src/")

using Documenter
using MatrixLMnet

makedocs(
        modules  = [MatrixLMnet], 
        sitename = "MatrixLMnet",
               
         pages=[
            "Home" => "index.md",
            "Getting Started" => "MLMnet_Simulation.md",
            #"More examples" => "moreExamples.md",
            "Types and Functions" => "functions.md"
               ],
)
deploydocs(;
    repo= "https://github.com/senresearch/MatrixLMnet.jl",
    devbranch= "main",
    devurl = "stable"
)