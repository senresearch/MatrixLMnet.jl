using MatrixLMnet
using Distributions, LinearAlgebra
using Helium
using Test

@testset "MatrixLMnet" begin
    # include("l1_helpers.jl")
    include("mlmnetTests.jl")
    include("mlmnetCvTests.jl")
    include("summaryCvTests.jl")
    include("mlmnetBicTests.jl")
end