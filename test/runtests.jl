using MatrixLMnet
using StableRNGs
using Distributions, LinearAlgebra
using Helium
using Test

@testset "MatrixLMnet" begin
    # include("l1_helpers.jl")
    include("mlmnetTests.jl")
    include("mlmnetCvTests.jl")
    include("summaryCvTests.jl")
    include("mlmnetBicTests.jl")
    include("cdTests.jl")
    include("ista_fista_updatesTests.jl")
    include("utilitiesTests.jl")
    include("mlmnetCV_helpersTests.jl")
end