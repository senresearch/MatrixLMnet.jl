using MatrixLMnet
using Test

@testset "MatrixLMnet" begin
    # include("l1_helpers.jl")
    include("mlmnetTests.jl")
    include("mlmnetCvTests.jl")
end