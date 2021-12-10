using MatrixLMnet2
using Test

@testset "MatrixLMnet2" begin
    include("lasso_helpers.jl")
    include("mlmnetTests.jl")
    include("mlmnetCvTests.jl")
end