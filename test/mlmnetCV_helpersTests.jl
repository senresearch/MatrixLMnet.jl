
###########
# Library #
###########
# using MatrixLMnet
# using Helium
# using Test

######################
# TEST minimize_rows #
######################

@testset "Testing minimize_rows function" begin
    # Test with a single element
    @test MatrixLMnet.minimize_rows([CartesianIndex(1, 1)]) == [CartesianIndex(1, 1)]

    # Test with multiple elements in the same column
    @test MatrixLMnet.minimize_rows([CartesianIndex(3, 1), CartesianIndex(2, 1)]) == [CartesianIndex(2, 1)]

    # Test with multiple elements in different columns
    @test MatrixLMnet.minimize_rows([CartesianIndex(1, 1), CartesianIndex(2, 2), CartesianIndex(3, 2)]) == [CartesianIndex(2, 2), CartesianIndex(1, 1)]

    # Test with non-sequential columns
    @test MatrixLMnet.minimize_rows([CartesianIndex(3, 5), CartesianIndex(2, 1)]) == [CartesianIndex(3, 5), CartesianIndex(2, 1)]

    # Test with duplicate indices
    @test MatrixLMnet.minimize_rows([CartesianIndex(2, 2), CartesianIndex(2, 2)]) == [CartesianIndex(2, 2)]

    # Test with a more complex scenario
    @test MatrixLMnet.minimize_rows([CartesianIndex(4, 3), CartesianIndex(1, 3), CartesianIndex(2, 4), CartesianIndex(3, 4)]) == [CartesianIndex(2, 4), CartesianIndex(1, 3)]
end
