###########
# Library #
###########
using MatrixLMnet
using Test

TOL = 1.0e-12


#################################
# TEST Mlmnet Helpers functions #
#################################

@testset "prox scalar without norms" begin
    b = 0.8
    gradient = -0.3
    stepsize = 0.5
    lambda = 0.4
    b2update = b - stepsize * gradient
    b2sign = sign(b2update)
    expected = max(0.0, b2sign * b - stepsize * (b2sign * gradient + lambda)) * b2sign
    actual = MatrixLMnet.prox(b, gradient, b2sign, lambda, nothing, stepsize)
    @test isapprox(actual, expected; atol = TOL)

    # With unit step size the stepsize-1 specialization should match.
    expected_unit = max(0.0, b2sign * b - (b2sign * gradient + lambda)) * b2sign
    actual_unit = MatrixLMnet.prox(b, gradient, b2sign, lambda, nothing)
    @test isapprox(actual_unit, expected_unit; atol = TOL)
end

@testset "prox scalar with norms" begin
    b = -0.5
    gradient = 0.75
    norm = 1.8
    stepsize = 0.3
    lambda = 0.6
    b2update = b - stepsize * gradient / norm
    b2sign = sign(b2update)
    expected = max(0.0, b2sign * b - stepsize * (b2sign * gradient + lambda) / norm) * b2sign
    actual = MatrixLMnet.prox(b, gradient, b2sign, lambda, norm, stepsize)
    @test isapprox(actual, expected; atol = TOL)

    expected_unit = max(0.0, b2sign * b - (b2sign * gradient + lambda) / norm) * b2sign
    actual_unit = MatrixLMnet.prox(b, gradient, b2sign, lambda, norm)
    @test isapprox(actual_unit, expected_unit; atol = TOL)
end

@testset "prox_mat without norms" begin
    B = [0.5 -0.2; -0.3 0.1]
    gradients = [0.7 -0.4; 0.2 0.9]
    stepsize = 0.25
    lambda = 0.3
    b2update = B .- stepsize .* gradients
    b2sign = sign.(b2update)
    expected = max.(0.0, abs.(B) .- stepsize * lambda) .* b2sign
    actual = MatrixLMnet.prox_mat(B, b2sign, lambda, nothing, stepsize)
    @test isapprox(actual, expected; atol = TOL)

    expected_unit = max.(0.0, abs.(B) .- lambda) .* b2sign
    actual_unit = MatrixLMnet.prox_mat(B, b2sign, lambda, nothing)
    @test isapprox(actual_unit, expected_unit; atol = TOL)
end

@testset "prox_mat with norms" begin
    B = [-0.6 0.4; 0.2 -0.1]
    gradients = [0.2 1.1; -0.7 0.5]
    stepsize = 0.2
    lambda = 0.45
    norms = [1.2 0.8; 1.5 2.0]
    b2update = B .- stepsize .* gradients ./ norms
    b2sign = sign.(b2update)
    expected = max.(0.0, abs.(B) .- stepsize * lambda ./ norms) .* b2sign
    actual = MatrixLMnet.prox_mat(B, b2sign, lambda, norms, stepsize)
    @test isapprox(actual, expected; atol = TOL)

    expected_unit = max.(0.0, abs.(B) .- lambda ./ norms) .* b2sign
    actual_unit = MatrixLMnet.prox_mat(B, b2sign, lambda, norms)
    @test isapprox(actual_unit, expected_unit; atol = TOL)
end
