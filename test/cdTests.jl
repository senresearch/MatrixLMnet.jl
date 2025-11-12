###########
# Library #
###########

using LinearAlgebra
using MatrixLMnet
using Random
using Test


#############
# Functions #
#############

# Small synthetic problems keep the updates 
# deterministic and easy to test.
function simple_problem()
    X = [1.0 0.5; -1.0 2.0; 0.0 1.0]
    Z = [1.0 0.2; -0.5 1.5]
    resid = [0.5 -0.2; 0.1 0.3; -0.4 0.2]
    B = [0.6 -0.4; 0.3 0.1]
    return X, Z, resid, B
end

function active_problem()
    X = [1.0 0.0; 0.0 1.0]
    Z = [1.0 0.0; 0.0 1.0]
    resid = [0.3 -0.1; 0.2 0.5]
    B = [0.2 0.0; 0.0 0.4]
    reg = trues(size(B))
    reg[1, 1] = false
    nonreg_idx = ([1], [1])
    active_idx = ([2], [2])
    return X, Z, resid, B, reg, nonreg_idx, active_idx
end

# Identity blocks let `cd!` recover the closed-form solution
# B = Y when λ = 0.
function identity_problem()
    X = Matrix{Float64}(I, 2, 2)
    Z = Matrix{Float64}(I, 2, 2)
    Y = [2.0 -1.0; 0.5 3.0]
    B = zeros(2, 2)
    reg = trues(size(B))
    regXidx = collect(1:size(B, 1))
    regZidx = collect(1:size(B, 2))
    return X, Y, Z, B, reg, regXidx, regZidx
end


############################################
# TEST Coordinate Descent - Simulated Data #
############################################

@testset "Coordinate Descent (cd.jl)" begin
    @testset "inner_update_cd! standardized" begin
        X, Z, resid, B = simple_problem()
        lambda = 0.3
        reg = trues(size(B))
        i, j = 1, 2
        B_work = copy(B)
        resid_work = copy(resid)
        resid_orig = copy(resid)
        MatrixLMnet.inner_update_cd!(i, j, B_work, resid_work, X, Z, nothing, lambda, reg)
        gradient = MatrixLMnet.calc_grad(X[:, i], Z[:, j], resid_orig)
        b_old = B[i, j]
        b2update = b_old - gradient
        b2sign = sign(b2update)
        expected_b = MatrixLMnet.prox(b_old, gradient, b2sign, lambda, nothing)
        # Residuals update via rank-one adjustment from the BLAS ger! call.
        expected_resid = resid_orig .+ (b_old - expected_b) .* (X[:, i] * transpose(Z[:, j]))
        @test isapprox(B_work[i, j], expected_b; atol = 1e-12)
        @test isapprox(resid_work, expected_resid; atol = 1e-12)
    end

    @testset "inner_update_cd! with norms" begin
        X, Z, resid, B = simple_problem()
        lambda = 0.3
        norms = fill(2.0, size(B))
        reg = trues(size(B))
        reg[2, 1] = false
        i, j = 2, 1
        B_work = copy(B)
        resid_work = copy(resid)
        resid_orig = copy(resid)
        MatrixLMnet.inner_update_cd!(i, j, B_work, resid_work, X, Z, norms, lambda, reg)
        gradient = MatrixLMnet.calc_grad(X[:, i], Z[:, j], resid_orig)
        expected_b = B[i, j] - gradient / norms[i, j]
        # Expect no shrinkage when reg is false for this coordinate.
        expected_resid = resid_orig .+ (B[i, j] - expected_b) .* (X[:, i] * transpose(Z[:, j]))
        @test isapprox(B_work[i, j], expected_b; atol = 1e-12)
        @test isapprox(resid_work, expected_resid; atol = 1e-12)
    end

    @testset "update_cd_cyclic!" begin
        X, Z, resid, B = simple_problem()
        lambda = 0.2
        norms = fill(1.25, size(B))
        reg = trues(size(B))
        B_expected = copy(B)
        resid_expected = copy(resid)
        for j in 1:size(B, 2), i in 1:size(B, 1)
            MatrixLMnet.inner_update_cd!(i, j, B_expected, resid_expected, X, Z, norms, lambda, reg)
        end
        B_actual = copy(B)
        resid_actual = copy(resid)
        MatrixLMnet.update_cd_cyclic!(B_actual, resid_actual, X, Z, norms, lambda, reg)
        @test isapprox(B_actual, B_expected; atol = 1e-12)
        @test isapprox(resid_actual, resid_expected; atol = 1e-12)
    end

    @testset "update_cd_random!" begin
        X, Z, resid, B = simple_problem()
        lambda = 0.2
        reg = trues(size(B))
        seed = 2024
        norms = nothing
        Random.seed!(seed)
    # Mirror the shuffle sequence that `update_cd_random!` will generate.
        random_pairs = Random.shuffle(collect(zip(repeat(1:size(B, 1), inner = size(B, 2)),
                                                  repeat(1:size(B, 2), outer = size(B, 1)))))
        B_expected = copy(B)
        resid_expected = copy(resid)
        for (i, j) in random_pairs
            MatrixLMnet.inner_update_cd!(i, j, B_expected, resid_expected, X, Z, norms, lambda, reg)
        end
        Random.seed!(seed)
        B_actual = copy(B)
        resid_actual = copy(resid)
        MatrixLMnet.update_cd_random!(B_actual, resid_actual, X, Z, norms, lambda, reg)
        @test isapprox(B_actual, B_expected; atol = 1e-12)
        @test isapprox(resid_actual, resid_expected; atol = 1e-12)
    end

    @testset "update_cd_active_cyclic!" begin
        X, Z, resid, B, reg, nonreg_idx, active_idx = active_problem()
        lambda = 0.4
        norms = nothing
        B_expected = copy(B)
        resid_expected = copy(resid)
        for (i, j) in zip(nonreg_idx[1], nonreg_idx[2])
            MatrixLMnet.inner_update_cd!(i, j, B_expected, resid_expected, X, Z, norms, lambda, reg)
        end
        for (i, j) in zip(active_idx[1], active_idx[2])
            MatrixLMnet.inner_update_cd!(i, j, B_expected, resid_expected, X, Z, norms, lambda, reg)
        end
        B_actual = copy(B)
        resid_actual = copy(resid)
        MatrixLMnet.update_cd_active_cyclic!(B_actual, resid_actual, X, Z, norms, lambda, reg, nonreg_idx, active_idx)
        @test isapprox(B_actual, B_expected; atol = 1e-12)
        @test isapprox(resid_actual, resid_expected; atol = 1e-12)
    end

    @testset "update_cd_active_random!" begin
        X, Z, resid, B, reg, nonreg_idx, active_idx = active_problem()
        lambda = 0.4
        norms = nothing
        seed = 42
        Random.seed!(seed)
        pairs = Random.shuffle(collect(zip(vcat(nonreg_idx[1], active_idx[1]),
                                           vcat(nonreg_idx[2], active_idx[2]))))
        B_expected = copy(B)
        resid_expected = copy(resid)
        for (i, j) in pairs
            MatrixLMnet.inner_update_cd!(i, j, B_expected, resid_expected, X, Z, norms, lambda, reg)
        end
        Random.seed!(seed)
        B_actual = copy(B)
        resid_actual = copy(resid)
        MatrixLMnet.update_cd_active_random!(B_actual, resid_actual, X, Z, norms, lambda, reg, nonreg_idx, active_idx)
        @test isapprox(B_actual, B_expected; atol = 1e-12)
        @test isapprox(resid_actual, resid_expected; atol = 1e-12)
    end

    @testset "cd! convergence" begin
        X, Y, Z, B, reg, regXidx, regZidx = identity_problem()
        lambda = 0.0
        MatrixLMnet.cd!(X, Y, Z, lambda, 1.0, B, regXidx, regZidx, reg, nothing; isVerbose = false, isRandom = false, thresh = 1e-12, maxiter = 100)
        @test isapprox(B, Y; atol = 1e-10)
    end

    @testset "cd! alpha warning" begin
        X, Y, Z, B, reg, regXidx, regZidx = identity_problem()
        lambda = 0.0
        @test_logs (:warn, "Only L1-penalized coordinate descent is available for the moment.") begin
            MatrixLMnet.cd!(X, Y, Z, lambda, 0.5, B, regXidx, regZidx, reg, nothing; isVerbose = false, isRandom = false, thresh = 1e-12, maxiter = 100)
        end
        @test isapprox(B, Y; atol = 1e-10)
    end

    @testset "cd_active! convergence" begin
        X, Y, Z, B, reg, regXidx, regZidx = identity_problem()
        lambda = 0.0
        MatrixLMnet.cd_active!(X, Y, Z, lambda, 1.0, B, regXidx, regZidx, reg, nothing; isVerbose = false, isRandom = false, thresh = 1e-12, maxiter = 100)
        @test isapprox(B, Y; atol = 1e-10)
    end

    @testset "cd_active! alpha warning" begin
        X, Y, Z, B, reg, regXidx, regZidx = identity_problem()
        lambda = 0.0
        @test_logs (:warn, "Only L1-penalized coordinate descent is available for the moment.") begin
            MatrixLMnet.cd_active!(X, Y, Z, lambda, 0.5, B, regXidx, regZidx, reg, nothing; isVerbose = false, isRandom = false, thresh = 1e-12, maxiter = 100)
        end
        @test isapprox(B, Y; atol = 1e-10)
    end
end
