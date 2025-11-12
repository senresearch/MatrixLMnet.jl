###########
# Library #
###########
using MatrixLMnet
using Test

TOL = 1.0e-12

#############
# Functions #
#############


# Functions to generate Small deterministic dataset used across tests.
function small_data()
	X = [1.0 0.5; 0.3 1.2]
	Y = [0.5 -0.3; 1.1 0.2]
	Z = [1.0 -0.4; 0.6 0.8]
	B = [0.2 -0.1; 0.05 0.15]
	reg = BitArray([true false; true true])
	return X, Y, Z, B, reg
end


# Functions to compute expected values for ISTA updates.
# Reference computation for a single ISTA update pass. This mirrors the
# algorithm implemented in `update_ista!` and is used to construct the
# expected B matrix for comparisons. 
function ista_expected(B0, grad, reg, lambdaL1, lambdaL2, step, norms)
	B_expected = copy(B0)
	for j in 1:size(B0, 2), i in 1:size(B0, 1)
		b_old = B_expected[i, j]
		if norms === nothing
			b2update = b_old - step * grad[i, j]
			b2sign = sign(b2update)
			B_expected[i, j] = reg[i, j] ? MatrixLMnet.prox(b_old, grad[i, j], b2sign, lambdaL1, nothing, step) / (1 + 2 * lambdaL2 * step) : b2update
		else
			b2update = b_old - step * grad[i, j] / norms[i, j]
			b2sign = sign(b2update)
			B_expected[i, j] = reg[i, j] ? MatrixLMnet.prox(b_old, grad[i, j], b2sign, lambdaL1, norms[i, j], step) / (1 + 2 * lambdaL2 * step) : b2update
		end
	end
	return B_expected
end

# Functions to compute expected values for FISTA updates.
# Reference computation for a single FISTA update pass. This mirrors the
# in-place updates in `update_fista!`. It updates B_prev when needed,
# computes a shrinked B from the extrapolated A, and then forms the
# extrapolated A for the next iteration. The iter_val controls the
# Nesterov extrapolation factor.
function fista_expected(B0, Bprev0, A0, grad, reg, lambdaL1, lambdaL2, step, norms, iter_val)
	B_expected = copy(B0)
	Bprev_expected = copy(Bprev0)
	A_expected = copy(A0)
	factor = (iter_val - 1.0) / (iter_val + 2.0)
	for j in 1:size(B0, 2), i in 1:size(B0, 1)
		b_old = B_expected[i, j]
		if Bprev_expected[i, j] != b_old
			Bprev_expected[i, j] = b_old
		end
		a_old = A_expected[i, j]
		if norms === nothing
			raw = a_old - step * grad[i, j]
			s = sign(raw)
			raw = reg[i, j] ? MatrixLMnet.prox(a_old, grad[i, j], s, lambdaL1, nothing, step) / (1 + 2 * lambdaL2 * step) : raw
		else
			raw = a_old - step * grad[i, j] / norms[i, j]
			s = sign(raw)
			raw = reg[i, j] ? MatrixLMnet.prox(a_old, grad[i, j], s, lambdaL1, norms[i, j], step) / (1 + 2 * lambdaL2 * step) : raw
		end
		B_expected[i, j] = raw
		A_expected[i, j] = raw + factor * (raw - Bprev_expected[i, j])
	end
	return B_expected, Bprev_expected, A_expected
end



####################################################
# TEST  FISTA/ISTA Update Helpers - Simulated Data #
####################################################

# These unit tests exercise the low-level update functions `update_ista!`
# and `update_fista!` directly on small deterministic matrices. 



@testset "ISTA/FISTA updates" begin
    
	@testset "update_ista! with standardized design" begin
		X, Y, Z, B0, reg = small_data()
		lambdaL1 = 0.25
		lambdaL2 = 0.1
		step = 0.2
		resid0 = MatrixLMnet.calc_resid(X, Y, Z, B0)
		grad_expected = zeros(size(B0))
		MatrixLMnet.calc_grad!(grad_expected, X, Z, resid0)
		B_expected = ista_expected(B0, grad_expected, reg, lambdaL1, lambdaL2, step, nothing)
		resid_expected = MatrixLMnet.calc_resid(X, Y, Z, B_expected)

		B_actual = copy(B0)
		resid_actual = copy(resid0)
		grad_actual = similar(B0)
		MatrixLMnet.update_ista!(B_actual, resid_actual, grad_actual, X, Y, Z, nothing, lambdaL1, lambdaL2, reg, [step])

		@test isapprox(grad_actual, grad_expected; atol = TOL)
		@test isapprox(B_actual, B_expected; atol = TOL)
		@test isapprox(resid_actual, resid_expected; atol = TOL)
	end

	@testset "update_ista! with norms" begin
		X, Y, Z, B0, reg = small_data()
		lambdaL1 = 0.25
		lambdaL2 = 0.1
		step = 0.15
		norms = [1.5 0.9; 1.2 2.0]
		resid0 = MatrixLMnet.calc_resid(X, Y, Z, B0)
		grad_expected = zeros(size(B0))
		MatrixLMnet.calc_grad!(grad_expected, X, Z, resid0)
		B_expected = ista_expected(B0, grad_expected, reg, lambdaL1, lambdaL2, step, norms)
		resid_expected = MatrixLMnet.calc_resid(X, Y, Z, B_expected)

		B_actual = copy(B0)
		resid_actual = copy(resid0)
		grad_actual = similar(B0)
		MatrixLMnet.update_ista!(B_actual, resid_actual, grad_actual, X, Y, Z, norms, lambdaL1, lambdaL2, reg, [step])

		@test isapprox(grad_actual, grad_expected; atol = TOL)
		@test isapprox(B_actual, B_expected; atol = TOL)
		@test isapprox(resid_actual, resid_expected; atol = TOL)
	end

	@testset "update_fista! with standardized design" begin
		X, Y, Z, B0, reg = small_data()
		lambdaL1 = 0.2
		lambdaL2 = 0.05
		step = 0.1
		iter_val = 3
		A0 = [0.18 -0.08; 0.07 0.13]
		Bprev0 = [0.22 -0.09; 0.06 0.16]
		resid_for_grad = MatrixLMnet.calc_resid(X, Y, Z, A0)
		grad_expected = zeros(size(B0))
		MatrixLMnet.calc_grad!(grad_expected, X, Z, resid_for_grad)
		B_expected, Bprev_expected, A_expected = fista_expected(B0, Bprev0, A0, grad_expected, reg, lambdaL1, lambdaL2, step, nothing, iter_val)
		resid_B_expected = MatrixLMnet.calc_resid(X, Y, Z, B_expected)

		B_actual = copy(B0)
		Bprev_actual = copy(Bprev0)
		A_actual = copy(A0)
		resid_actual = similar(Y)
		resid_B_actual = similar(Y)
		grad_actual = similar(B0)
		MatrixLMnet.update_fista!(B_actual, Bprev_actual, A_actual, resid_actual, resid_B_actual, grad_actual, X, Y, Z, nothing, lambdaL1, lambdaL2, reg, [iter_val], [step])

		@test isapprox(resid_actual, resid_for_grad; atol = TOL)
		@test isapprox(grad_actual, grad_expected; atol = TOL)
		@test isapprox(Bprev_actual, Bprev_expected; atol = TOL)
		@test isapprox(B_actual, B_expected; atol = TOL)
		@test isapprox(A_actual, A_expected; atol = TOL)
		@test isapprox(resid_B_actual, resid_B_expected; atol = TOL)
	end

	@testset "update_fista! with norms" begin
		X, Y, Z, B0, reg = small_data()
		lambdaL1 = 0.2
		lambdaL2 = 0.05
		step = 0.08
		iter_val = 4
		norms = [1.4 0.7; 1.1 1.8]
		A0 = [0.21 -0.06; 0.08 0.17]
		Bprev0 = [0.19 -0.11; 0.05 0.14]
		resid_for_grad = MatrixLMnet.calc_resid(X, Y, Z, A0)
		grad_expected = zeros(size(B0))
		MatrixLMnet.calc_grad!(grad_expected, X, Z, resid_for_grad)
		B_expected, Bprev_expected, A_expected = fista_expected(B0, Bprev0, A0, grad_expected, reg, lambdaL1, lambdaL2, step, norms, iter_val)
		resid_B_expected = MatrixLMnet.calc_resid(X, Y, Z, B_expected)

		B_actual = copy(B0)
		Bprev_actual = copy(Bprev0)
		A_actual = copy(A0)
		resid_actual = similar(Y)
		resid_B_actual = similar(Y)
		grad_actual = similar(B0)
		MatrixLMnet.update_fista!(B_actual, Bprev_actual, A_actual, resid_actual, resid_B_actual, grad_actual, X, Y, Z, norms, lambdaL1, lambdaL2, reg, [iter_val], [step])

		@test isapprox(resid_actual, resid_for_grad; atol = TOL)
		@test isapprox(grad_actual, grad_expected; atol = TOL)
		@test isapprox(Bprev_actual, Bprev_expected; atol = TOL)
		@test isapprox(B_actual, B_expected; atol = TOL)
		@test isapprox(A_actual, A_expected; atol = TOL)
		@test isapprox(resid_B_actual, resid_B_expected; atol = TOL)
	end
end
