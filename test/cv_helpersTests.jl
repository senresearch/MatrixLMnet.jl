###########
# Library #
###########
# using Random
using MatrixLMnet
using Helium
using Test

#####################################################################
# TEST Cross Validation Lasso vs Elastic Net (𝛼=1) - Simulated Data #
#####################################################################

# Data testing directory name
dataDir = realpath(joinpath(@__DIR__,"data"))

# Get predictors
X = Helium.readhe(joinpath(dataDir, "Xmat.he"))

# Get response
Y = Helium.readhe(joinpath(dataDir, "Ymat.he"))

# Get Z matrix
Z = Helium.readhe(joinpath(dataDir, "Zmat.he"))


# Build raw data object from MatrixLM.jl
dat = RawData(Response(Y), Predictors(X, Z));

dat.n

test = make_folds(dat.n, 10, max(10, 1))
testcol = make_folds(dat.m, 1, max(10, 1))


# Check https://mldatautilsjl.readthedocs.io/en/latest/
# check resid

###################
# Test make_folds #
###################

#######################################################
# Test 1: Basic functionality with default parameters #
#######################################################
@test length(make_folds(100)) == 10
@test all([length(fold) ≈ 90 for fold in make_folds(100)]) # Approximate because the folds may not be exactly equal

##################################
# Test 2: Non-default `k` values #
##################################
@test length(make_folds(100, 5)) == 5
@test length(make_folds(100, 100)) == 100 # Each fold should have exactly one element
@test length(make_folds(100, 20)) == 20

#################################
# Test 3: `k` Equals 1 Scenario #
#################################
@test length(make_folds(100, 1, 5)) == 5 # Should repeat 5 times
@test all([length(fold) == 100 for fold in make_folds(100, 1, 5)]) # Each fold contains all indices

##############################
# Test 4: Invalid `k` values #
##############################
@test_throws ErrorException make_folds(100, 0)
@test_throws ErrorException make_folds(100, -1)

#################################
# Test 5: Type Check (Optional) #
#################################
@test_throws MethodError make_folds(100.0, 10) # Float instead of Int for `n`
@test_throws MethodError make_folds(100, "10") # String instead of Int for `k`




#########################
# Test make_folds_conds #
#########################




##################
# Test findnotin #
##################

# Basic Functionality
@testset "Basic Functionality" begin
    a = [1, 2, 3, 4]
    b = [3, 4, 5, 6]
    @test MatrixLMnet.findnotin(a, b) == [5, 6]
end

# No Missing Elements
@testset "No Missing Elements" begin
    a = [1, 2, 3, 4]
    b = [1, 2, 3, 4]
    @test isempty(MatrixLMnet.findnotin(a, b))
end

# All Elements Missing
@testset "All Elements Missing" begin
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    @test MatrixLMnet.findnotin(a, b) == b
end

# Duplicate Elements
@testset "Duplicate Elements" begin
    a = [1, 2, 3]
    b = [2, 2, 4, 4]
    @test MatrixLMnet.findnotin(a, b) == [4, 4] # Expect duplicates in b that are not in a to be returned as-is
end



a= [1, 2, 3, 4]
b = collect(1:10)
setdiff(b, a)
setdiff(b, [])
setdiff(b, b)

b[b .∉ Ref(a)]


##################
# Cross-validate #
##################

function compute_MatrixLMnet() 


end

scores = MatrixLMnet.cross_validate(
    inds -> compute_center(data[:, inds]),        # training function
    (c, inds) -> compute_rmse(c, data[:, inds]),  # evaluation function
    n,              # total number of samples
    Kfold(n, 5))    # cross validation plan: 5-fold



