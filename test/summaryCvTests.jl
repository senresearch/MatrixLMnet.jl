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

#=
Description:
-----------

Model: 𝐘 = 𝐗 𝛃 𝐙' + 𝜎𝜖, with 𝜖∼𝑁(0,1) 

Simulate data set consisting  of 20/20/200 observations and 8 predictors.
We let 𝛽₁ = (3, 1.5, 0, 0, 2, 0, 0, 0), 𝛽₂ = (0, 1.5, 0, 3.5, 2, 0, 0 , 2) where
𝛃 = [𝛽₁, 𝛽₂] and 𝜎 = 3.
The pairwise correlation between 𝑋ᵢ and 𝑋ⱼ was set to be 𝑐𝑜𝑟(𝑖,𝑗)=(0.5)^|𝑖−𝑗|.
Here, the Z matrix is an identity matrix.
=#

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

# Hyper parameters
λ = [10.0, 5.0, 3.0]
α = [1.0, 0.5]

rng = 2021#MatrixLMnet.Random.MersenneTwister(2021)

numVersion = VERSION
if Int(numVersion.minor) < 7
      tolVersion=2e-1
else
      tolVersion=1e-6
end 


##########################################
# TEST 1 Summary cross-validation - ista #
##########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "ista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net1 = MatrixLMnet.mlmnet_cv_summary(est1);
smmr_min_Net1 = MatrixLMnet.lambda_min(est1);

idxSmmr = argmin(smmr_Net1[:, :AvgMSE])
test_ElasticNet = (smmr_Net1[idxSmmr, :AvgMSE] == smmr_min_Net1[1, :AvgMSE]) && 
        (smmr_Net1[idxSmmr, :Lambda] == smmr_min_Net1[1, :Lambda]) &&
        (smmr_Net1[idxSmmr, :Alpha] == smmr_min_Net1[1, :Alpha])

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "ista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net2 = MatrixLMnet.mlmnet_cv_summary(est2);
smmr_min_Net2 = MatrixLMnet.lambda_min(est2);

idxSmmr = argmin(smmr_Net2[:, :AvgMSE])
test_Lasso = (smmr_Net2[idxSmmr, :AvgMSE] == smmr_min_Net2[1, :AvgMSE]) && 
        (smmr_Net2[idxSmmr, :Lambda] == smmr_min_Net2[1, :Lambda]) &&
        (smmr_Net2[idxSmmr, :Alpha] == smmr_min_Net2[1, :Alpha])

println("Summary cross-validation test 1 - ista: ",
         @test (test_ElasticNet && test_Lasso))


###########################################
# TEST 2 Summary cross-validation - fista #
###########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "fista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net1 = MatrixLMnet.mlmnet_cv_summary(est1);
smmr_min_Net1 = MatrixLMnet.lambda_min(est1);

idxSmmr = argmin(smmr_Net1[:, :AvgMSE])
test_ElasticNet = (smmr_Net1[idxSmmr, :AvgMSE] == smmr_min_Net1[1, :AvgMSE]) && 
        (smmr_Net1[idxSmmr, :Lambda] == smmr_min_Net1[1, :Lambda]) &&
        (smmr_Net1[idxSmmr, :Alpha] == smmr_min_Net1[1, :Alpha])

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "fista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net2 = MatrixLMnet.mlmnet_cv_summary(est2);
smmr_min_Net2 = MatrixLMnet.lambda_min(est2);

idxSmmr = argmin(smmr_Net2[:, :AvgMSE])
test_Lasso = (smmr_Net2[idxSmmr, :AvgMSE] == smmr_min_Net2[1, :AvgMSE]) && 
        (smmr_Net2[idxSmmr, :Lambda] == smmr_min_Net2[1, :Lambda]) &&
        (smmr_Net2[idxSmmr, :Alpha] == smmr_min_Net2[1, :Alpha])

println("Summary cross-validation test 2 - fista: ",
         @test (test_ElasticNet && test_Lasso))         


########################################################
# TEST 3 Summary cross-validation - fista backtracking #
########################################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "fista_bt", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net1 = MatrixLMnet.mlmnet_cv_summary(est1);
smmr_min_Net1 = MatrixLMnet.lambda_min(est1);

idxSmmr = argmin(smmr_Net1[:, :AvgMSE])
test_ElasticNet = (smmr_Net1[idxSmmr, :AvgMSE] == smmr_min_Net1[1, :AvgMSE]) && 
        (smmr_Net1[idxSmmr, :Lambda] == smmr_min_Net1[1, :Lambda]) &&
        (smmr_Net1[idxSmmr, :Alpha] == smmr_min_Net1[1, :Alpha])

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "fista_bt",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net2 = MatrixLMnet.mlmnet_cv_summary(est2);
smmr_min_Net2 = MatrixLMnet.lambda_min(est2);

idxSmmr = argmin(smmr_Net2[:, :AvgMSE])
test_Lasso = (smmr_Net2[idxSmmr, :AvgMSE] == smmr_min_Net2[1, :AvgMSE]) && 
        (smmr_Net2[idxSmmr, :Lambda] == smmr_min_Net2[1, :Lambda]) &&
        (smmr_Net2[idxSmmr, :Alpha] == smmr_min_Net2[1, :Alpha])

println("Summary cross-validation test 3 - fista_bt: ",
         @test (test_ElasticNet && test_Lasso))                  


##########################################
# TEST 4 Summary cross-validation - admm #
##########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "admm", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net1 = MatrixLMnet.mlmnet_cv_summary(est1);
smmr_min_Net1 = MatrixLMnet.lambda_min(est1);

idxSmmr = argmin(smmr_Net1[:, :AvgMSE])
test_ElasticNet = (smmr_Net1[idxSmmr, :AvgMSE] == smmr_min_Net1[1, :AvgMSE]) && 
        (smmr_Net1[idxSmmr, :Lambda] == smmr_min_Net1[1, :Lambda]) &&
        (smmr_Net1[idxSmmr, :Alpha] == smmr_min_Net1[1, :Alpha])

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "admm",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net2 = MatrixLMnet.mlmnet_cv_summary(est2);
smmr_min_Net2 = MatrixLMnet.lambda_min(est2);

idxSmmr = argmin(smmr_Net2[:, :AvgMSE])
test_Lasso = (smmr_Net2[idxSmmr, :AvgMSE] == smmr_min_Net2[1, :AvgMSE]) && 
        (smmr_Net2[idxSmmr, :Lambda] == smmr_min_Net2[1, :Lambda]) &&
        (smmr_Net2[idxSmmr, :Alpha] == smmr_min_Net2[1, :Alpha])

println("Summary cross-validation test 4 - admm: ",
         @test (test_ElasticNet && test_Lasso))                  
    
         
########################################
# TEST 5 Summary cross-validation - cd #
########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "admm",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
# Summaries
smmr_Net2 = MatrixLMnet.mlmnet_cv_summary(est2);
smmr_min_Net2 = MatrixLMnet.lambda_min(est2);

idxSmmr = argmin(smmr_Net2[:, :AvgMSE])
test_Lasso = (smmr_Net2[idxSmmr, :AvgMSE] == smmr_min_Net2[1, :AvgMSE]) && 
        (smmr_Net2[idxSmmr, :Lambda] == smmr_min_Net2[1, :Lambda]) &&
        (smmr_Net2[idxSmmr, :Alpha] == smmr_min_Net2[1, :Alpha])

println("Summary cross-validation test 5 - admm: ",
         @test (test_Lasso))                           