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
α = [1.0]

rng = MersenneTwister(2021)

#############################################
# TEST 1 Lasso vs Elastic Net (𝛼=1) - ista #
#############################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "ista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "ista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - ista cv
smmr_ista= Helium.readhe(joinpath(dataDir, "smmr_ista.he"))

println("CV Lasso vs Elastic Net when α=1 test 1 - ista: ",
         @test ≈(smmr_Net1.AvgMSE, smmr_ista[:,1]; atol=1.2e-8) && ≈(smmr_Net1.AvgPercentZero, smmr_ista[:,2], atol = 1.2e-8) &&
         ≈(smmr_Net2.AvgMSE, smmr_ista[:,1];atol=1.2e-8) && ≈(smmr_Net2.AvgPercentZero, smmr_ista[:,2]; atol=1.2e-8))

#############################################
# TEST 2 Lasso vs Elastic Net (𝛼=1) - fista #
#############################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "fista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "fista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - fista cv
smmr_fista= Helium.readhe(joinpath(dataDir, "smmr_fista.he"))

println("CV Lasso vs Elastic Net when α=1 test 2 - fista: ",
         @test smmr_Net1.AvgMSE ≈ smmr_fista[:,1] && smmr_Net1.AvgPercentZero ≈ smmr_fista[:,2] &&
               smmr_Net2.AvgMSE ≈ smmr_fista[:,1] && smmr_Net2.AvgPercentZero ≈ smmr_fista[:,2] )

##########################################################
# TEST 3 Lasso vs Elastic Net (𝛼=1) - fista backtracking #
##########################################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "fista_bt", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "fista_bt",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - fista-bt cv
smmr_fistabt= Helium.readhe(joinpath(dataDir, "smmr_fistabt.he"))

println("CV Lasso vs Elastic Net when α=1 test 3 - fista-bt: ",
         @test smmr_Net1.AvgMSE ≈ smmr_fistabt[:,1] && smmr_Net1.AvgPercentZero ≈ smmr_fistabt[:,2] &&
               smmr_Net2.AvgMSE ≈ smmr_fistabt[:,1] && smmr_Net2.AvgPercentZero ≈ smmr_fistabt[:,2] )
≈     

############################################
# TEST 4 Lasso vs Elastic Net (𝛼=1) - admm #
############################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "admm", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "admm",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - fista-bt cv
smmr_admm = Helium.readhe(joinpath(dataDir, "smmr_admm.he"))

println("CV Lasso vs Elastic Net when α=1 test 4 - admm: ",
         @test smmr_Net1.AvgMSE ≈ smmr_admm[:,1] && smmr_Net1.AvgPercentZero ≈ smmr_admm[:,2] &&
               smmr_Net2.AvgMSE ≈ smmr_admm[:,1] && smmr_Net2.AvgPercentZero ≈ smmr_admm[:,2] )


##########################################
# TEST 5 Lasso vs Elastic Net (𝛼=1) - cd #
##########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, 10, 1, method = "cd", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, 10, 1, method = "cd",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - cd cv
smmr_cd = Helium.readhe(joinpath(dataDir, "smmr_cd.he"))

println("CV Lasso vs Elastic Net when α=1 test 5 - cd: ",
         @test smmr_Net1.AvgMSE ≈ smmr_cd[:,1] && smmr_Net1.AvgPercentZero ≈ smmr_cd[:,2] &&
               smmr_Net2.AvgMSE ≈ smmr_cd[:,1] && smmr_Net2.AvgPercentZero ≈ smmr_cd[:,2] )

println("Tests mlmnet_cv finished!")

