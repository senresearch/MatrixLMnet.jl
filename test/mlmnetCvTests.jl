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

numVersion = VERSION
if Int(numVersion.minor) < 7
      tolVersion=2e-1
else
      tolVersion=1e-5
end 

# Folding indices
mrow_folds = Helium.readhe(joinpath(dataDir, "row_folds.he"))
row_folds = [mrow_folds[:,i] for i in 1:size(mrow_folds, 2)]

mcol_folds = Helium.readhe(joinpath(dataDir, "col_folds.he"))
col_folds = [mcol_folds[:,i] for i in 1:size(mcol_folds, 2)]

rng = MatrixLMnet.Random.Xoshiro(2021);
#############################################
# TEST 1 Lasso vs Elastic Net (𝛼=1) - ista #
#############################################

# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, row_folds, col_folds, method = "ista", addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, row_folds, col_folds, method = "ista",  addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - ista cv
smmr_ista= Helium.readhe(joinpath(dataDir, "smmr_ista.he"))

println("CV Lasso vs Elastic Net when α=1 test 1 - ista: ",
         @test ≈(smmr_Net1.AvgMSE, smmr_ista[:,1]; atol=tolVersion) && ≈(smmr_Net1.AvgPropZero, smmr_ista[:,2], atol = tolVersion) &&
         ≈(smmr_Net2.AvgMSE, smmr_ista[:,1];atol=tolVersion) && ≈(smmr_Net2.AvgPropZero, smmr_ista[:,2]; atol=tolVersion))

#############################################
# TEST 2 Lasso vs Elastic Net (𝛼=1) - fista #
#############################################

# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, row_folds, col_folds, method = "fista", addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
est2 = MatrixLMnet.mlmnet_cv(dat, λ, row_folds, col_folds, method = "fista",  addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - fista cv
smmr_fista= Helium.readhe(joinpath(dataDir, "smmr_fista.he"))

println("CV Lasso vs Elastic Net when α=1 test 2 - fista: ",
@test ≈(smmr_Net1.AvgMSE, smmr_fista[:,1]; atol=tolVersion) && ≈(smmr_Net1.AvgPropZero, smmr_fista[:,2], atol = tolVersion) &&
≈(smmr_Net2.AvgMSE, smmr_fista[:,1];atol=tolVersion) && ≈(smmr_Net2.AvgPropZero, smmr_fista[:,2]; atol=tolVersion))

##########################################################
# TEST 3 Lasso vs Elastic Net (𝛼=1) - fista backtracking #
##########################################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, row_folds, col_folds, method = "fista_bt", addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, row_folds, col_folds, method = "fista_bt",  addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - fista-bt cv
smmr_fistabt= Helium.readhe(joinpath(dataDir, "smmr_fistabt.he"))

println("CV Lasso vs Elastic Net when α=1 test 3 - fista-bt: ",
@test ≈(smmr_Net1.AvgMSE, smmr_fistabt[:,1]; atol=tolVersion) && ≈(smmr_Net1.AvgPropZero, smmr_fistabt[:,2], atol = tolVersion) &&
≈(smmr_Net2.AvgMSE, smmr_fistabt[:,1];atol=tolVersion) && ≈(smmr_Net2.AvgPropZero, smmr_fistabt[:,2]; atol=tolVersion))


############################################
# TEST 4 Lasso vs Elastic Net (𝛼=1) - admm #
############################################

# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, row_folds, col_folds, method = "admm", addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, row_folds, col_folds, method = "admm",  addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - fista-bt cv
smmr_admm = Helium.readhe(joinpath(dataDir, "smmr_admm.he"))

println("CV Lasso vs Elastic Net when α=1 test 4 - admm: ",
@test ≈(smmr_Net1.AvgMSE, smmr_admm[:,1]; atol=tolVersion) && ≈(smmr_Net1.AvgPropZero, smmr_admm[:,2], atol = tolVersion) &&
≈(smmr_Net2.AvgMSE, smmr_admm[:,1];atol=tolVersion) && ≈(smmr_Net2.AvgPropZero, smmr_admm[:,2]; atol=tolVersion))

##########################################
# TEST 5 Lasso vs Elastic Net (𝛼=1) - cd #
##########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est1 = MatrixLMnet.mlmnet_cv(dat, λ, α, row_folds, col_folds, method = "cd", addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net1 = MatrixLMnet.lambda_min(est1);

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng)
est2 = MatrixLMnet.mlmnet_cv(dat, λ, row_folds, col_folds, method = "cd",  addZIntercept = false, addXIntercept = false, isVerbose = false);
smmr_Net2 = MatrixLMnet.lambda_min(est2);

# Lasso penalized regression - cd cv
smmr_cd = Helium.readhe(joinpath(dataDir, "smmr_cd.he"))

println("CV Lasso vs Elastic Net when α=1 test 5 - cd: ",
@test ≈(smmr_Net1.AvgMSE, smmr_cd[:,1]; atol=tolVersion) && ≈(smmr_Net1.AvgPropZero, smmr_cd[:,2], atol = tolVersion) &&
≈(smmr_Net2.AvgMSE, smmr_cd[:,1];atol=tolVersion) && ≈(smmr_Net2.AvgPropZero, smmr_cd[:,2]; atol=tolVersion))

println("Tests mlmnet_cv finished!")

