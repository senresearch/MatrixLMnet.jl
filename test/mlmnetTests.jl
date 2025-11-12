###########
# Library #
###########
# # using MatrixLM
# # using Distributions, Random, Statistics, LinearAlgebra, StatsBase
# # using Random
using MatrixLMnet
using Helium
using Test


####################################################
# TEST Lasso vs Elastic Net (𝛼=1) - Simulated Data #
####################################################

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

rng = StableRNG(123) # MatrixLMnet.Random.MersenneTwister(2021);

#############################################
# TEST 1a Lasso vs Elastic Net (𝛼=1) - ista #
#############################################

# Elastic net penalized regression
est_ista_1 = MatrixLMnet.mlmnet(dat, λ, α, method = "ista", addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_ista_1 = est_ista_1.B[:, :, 3, 1];

# Elastic net penalized regression
est_ista_2 = MatrixLMnet.mlmnet(dat, λ, method = "ista",  addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_ista_2 = est_ista_2.B[:, :, 3, 1];

# Lasso penalized regression - ista
B_ista= Helium.readhe(joinpath(dataDir, "B_ista.he"))

println("Lasso vs Elastic Net when α=1 test 1 - ista: ", @test (B_Net_ista_1 ≈ B_ista) && (B_Net_ista_2 ≈ B_ista))

#############################################
# TEST 2 Lasso vs Elastic Net (𝛼=1) - fista #
#############################################

# Elastic net penalized regression
est_fista_1 = MatrixLMnet.mlmnet(dat, λ, α, method = "fista", addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_fista_1 = est_fista_1.B[:, :, 3, 1];

# Elastic net penalized regression
est_fista_2 = MatrixLMnet.mlmnet(dat, λ, method = "fista",  addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_fista_2 = est_fista_2.B[:, :, 3, 1];

# Lasso penalized regression - fista
B_fista= Helium.readhe(joinpath(dataDir, "B_fista.he"))

println("Lasso vs Elastic Net when α=1 test 2 - fista: ", @test (B_Net_fista_1 ≈ B_fista) && (B_Net_fista_2 ≈ B_fista))

##########################################################
# TEST 3 Lasso vs Elastic Net (𝛼=1) - fista backtracking #
##########################################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng, 2024)
est_fistabt_1 = MatrixLMnet.mlmnet(dat, λ, α, method = "fista_bt", addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_fistabt_1 = est_fistabt_1.B[:, :, 3, 1];

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng, 2024)
est_fistabt_2 = MatrixLMnet.mlmnet(dat, λ, method = "fista_bt",  addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_fistabt_2 = est_fistabt_2.B[:, :, 3, 1];

# Lasso penalized regression - fista-bt
B_fistabt = Helium.readhe(joinpath(dataDir, "B_fistabt.he"))

println("Lasso vs Elastic Net when α=1 test 3 - fista-bt: ", @test (B_Net_fistabt_1 ≈ B_fistabt) && (B_Net_fistabt_2 ≈ B_fistabt))


############################################
# TEST 4 Lasso vs Elastic Net (𝛼=1) - admm #
############################################

# Elastic net penalized regression
est_admm_1 = MatrixLMnet.mlmnet(dat, λ, α, method = "admm", addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_admm_1 = est_admm_1.B[:, :, 3, 1];

# Elastic net penalized regression
est_admm_2 = MatrixLMnet.mlmnet(dat, λ, method = "admm",  addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_admm_2 = est_admm_2.B[:, :, 3, 1];

# Lasso penalized regression - admm
B_admm = Helium.readhe(joinpath(dataDir, "B_admm.he"))

println("Lasso vs Elastic Net when α=1 test 4 - admm: ", @test (B_Net_admm_1 ≈ B_admm) && (B_Net_admm_2 ≈ B_admm))


##########################################
# TEST 5 Lasso vs Elastic Net (𝛼=1) - cd #
##########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng, 2024)
est_cd_1 = MatrixLMnet.mlmnet(dat, λ, α, method = "cd", addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_cd_1 = est_cd_1.B[:, :, 3, 1];

# Elastic net penalized regression
MatrixLMnet.Random.seed!(rng, 2024)
est_cd_2 = MatrixLMnet.mlmnet(dat, λ, method = "cd",  addZIntercept = false, addXIntercept = false, isVerbose = false);
B_Net_cd_2 = est_cd_2.B[:, :, 3, 1];

# Lasso penalized regression - cd
B_cd = Helium.readhe(joinpath(dataDir, "B_cd.he"))

println("Lasso vs Elastic Net when α=1 test 5 - cd: ", @test ≈(B_Net_cd_1,  B_cd; atol=1.2e-4) && ≈(B_Net_cd_2, B_cd;  atol=1.2e-4))


##################################
# TEST 6 Data remains unchanged  #
##################################

# Elastic net penalized regression
original_dat_predictors_colsize = size(dat.predictors.X, 2);
est_ista_1 = MatrixLMnet.mlmnet(dat, λ, α, method = "ista", addZIntercept = false, addXIntercept = true, isVerbose = false);


println("Test that original data remains unchanged test 6: ", 
    @test original_dat_predictors_colsize == size(dat.predictors.X, 2))

println("Tests mlmnet finished!")
