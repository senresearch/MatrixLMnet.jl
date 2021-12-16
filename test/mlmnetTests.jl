###########
# Library #
###########
# using MatrixLM
# using Distributions, Random, Statistics, LinearAlgebra, StatsBase
# using Random
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

############################################
# TEST 1 Lasso vs Elastic Net (𝛼=1) - ista #
############################################


# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet(dat, λ, α, method = "ista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net1 = est1.B[:, :, 3, 1];

# Elastic net penalized regression
est2 = MatrixLMnet.mlmnet(dat, λ, method = "ista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net2 = est2.B[:, :, 3, 1];

# Lasso penalized regression - ista
B_ista= Helium.readhe(joinpath(dataDir, "B_ista.he"))

println("Lasso vs Elastic Net when α=1 test 1 - ista: ", @test (est_B_Net1 == B_ista) && (est_B_Net2 == B_ista))

#############################################
# TEST 2 Lasso vs Elastic Net (𝛼=1) - fista #
#############################################

# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet(dat, λ, α, method = "fista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net1 = est1.B[:, :, 3, 1];

# Elastic net penalized regression
est2 = MatrixLMnet.mlmnet(dat, λ, method = "fista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net2 = est2.B[:, :, 3, 1];

# Lasso penalized regression - fista
B_fista= Helium.readhe(joinpath(dataDir, "B_fista.he"))

println("Lasso vs Elastic Net when α=1 test 2 - fista: ", @test (est_B_Net1 == B_fista) && (est_B_Net2 == B_fista))

##########################################################
# TEST 3 Lasso vs Elastic Net (𝛼=1) - fista backtracking #
##########################################################

# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet(dat, λ, α, method = "fista_bt", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net1 = est1.B[:, :, 3, 1];

# Elastic net penalized regression
est2 = MatrixLMnet.mlmnet(dat, λ, method = "fista_bt",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net2 = est2.B[:, :, 3, 1];

# Lasso penalized regression - fista-bt
B_fistabt = Helium.readhe(joinpath(dataDir, "B_fistabt.he"))

println("Lasso vs Elastic Net when α=1 test 3 - fista-bt: ", @test (est_B_Net1 == B_fistabt) && (est_B_Net2 == B_fistabt))


############################################
# TEST 4 Lasso vs Elastic Net (𝛼=1) - admm #
############################################

# Elastic net penalized regression
est1 = MatrixLMnet.mlmnet(dat, λ, α, method = "admm", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net1 = est1.B[:, :, 3, 1];

# Elastic net penalized regression
est2 = MatrixLMnet.mlmnet(dat, λ, method = "admm",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net2 = est2.B[:, :, 3, 1];

# Lasso penalized regression - admm
B_admm = Helium.readhe(joinpath(dataDir, "B_admm.he"))

println("Lasso vs Elastic Net when α=1 test 4 - admm: ", @test (est_B_Net1 == B_admm) && (est_B_Net2 == B_admm))


##########################################
# TEST 5 Lasso vs Elastic Net (𝛼=1) - cd #
##########################################

# Elastic net penalized regression
MatrixLMnet.Random.seed!(2021)
est1 = MatrixLMnet.mlmnet(dat, λ, α, method = "cd", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net1 = est1.B[:, :, 3, 1];

# Elastic net penalized regression
MatrixLMnet.Random.seed!(2021)
est2 = MatrixLMnet.mlmnet(dat, λ, method = "cd",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
est_B_Net2 = est2.B[:, :, 3, 1];

# Lasso penalized regression - cd
B_cd = Helium.readhe(joinpath(dataDir, "B_cd.he"))

println("Lasso vs Elastic Net when α=1 test 5 - cd: ", @test (est_B_Net1 == B_cd) && (est_B_Net2 == B_cd))


println("Tests mlmnet finished!")
