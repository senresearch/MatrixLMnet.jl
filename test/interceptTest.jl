# ###########
# # Library #
# ###########
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

# Get predictors (already centered)
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

rng = MatrixLMnet.Random.MersenneTwister(2021)

# flag intercept
flag_intercept = false 


#######
# MLM #
#######

mlmdata = RawData(Response(Y), Predictors(X, Z));
mlm_est = MatrixLMnet.MatrixLM.mlm(mlmdata, addXIntercept = flag_intercept, addZIntercept = false);

##########
# MLMnet #
##########

mlmdata = RawData(Response(Y), Predictors(X, Z));
mlmnet_est = mlmnet(
    mlmdata, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    isNaive = true,
    addXIntercept = flag_intercept, 
    addZIntercept = false, 
    isVerbose = false,
    thresh = 1e-16 
);

hcat(mlm_est.B, mlmnet_est.B)

