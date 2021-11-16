###########
# Library #
###########
using MatrixLM
using Distributions, Random, Statistics, LinearAlgebra, StatsBase
using MatrixLMnet2
using DataFrames
using LinearAlgebra
using Test
using BenchmarkTools

####################
# External sources #
####################
include("sim_helpers.jl")


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

# Simulation parameters
p = 8; # Number of predictors
β1 = [3.5, 1.5, 0,   0, 2, 0, 0 , 0];
β2 = [  0, 1.5, 0, 3.5, 2, 0, 0 , 2];
σ = 3;
n = 240;

# Generate correlation matrix 
matCor = zeros(p,p)
for j = 1:p, i = 1:p
    matCor[i,j] = 0.5^abs(i-j)
end

# Generate predictors
X = simulateCorrelatedData(matCor, n);

# Generate response
Y1 = X*β1 + σ*rand(Normal(0, 1), n);
Y2 = X*β2 + σ*rand(Normal(0, 1), n);
Y = hcat(Y1, Y2);

# Generate Z matrix
Z = 1.0*Matrix(I, 2, 2);

# Build raw data object from MatrixLM.jl
dat = RawData(Response(Y), Predictors(X, Z));

# Hyper parameters
λ = [10.0, 5.0, 3.0]
α = [1.0]

#############################################
# TEST 1 Lasso vs Elastic Net (𝛼=1) - ista #
#############################################


# # Elastic net penalized regression
Random.seed!(2021)
est1 = mlmnet_cv(dat, λ, α, 10, 1, method = "ista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = lambdaNet_min(est1);

# # Elastic net penalized regression
Random.seed!(2021)
est3 = mlmnet_cv(dat, λ, 10, 1, method = "ista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net3 = lambdaNet_min(est3);

# Lasso penalized regression
Random.seed!(2021)
est2 = mlmnet_cv(ista!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Lasso = lambda_min(est2);



println("CV Lasso vs Elastic Net when α=1 test 1 - ista: ", @test smmr_Net3.AvgMSE == smmr_Lasso.AvgMSE && 
                                                            smmr_Net3.AvgPercentZero == smmr_Lasso.AvgPercentZero)

@btime  mlmnet_cv(dat, λ, α, 10, 1, method = "ista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(dat, λ, 10, 1, method = "ista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(ista!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);

#############################################
# TEST 2 Lasso vs Elastic Net (𝛼=1) - fista #
#############################################


# Elastic net penalized regression
Random.seed!(2021)
est1 = mlmnet_cv(dat, λ, α, 10, 1, method = "fista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = lambdaNet_min(est1);

# Elastic net penalized regression
Random.seed!(2021)
est3 = mlmnet_cv(dat, λ, 10, 1, method = "fista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net3 = lambdaNet_min(est3);

# Lasso penalized regression
Random.seed!(2021)
est2 = mlmnet_cv(fista!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Lasso = lambda_min(est2);



println("CV Lasso vs Elastic Net when α=1 test 2 - fista: ", @test smmr_Net3.AvgMSE == smmr_Lasso.AvgMSE && 
                                                            smmr_Net3.AvgPercentZero == smmr_Lasso.AvgPercentZero)

@btime  mlmnet_cv(dat, λ, α, 10, 1, method = "fista", hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(dat, λ, 10, 1, method = "fista",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(fista!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);

##########################################################
# TEST 3 Lasso vs Elastic Net (𝛼=1) - fista backtracking #
##########################################################


# Elastic net penalized regression
Random.seed!(2021)
est1 = mlmnet_cv(dat, λ, α, 10, 1, method = "fista_bt", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = lambdaNet_min(est1);

# Elastic net penalized regression
Random.seed!(2021)
est3 = mlmnet_cv(dat, λ, 10, 1, method = "fista_bt",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net3 = lambdaNet_min(est3);

# Lasso penalized regression
Random.seed!(2021)
est2 = mlmnet_cv(fista_bt!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Lasso = lambda_min(est2);



println("CV Lasso vs Elastic Net when α=1 test 2 - fista_bt: ", @test smmr_Net3.AvgMSE == smmr_Lasso.AvgMSE && 
                                                            smmr_Net3.AvgPercentZero == smmr_Lasso.AvgPercentZero)

@btime  mlmnet_cv(dat, λ, α, 10, 1, method = "fista_bt", hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(dat, λ, 10, 1, method = "fista_bt",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(fista_bt!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);


############################################
# TEST 4 Lasso vs Elastic Net (𝛼=1) - admm #
############################################


# Elastic net penalized regression
Random.seed!(2021)
est1 = mlmnet_cv(dat, λ, α, 10, 1, method = "admm", hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net1 = lambdaNet_min(est1);

# Elastic net penalized regression
Random.seed!(2021)
est3 = mlmnet_cv(dat, λ, 10, 1, method = "admm",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Net3 = lambdaNet_min(est3);

# Lasso penalized regression
Random.seed!(2021)
est2 = mlmnet_cv(admm!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);
smmr_Lasso = lambda_min(est2);



println("CV Lasso vs Elastic Net when α=1 test 2 - admm: ", @test smmr_Net3.AvgMSE == smmr_Lasso.AvgMSE && 
                                                            smmr_Net3.AvgPercentZero == smmr_Lasso.AvgPercentZero)

@btime  mlmnet_cv(dat, λ, α, 10, 1, method = "admm", hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(dat, λ, 10, 1, method = "admm",  hasZIntercept = false, hasXIntercept = false, isVerbose = false);

@btime  mlmnet_cv(admm!, dat, λ, 10, 1, hasZIntercept = false, hasXIntercept = false, isVerbose = false);

println("Tests finished!")

