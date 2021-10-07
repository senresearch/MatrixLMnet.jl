###########
# Library #
###########
using MatrixLM
using Distributions, Random, Statistics, LinearAlgebra, StatsBase
using MatrixLMnet2
using DataFrames
using Random
using LinearAlgebra
using Test

####################
# External sources #
####################
include("../src/sim_helpers.jl")

###########################
# Generate Simulated Data #
###########################

#=
Description:
-----------

Model: 𝑌 = 𝐗𝛽𝐙 + 𝜎𝜖, with 𝜖∼𝑁(0,1)

Simulate data set consisting  of 20/20/200 observations and 8 predictors.
We let 𝛽₁ = (3, 1.5, 0, 0, 2, 0, 0, 0), 𝛽₂ = (0, 1.5, 0, 3.5, 2, 0, 0 , 2) where
𝛽 = [𝛽₁, 𝛽₂] and 𝜎 = 3.
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
Random.seed!(705)
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


# Elastic net penalized regression
est1 = mlmnetNet(fistaNet!, dat, λ, α, isZIntercept = false, isXIntercept = false)
est_B_Net = est1.B[:, :, 3, 1]

# Lasso penalized regression
est2 = mlmnet(fista!, dat, λ, isZIntercept = false, isXIntercept = false)
est_B_Lasso = est2.B[:, :, 3]

@test est_B_Net == est_B_Lasso

