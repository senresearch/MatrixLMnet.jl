###########
# Library #
###########
using MatrixLM
using Distributions, Random, Statistics, LinearAlgebra, StatsBase
using MatrixLMnet2
using DataFrames
using Random
using LinearAlgebra

####################
# External sources #
####################
include("../src/sim_helpers.jl")

p = 8; # Number of predictors
β1 = [3.5, 1.5, 0, 0, 2, 0, 0 ,0];
β2 = [0, 1.5, 0, 3.5, 2, 0, 0 , 2];
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

Z = 1.0*Matrix(I, 2, 2);

dat = RawData(Response(Y), Predictors(X, Z));

lambdasL1 = [10.0]
lambdasL2 = [0.0]
est1 = mlmnetNet(fistaNet!, dat, lambdasL1, lambdasL2, isZIntercept = false, isXIntercept = false)
est_B_Net = est1.B[1, 1, :, :]

################################################################################

# Generate predictors
X = simulateCorrelatedData(matCor, n);

# Generate response
Random.seed!(705)
Y1 = X*β1 + σ*rand(Normal(0, 1), n);
Y2 = X*β2 + σ*rand(Normal(0, 1), n);

Y = hcat(Y1, Y2);

Z = 1.0*Matrix(I, 2, 2);

dat = RawData(Response(Y), Predictors(X, Z));

est2 = mlmnet(fista!, dat, lambdasL1, isZIntercept = false, isXIntercept = false)
est_B_Lasso = est2.B[1, :, :]

@test est_B_Net == est_B_Lasso

