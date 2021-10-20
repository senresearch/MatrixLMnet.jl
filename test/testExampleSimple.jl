using MatrixLM
using MatrixLMnet2
using DataFrames
using Random
using LinearAlgebra
using Test

# Dimensions of matrices 
n = 100
m = 250
# Number of column covariates
q = 20

# Randomly generate an X matrix of row covariates with 2 categorical variables
# and 4 continuous variables
Random.seed!(1)
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n)), 
            DataFrame(rand(n,4)))
# Use the contr function to get contrasts for the two categorical variables 
# (treatment contrasts for catvar1 and sum contrasts for catvar2).
# contr returns a DataFrame, so X needs to be converted into a 2d array.
X = convert(Array{Float64,2}, contr(X_df, [:catvar1, :catvar2], 
                                    ["treat", "sum"]))
# Number of row covariates
p = size(X)[2]

# Randomly generate some data for column covariates Z and response variable Y
Z = rand(m,q)
B = rand(vcat(-5:5, zeros(19)),p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E

# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z));

lambdasL1 = [10.0, 5.0]
lambdasL2 = [0.0]

# Random.seed!(705)
est = mlmnetNet(fistaNet!, dat, lambdasL1, lambdasL2, hasZIntercept = false, hasXIntercept = false, 
                setStepsize = true);
est_Net = est.B[2, 1, :, :]

# Random.seed!(705)
est_2 = mlmnet(fista!, dat, lambdasL1, hasZIntercept = false, hasXIntercept = false, 
                setStepsize = true);
est_Lasso = est_2.B[2, :, :]

# # Dimensions of matrices 
# n = 100
# m = 250
# # Number of column covariates
# q = 20

# # Randomly generate an X matrix of row covariates with 2 categorical variables
# # and 4 continuous variables
# Random.seed!(1)
# X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n)), 
#             DataFrame(rand(n,4)))
# # Use the contr function to get contrasts for the two categorical variables 
# # (treatment contrasts for catvar1 and sum contrasts for catvar2).
# # contr returns a DataFrame, so X needs to be converted into a 2d array.
# X = convert(Array{Float64,2}, contr(X_df, [:catvar1, :catvar2], 
#                                     ["treat", "sum"]))
# # Number of row covariates
# p = size(X)[2]

# # Randomly generate some data for column covariates Z and response variable Y
# Z = rand(m,q)
# B = rand(vcat(-5:5, zeros(19)),p,q)
# E = randn(n,m)
# Y = X*B*transpose(Z)+E

# # Construct a RawData object
# dat = RawData(Response(Y), Predictors(X, Z));

# lambdasL1 = [10.0, 5.0]
# lambdasL2 = [0.0]

# est_2 = mlmnet(fista!, dat, lambdasL1, hasZIntercept = false, hasXIntercept = false);
# est_Lasso = est_2.B[2, :, :]

@test est_Lasso == est_Net