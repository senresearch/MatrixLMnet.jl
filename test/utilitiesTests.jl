###########
# Library #
###########
# using Random
using MatrixLMnet
using Helium
using Test

################
# Data loading #
################

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

rng = MatrixLMnet.Random.MersenneTwister(2021)


######################################################
# TEST 1: the dimension of results by util functions #
######################################################

est = mlmnet(dat, λ, α, method = "cd", addZIntercept = true, addXIntercept = true, isVerbose = true)

predicted = predict(est, est.data.predictors)

#Test the function predict
#println(size(predicted[:,:,1,1] ))

coef = MatrixLMnet.coef_3d(est)

#println(size(coef[:,:,1,1] ))

@testset "coef and predict" begin
    @test size(predicted[:,:,1,1] ) == (240, 2)
    @test size(coef[:,:,1,1] ) == (27, 3)
end


#################################
# TEST 2: test predict function #
#################################

newPredictors = Predictors(X, Z, false, false)
predicted = predict(est, newPredictors)
est2 = mlmnet(dat, λ, α, method = "cd", addZIntercept = false, addXIntercept = false, isVerbose = true)
newPredictors2 = Predictors(hcat(ones(size(X, 1)), X), hcat(ones(size(Z, 1)), Z), true, true)
predicted2 = predict(est2, newPredictors2)

@test size(predicted[:,:,1,1] ) == size(predicted2[:,:,1,1] )



#######################################
# TEST 3: test backtransform function #
#######################################

using MatrixLMnet: normalize!, mean, norm, mlmnet_test
using Distributions, Random
# using CSV, DataFrames


###################
#  Simulated Data #
###################
# Model: 𝐘 = 𝐗 𝛃 𝐙' + 𝜎𝜖, with 𝜖∼𝑁(0,1) 
rng = MersenneTwister(2024)

d = Normal(5.0, 1.0);
# Matrices dimensions
n = 240; m = 7; p = 9; q = 4;

# Simulate the coefficients matrix B
list_coefs = [0,1,1.5,2,2.5,3,.5]
B = rand(list_coefs, p, q)

# Simulate predictors
X = hcat(ones(n), rand(d, n, p-1));

# Simulate Z
list_Z_coefs = [0,1]
Z = hcat(ones(m), rand(list_Z_coefs, m, q-1))

# Simulate Y 
σ = 3;
Y = X*B*Z' + σ*rand(Normal(0, 1), n, m);

X = X[:, 2:end];
Z = Z[:, 2:end];

#############################################################################
# TEST 3-a test backtransform: addXIntercept = false, addZIntercept = false #
#############################################################################

mlm_data = RawData(Response(Y), Predictors(X, Z));
mlm_est = MatrixLMnet.MatrixLM.mlm(
    mlmdata, 
    addXIntercept = false, 
    addZIntercept = false
);

# MLMnet
# mlm_data = RawData(Response(Y), Predictors(X, Z));
mlmnet_est = mlmnet(
    mlm_data, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    toNormalize = true,
    isNaive = false,
    addXIntercept = false, 
    addZIntercept = false, 
    isVerbose = false,
    thresh = 1e-16 
);

println("Backtransform test  α=0 and λ=0 test 3-a: ", @test isapprox(mlm_est.B, mlmnet_est.B, atol = 1e-3))
hcat(mlm_est.B, mlmnet_est.B)

############################################################################
# TEST 3-b test backtransform: addXIntercept = true, addZIntercept = false #
############################################################################
mlmdata = RawData(Response(Y), Predictors(X, Z));
mlm_est = MatrixLMnet.MatrixLM.mlm(
    mlmdata, 
    addXIntercept = true, 
    addZIntercept = false
);

# MLMnet
# mlmdata = RawData(Response(Y), Predictors(X, Z));
mlmnet_est = mlmnet(
    mlm_data, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    toNormalize = true,
    isNaive = true,
    addXIntercept = true, 
    addZIntercept = false, 
    isVerbose = false,
    thresh = 1e-16 
);

println("Backtransform test  α=0 and λ=0 test 3-b: ", @test isapprox(mlm_est.B, mlmnet_est.B, atol = 1e-3))

############################################################################
# TEST 3-c test backtransform: addXIntercept = false, addZIntercept = true #
############################################################################
# mlmdata = RawData(Response(Y), Predictors(X, Z));
mlm_est = MatrixLMnet.MatrixLM.mlm(
    mlmdata, 
    addXIntercept = false, 
    addZIntercept = true
);

# MLMnet
# mlmdata = RawData(Response(Y), Predictors(X, Z));
mlmnet_est = mlmnet(
    mlm_data, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    toNormalize = true,
    isNaive = true,
    addXIntercept = false, 
    addZIntercept = true, 
    isVerbose = false,
    thresh = 1e-16 
);

println("Backtransform test  α=0 and λ=0 test 3-c: ", @test isapprox(mlm_est.B, mlmnet_est.B, atol = 1e-3))

###########################################################################
# TEST 3-d test backtransform: addXIntercept = true, addZIntercept = true #
###########################################################################
# mlmdata = RawData(Response(Y), Predictors(X, Z));
mlm_est = MatrixLMnet.MatrixLM.mlm(
    mlmdata, 
    addXIntercept = true, 
    addZIntercept = true
);

# MLMnet
# mlmdata = RawData(Response(Y), Predictors(X, Z));
mlmnet_est = mlmnet(
    mlm_data, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    toNormalize = true,
    isNaive = true,
    addXIntercept = true, 
    addZIntercept = true, 
    isVerbose = false,
    thresh = 1e-16 
);

println("Backtransform test  α=0 and λ=0 test 3-d: ", @test isapprox(mlm_est.B, mlmnet_est.B, atol = 1e-5))


println("Tests utilities finished!")
