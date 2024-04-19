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

flag_intercept = true

# MLM
# mlmdata = RawData(Response(Y[:,2]|> x->reshape(x,:,1)), 
    # Predictors(X, Z[1,1]|> x ->reshape([x], :,1)));
mlmdata = RawData(Response(Y), Predictors(X, Z));
mlm_est = MatrixLMnet.MatrixLM.mlm(
    mlmdata, 
    addXIntercept = flag_intercept, 
    addZIntercept = false
);



# MLMnet
# mlmdata = RawData(Response(Y[:,2]|> x->reshape(x,:,1)), 
#     Predictors(X, Z[1,1]|> x ->reshape([x], :,1)));
mlmdata = RawData(Response(Y), Predictors(X, Z));
mlmnet_est = mlmnet(
    mlmdata, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    toNormalize = true,
    isNaive = false,
    addZIntercept = false, addXIntercept = flag_intercept, 
    toXInterceptReg = false,
    isVerbose = false,
    thresh = 1e-16 
);


mlmnet_est_test = mlmnet_test(
    mlmdata, 
    [0.0], [0.0], # lambda and alpha are set to 0
    method = "fista", stepsize = 0.01, 
    toNormalize = true,
    isNaive = false,
    addZIntercept = false, addXIntercept = flag_intercept, 
    toXInterceptReg = false,
    isVerbose = false,
    thresh = 1e-16 
);

# Centers and normalizes predictors
meansX, normsX, = normalize!(hcat(ones(240), copy(get_X(mlmdata))), true) 
meansZ, normsZ, = normalize!(copy(get_Z(mlmdata)), false)



B_t = copy(mlmnet_est_test.B)


hcat(mlm_est.B,  B_t, mlmnet_est.B,)


isXinterceptexist = true
isZinterceptexist = false

# reverse scale from X and Z normalization
B_t[:,:,1,1] = (B_t[:,:,1,1]./permutedims(normsX))./normsZ 

# Back transform the X intercepts (row main effects)
if isXinterceptexist == true
    # B̂intercept = Ȳ - X̄ B̂coef. Since X centered, B̂intercept|Xcentered = Ȳ      
    B_t[1,:,1,1] = B_t[1,:,1,1] - meansX[:,2:end]*B_t[2:end,:,1,1]
end

# Back transform the Z intercepts (column main effects)
if isZinterceptexist == true
    # B̂intercept = Ȳ - Z̄ B̂coef. Since X centered, B̂intercept|Xcentered = Ȳ      
    B_t[:,1,1,1] = B_t[:,1,1,1] - B_t[:,2:end,1,1]*permutedims(meansZ[:,2:end])
end




if isXinterceptexist == true
    prodX = (meansX[:,2:end]./normsX[:,2:end])*B_t[2:end,:,1,1]
    B_t[1,:,1,1] = (B_t[1,:,1,1]./normsX[:,1]) - vec(prodX)
end




hcat(mlm_est.B, mlmnet_est.B, B_t)


println("Tests utilities finished!")
