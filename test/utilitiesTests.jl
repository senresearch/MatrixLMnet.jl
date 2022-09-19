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


###################################################
#Test the dimension of results by util functions #
###################################################

est = mlmnet(dat, λ, α, method = "cd", hasZIntercept = true, hasXIntercept = true, isVerbose = true)

predicted = predict(est, est.data.predictors)

#Test the function predict
#println(size(predicted[:,:,1,1] ))

coef = MatrixLMnet.coef_3d(est)

#println(size(coef[:,:,1,1] ))

@testset "coef and predict" begin
    @test size(predicted[:,:,1,1] ) == (240, 2)
    @test size(coef[:,:,1,1] ) == (27, 3)
end


################################
#Test2: test predict function ##
################################

newPredictors = Predictors(X, Z, false, false)
predicted = predict(est, newPredictors)
est2 = mlmnet(dat, λ, α, method = "cd", hasZIntercept = false, hasXIntercept = false, isVerbose = true)
newPredictors2 = Predictors(hcat(ones(size(X, 1)), X), hcat(ones(size(Z, 1)), Z), true, true)
predicted2 = predict(est2, newPredictors2)

@test size(predicted[:,:,1,1] ) == size(predicted2[:,:,1,1] )