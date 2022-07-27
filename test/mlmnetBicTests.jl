###########
# Library #
###########
# using Random
using MatrixLMnet, Distributions, LinearAlgebra
using Helium
using Test

########################################
# TEST BIC Validation - Simulated Data #
########################################

#=
Description:
-----------

Model: 𝐘 = 𝐗 𝛃 𝐙' + 𝜎𝐄 with 𝐄∼𝑁(0,𝐈) 

- 𝐘ₙₘ is the response matrix
- 𝐗ₙₚ is the covariate matrix
- 𝐄ₙₘ is the error matrix

Here, the 𝐙 matrix is an identity matrix.
=#

# Dataset parameters
m = 3;                    # number of responses
n = 200;                  # number of observations
p = 100;                  # total number of features (i.e., covariates)
s = 10;                   # number of relevant features
β = 3.0;                  # value of coefficient
σ² = s.*(β^2)./9;         # variance of the error
μvecX = zeros(p);         # vector containing the mean observation of each p covariates
ΣmatX = I(p);             # covariance matrix (p,p) of the matrix covariates

# Random seed
rng = 2021

# Generate X covariate matrix
MatrixLMnet.Random.seed!(rng)
X = permutedims(rand(MvNormal(μvecX, ΣmatX), n));

# Generate Z matrix
Z = Matrix(I(m)).*1.0;

# Generate B matrix
B = zeros(p, m); 
B[1:s,:] .= β
if size(B)[2] > 2
    funShift(vX) = circshift(vX, sample(2:(p-s), 1)[1])# use shuffle for unblocked relevant covariates
    B[:,2:end] .= mapslices(funShift, B[:,2:end], dims = [1])
    B[:,end].= zeros(p)
else 
    if size(B)[2] == 2
        B[:,end].= zeros(p)
    end
end;

# Generate E error matrix
MatrixLMnet.Random.seed!(rng);
E = permutedims(rand(MvNormal(zeros(m), σ² .*I(m)), n));

# Generate Y response matrix
Y = X*B*Z + E;
meanY, normY = MatrixLMnet.normalize!(Y, false);

# Build raw data object from MatrixLM.jl
dat = RawData(Response(Y), Predictors(X, Z));

# Hyper parameters
λ = 1.2.^[5, 1, -5];
α = [1.0, 0.5, 0.0];


numVersion = VERSION
if Int(numVersion.minor) < 7
      tolVersion=2e-1
else
      tolVersion=1e-6
end 

####################################
# TEST BIC Validation - Estimation #
####################################

est = mlmnet(dat, λ, α; method = "fista_bt", hasXIntercept = false, hasZIntercept=false, isVerbose = false);

#############################
# TEST BIC Validation - BIC #
#############################

est_BIC =  mlmnet_bic(dat, λ, α; method = "fista_bt", hasXIntercept = false, hasZIntercept=false, isVerbose = false);

df_BIC = mlmnet_bic_summary(est_BIC);

#################################################
# TEST BIC Validation - BIC loglikelihood based #
#################################################

resids = resid(est).^2; # resids squared
          
# Initialize array to store test MSEs 
BIC2 = Array{Float64}(undef, length(est.lambdas), length(est.alphas));

for i in 1:length(est.lambdas), j in 1:length(est.alphas)
      # BIC for (lambdas i, alphas j)
      k = sum(est.B[:,:,i,j] .!= 0.0, dims = 1) .+ m;
      
      distResids = MvNormal(zeros(m), (sqrt.(sum(resids[:,:,i,j], dims = 1)./n))[:]);
      L̂ = loglikelihood(distResids, permutedims(resids[:,:,i,j]))
      BIC2[i,j] = sum(k)*log(n) - 2*(L̂)
end; 

df_BIC.BIC2 = vec(BIC2);

##############################
# TEST BIC Validation - test #
##############################

df1 = filter(row -> row.BIC == minimum(df_BIC.BIC), df_BIC );

df2 = filter(row -> row.BIC2 == minimum(df_BIC.BIC2), df_BIC ;)

println("BIC validation test 1:",
         @test df1 == df2;)
