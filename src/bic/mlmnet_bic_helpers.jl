
"""
calc_bic(MLMNet)

Calculates BIC for each model according to the lambda-alpha 
pair parameter. 

# Arguments 

- MLMNets = Mlmnet object resulting from `mlmnet()` function.

# Value

2d array of floats with dimensions equal to the number of lambdas by the 
number of alphas. 

"""
function calc_bic(MLMNet::Mlmnet)

          resids = resid(MLMNet).^2 # resids squared
          n = size(resids, 1)
          m = size(resids,2)
          
          # Initialize array to store test MSEs 
          bic = Array{Float64}(undef, length(MLMNet.lambdas), length(MLMNet.alphas))

          for i in 1:length(MLMNet.lambdas), j in 1:length(MLMNet.alphas)
            # BIC for (lambdas i, alphas j)
            dM = sum(MLMNet.B[:,:,i,j] .!= 0.0);
            sse = sum(resids[:,:,i,j])
            bic[i, j] = sse + log(n*m).*dM.*sse/(n*m); 
          end 

        return bic
end



"""
        calc_mse(MLMNet)

Calculates test MSE for each pair of lambda-alpha. 

# Arguments 

- MLMNet = Mlmnet object

# Value

Matrix of floats with dimensions equal to the number of lambdas by the number of alphas.
 

"""
function calc_mse(MLMNet::Mlmnet)


# Initialize array to store test MSEs 
mse = Array{Float64}(undef, length(MLMNet.lambdas), length(MLMNet.alphas))

# Iterate through all folds and lambdas-alphas parameters to calculate test MSE

# Residuals for all lambdas and alphas at fold j
resids = resid(MLMNet).^2 
for i in 1:length(MLMNet.lambdas), j in 1:length(MLMNet.alphas)
    # MSE for (lambdas i, alphas j) 
    mse[i,j] = mean(resids[:,:,i,j]) 
end

return mse
end


"""
    calc_prop_zero(MLMNet; dig)

Calculates proportion of zero interaction coefficients for each of the CV 
folds for each lambda. 

# Arguments 

- MLMNet = Mlmnet object 

# Keyword arguments 

- dig = integer; digits of precision for zero coefficients. Defaults to 12. 

# Value

Matrix of floats with dimensions equal to the number of lambdas by the number of alphas. 

"""
function calc_prop_zero(MLMNet::Mlmnet; 
                        dig::Int64=12)


propZero = Array{Float64}(undef, length(MLMNet.lambdas), length(MLMNet.alphas))

# Boolean arrays used to subset coefficients for interactions only
xIdx = trues(MLMNet.data.p)
xIdx[1] = MLMNet.data.predictors.hasXIntercept==false
zIdx = trues(MLMNet.data.q)
zIdx[1] = MLMNet.data.predictors.hasZIntercept==false 

# Iterate through all folds and lambdas to calculate proportion of zero 
# interaction coefficients 

for i in 1:length(MLMNet.lambdas), j in 1:length(MLMNet.alphas)
    # Proportion of zero interaction coefficients for lambda i and 
    # fold j  
    propZero[i,j] = mean(round.(MLMNet.B[xIdx,zIdx,i,j], 
                        digits=dig) .== 0) 
end


return propZero
end