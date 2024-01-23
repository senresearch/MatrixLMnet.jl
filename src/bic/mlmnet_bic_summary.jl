


"""
    mlmnet_bic_summary(MLMNet_bic::Mlmnet_bic)
	
Summarizes results of BIC-validation by returning a table with: 

- Lambdas used
- MSE across folds for each lambda
- Proportion of zero interaction coefficients across each pair 
  of lambda and alpha

# Arguments 

- MLMNet_bic = Mlmnet_bic object

# Value

DataFrame summarizing BIC, MSE, proportion of zero interactions across each pair 
of lambda and alpha. 
	
"""
function mlmnet_bic_summary(MLMNet_bic::Mlmnet_bic)
    
    # Calculate summary information across folds

    lenVecLambdas = length(MLMNet_bic.MLMNet.lambdas)
    lenVecAlphas = length(MLMNet_bic.MLMNet.alphas)

    λαIndex =  map((x,y) -> (x,y), 
            repeat(collect(1:lenVecLambdas), lenVecAlphas),
            vec(repeat(permutedims(collect(1:lenVecAlphas)), lenVecLambdas)) )

    # MSE column
    # Matrix of  MSEs w.r.p.t each (lambda, alpha)
    vMSE = vec(MLMNet_bic.mse);
    
    # Proportion Zero column
    # Matrix of zero proportions w.r.p.t each (lambda, alpha)
    vPropZero = vec(MLMNet_bic.propZero);

    # BIC column
    # Matrix of BIC w.r.p.t each (lambda, alpha)
    vBIC = vec(MLMNet_bic.bic);
    
    out_df = DataFrame(𝜆_𝛼_Index = λαIndex,
                       Lambda = repeat(MLMNet_bic.MLMNet.lambdas, lenVecAlphas),
                       Alpha =  vec(repeat(permutedims(MLMNet_bic.MLMNet.alphas), lenVecLambdas)),
                       MSE = vMSE,
                       PropZero =  vPropZero,
                       BIC = vBIC);
        
    return out_df

end


# """
#     lambda_min(MLMNet_cv)
    
# Returns summary information for lambdas corresponding to the minimum average 
# test MSE across folds and the MSE one that is standard error greater. 

# # Arguments 

# - MLMNet_cv = MLMNet_cv object

# # Value

# DataFrame from mlmnet_cv_summary restricted to the lambdas and alphas that correspond to 
# the minimum average test MSE across folds and the MSE that is one standard 
# error greater. 
    
# """
# function lambda_min(MLMNet_cv::Mlmnet_cv)
#     # Calculate average proportion of zeros
#     prop_zeroMean = calc_avg_prop_zero(MLMNet_cv)
    
#     # Calculate average test MSE across folds.
#     mseMean = calc_avg_mse(MLMNet_cv)
#     mseStd = valid_reduce2(MLMNet_cv.mse, std)
    
#     # Find index of minimum average test MSE
#     minIdx = argmin(mseMean)[1]
#     minIdy = argmin(mseMean)[2]
    
#     # Compute standard error across folds for the minimum MSE
#     mse1StdErr = mseMean[minIdx, minIdy] + mseStd[minIdx, minIdy]
#     # Find the index of the lambda that is closest to being 1 SE greater than 
#     # the lowest lambda, in the direction of the bigger lambdas
#     min1StdErrIdx = argmin(abs.(mseMean[1:minIdx[1], 1:minIdy[1]].-mse1StdErr))[1]
#     min1StdErrIdy = argmin(abs.(mseMean[1:minIdx[1], 1:minIdy[1]].-mse1StdErr))[2]
    
#     # Pull out summary information for these two lambdas
#     out = hcat(MLMNet_cv.lambdas[minIdx], MLMNet_cv.alphas[minIdy], 
#                mseMean[minIdx, minIdy], prop_zeroMean[minIdx, minIdy])
#     out2 = hcat(MLMNet_cv.lambdas[min1StdErrIdx], MLMNet_cv.alphas[min1StdErrIdy], 
#                 mseMean[min1StdErrIdx, min1StdErrIdy], 
#                 prop_zeroMean[min1StdErrIdx, min1StdErrIdy])
#     out = DataFrame(vcat(out, out2), :auto)

#     colnames = ["Lambda", "Alpha", "AvgMSE", "AvgPropZero"];
#     rename!(out, Symbol.(colnames))

#     # Add names first column
#     insertcols!(out, 1, :Name => ["(𝜆, 𝛼)_min", "(𝜆, 𝛼)_min1se"])
#     # Add indices as second column
#     insertcols!(out, 2, :Index => [Tuple([minIdx, minIdy]), Tuple([min1StdErrIdx, min1StdErrIdy])])
    
#     return out
# end