"""
    valid_reduce2(A::Array{Float64,3}, fun::Function=mean)
    
Reduce a 2d matrix across its columns using a given function, but ignoring 
NaN, Inf, and -Inf. 

# Arguments 

- A = 2d array of floats
- fun = function with which to reduce across the columns of A

# Value

2d array of floats
    
"""
function valid_reduce2(A::Array{Float64,3}, fun::Function=mean)
    
    # Initialize array for storing output 
    out = Array{Float64, 2}(undef, size(A, 1), size(A, 2))
    # Iterate through rows of A and reduce across columns
    for i in 1:size(A, 1), j in 1:size(A, 2)
        # Note that the following line also drops NaNs, which are not numbers
        out[i, j] = fun(A[i, j, (A[i, j, :] .< Inf) .& (A[i, j, :] .> -Inf)])
    end
    return out
end


"""
    calc_avg_mse(MLMNet_cv::Mlmnet_cv) 
	
Calculates average test MSE across folds. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

2d array of floats
	
"""
function calc_avg_mse(MLMNet_cv::Mlmnet_cv) 
    
    # return valid_reduce2(MLMNet_cv.mse, mean)[:, :, 1]
    return valid_reduce2(MLMNet_cv.mse, mean)
end


"""
    calc_avg_prop_zero(MLMNet_cv::Mlmnet_cv) 
	
Calculates average proportion of zero interaction coefficients across folds. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

1d array of floats
	
"""
function calc_avg_prop_zero(MLMNet_cv::Mlmnet_cv)
    
    # return valid_reduce2(MLMNet_cv.propZero, mean)[:,:,1]
    return valid_reduce2(MLMNet_cv.propZero, mean)
end


"""
    mlmnet_cv_summary(MLMNet_cv::Mlmnet_cv) 
	
Summarizes results of cross-validation by returning a table with: 

- Lambdas used
- Average MSE across folds for each lambda
- Average proportion of zero interaction coefficeints across folds for each 
  lambda

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

DataFrame summarizing average MSE and proportion of zero interactions across 
folds for each lambda. 
	
"""
function mlmnet_cv_summary(MLMNet_cv::Mlmnet_cv)
    
    # Calculate summary information across folds

    lenVecLambdas = length(MLMNet_cv.lambdas)
    lenVecAlphas = length(MLMNet_cv.alphas)

    λαIndex =  map((x,y) -> (x,y), 
            repeat(collect(1:lenVecLambdas), lenVecAlphas),
            vec(repeat(permutedims(collect(1:lenVecAlphas)), lenVecLambdas)) )

    # Second column
    # a 2-d matrix of average MSEs w.r.p.t each (l1, l2)
    avg_mse = vec(calc_avg_mse(MLMNet_cv))
    
    # Third column
    mseStd = vec(valid_reduce2(MLMNet_cv.mse, std))

    # Fourth column
    # a 2-d matrix of average zero proportions w.r.p.t each (l1, l2)
    avg_prop_zero = vec(calc_avg_prop_zero(MLMNet_cv))
    
    out_df = DataFrame(𝜆_𝛼_Index = λαIndex,
                       Lambda = repeat(MLMNet_cv.lambdas, lenVecAlphas),
                       Alpha =  vec(repeat(permutedims(MLMNet_cv.alphas), lenVecLambdas)),
                       AvgMSE = avg_mse,
                       StdMSE = mseStd,
                       AvgPropZero =  avg_prop_zero);
        
    return out_df
end


"""
    lambda_min(MLMNet_cv::Mlmnet_cv)
    
Returns summary information for lambdas corresponding to the minimum average 
test MSE across folds and the MSE one that is standard error greater. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

DataFrame from mlmnet_cv_summary restricted to the lambdas and alphas that correspond to 
the minimum average test MSE across folds and the MSE that is one standard 
error greater. 
    
"""
function lambda_min(MLMNet_cv::Mlmnet_cv)
    # Calculate average proportion of zeros
    prop_zeroMean = calc_avg_prop_zero(MLMNet_cv)
    
    # Calculate average test MSE across folds.
    mseMean = calc_avg_mse(MLMNet_cv)
    mseStd = valid_reduce2(MLMNet_cv.mse, std)
    
    # Find index of minimum average test MSE
    minIdx = argmin(mseMean)[1]
    minIdy = argmin(mseMean)[2]
    
    # Compute standard error across folds for the minimum MSE
    mse1StdErr = mseMean[minIdx, minIdy] + mseStd[minIdx, minIdy]
    # Find the index of the lambda that is closest to being 1 SE greater than 
    # the lowest lambda, in the direction of the bigger lambdas
    min1StdErrIdx = argmin(abs.(mseMean[1:minIdx[1], 1:minIdy[1]].-mse1StdErr))[1]
    min1StdErrIdy = argmin(abs.(mseMean[1:minIdx[1], 1:minIdy[1]].-mse1StdErr))[2]
    
    # Pull out summary information for these two lambdas
    out = hcat(MLMNet_cv.lambdas[minIdx], MLMNet_cv.alphas[minIdy], 
               mseMean[minIdx, minIdy], prop_zeroMean[minIdx, minIdy])
    out2 = hcat(MLMNet_cv.lambdas[min1StdErrIdx], MLMNet_cv.alphas[min1StdErrIdy], 
                mseMean[min1StdErrIdx, min1StdErrIdy], 
                prop_zeroMean[min1StdErrIdx, min1StdErrIdy])
    out = DataFrame(vcat(out, out2), :auto)

    colnames = ["Lambda", "Alpha", "AvgMSE", "AvgPropZero"];
    rename!(out, Symbol.(colnames))

    # Add names first column
    insertcols!(out, 1, :Name => ["(𝜆, 𝛼)_min", "(𝜆, 𝛼)_min1se"])
    # Add indices as second column
    insertcols!(out, 2, :Index => [Tuple([minIdx, minIdy]), Tuple([min1StdErrIdx, min1StdErrIdy])])
    
    return out
end