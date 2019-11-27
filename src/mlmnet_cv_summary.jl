"""
    valid_reduce2(A, fun)
    
Reduce a 2d matrix across its columns using a given function, but ignoring 
NaN, Inf, and -Inf. 

# Arguments 

- A = 2d array of floats
- fun = function with which to reduce across the columns of A

# Value

1d array of floats
    
"""
function valid_reduce2(A::Array{Float64,2}, fun::Function=mean)
    
    # Initialize array for storing output 
    out = Array{Float64}(undef, size(A, 1))
    # Iterate through rows of A and reduce across columns
    for i in 1:size(A, 1)
        # Note that the following line also drops NaNs, which are not numbers
        out[i] = fun(A[i, (A[i, :] .< Inf) .& (A[i, :] .> -Inf)])
    end
    return out
end


"""
    calc_avg_mse(MLMNet_cv)
	
Calculates average test MSE across folds. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

1d array of floats
	
"""
function calc_avg_mse(MLMNet_cv::Mlmnet_cv) 
    
    return valid_reduce2(MLMNet_cv.mse, mean)[:,1]
end


"""
    calc_avg_prop_zero(MLMNet_cv)
	
Calculates average proportion of zero interaction coefficients across folds. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

1d array of floats
	
"""
function calc_avg_prop_zero(MLMNet_cv::Mlmnet_cv)
    
    return valid_reduce2(MLMNet_cv.propZero, mean)[:,1]
end


"""
    mlmnet_cv_summary(MLMNet_cv)
	
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
    out_df = DataFrame(hcat(MLMNet_cv.lambdas, calc_avg_mse(MLMNet_cv), 
                            calc_avg_prop_zero(MLMNet_cv)))
    # Useful names
    names!(out_df, map(Meta.parse, ["Lambda", "AvgMSE", "AvgPercentZero"]))
    
    return out_df
end


"""
    lambda_min(MLMNet_cv)
    
Returns summary information for lambdas corresponding to the minimum average 
test MSE across folds and the MSE one that is standard error greater. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

DataFrame from mlmnet_cv_summary restricted to the lambdas that correspond to 
the minimum average test MSE across folds and the MSE that is one standard 
error greater. 
    
"""
function lambda_min(MLMNet_cv::Mlmnet_cv)
    
    # Calculate average test MSE across folds.
    mseMean = calc_avg_mse(MLMNet_cv)
    mseStd = valid_reduce2(MLMNet_cv.mse, std)
    
    # Find index of minimum average test MSE
    minIdx = argmin(mseMean)
    
    # Compute standard error across folds for the minimum MSE
    mse1StdErr = mseMean[minIdx] + mseStd[minIdx]
    # Find the index of the lambda that is closest to being 1 SE greater than 
    # the lowest lambda, in the direction of the bigger lambdas
    min1StdErrIdx = argmin(abs.(mseMean[1:minIdx[1]].-mse1StdErr))
    
    # Pull out summary information for these two lambdas
    out = mlmnet_cv_summary(MLMNet_cv)[[minIdx,min1StdErrIdx],:]
    # Add names
    insertcols!(out, 1, :Name => ["lambda_min", "lambda_min1se"])
    # Add indices
    insertcols!(out, 1, :Index => [minIdx, min1StdErrIdx])
    return out
end
