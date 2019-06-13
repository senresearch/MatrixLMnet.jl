"""
    calc_avg_mse(MLMNet_cv)
	
Calculates average test MSE across folds. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

1d array of floats
	
"""
function calc_avg_mse(MLMNet_cv::Mlmnet_cv) 
    
    return Statistics.mean(MLMNet_cv.mse, dims=2)[:,1]
end


"""
    lambda_min(MLMNet_cv)
	
Returns lambda corresponding to the minimum average test MSE across folds. 

# Arguments 

- MLMNet_cv = MLMNet_cv object

# Value

Float
	
"""
function lambda_min(MLMNet_cv::Mlmnet_cv)

    # Calculate average test MSE across folds.
    avgMse = calc_avg_mse(MLMNet_cv)
    # Find index of minimum average test MSE
    idx = indmin(avgMse)
    
    # Return corresponding lambda
    return MLMNet_cv.lambdas[idx]
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
    
    return Statistics.mean(MLMNet_cv.propZero, dims=2)[:,1]
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