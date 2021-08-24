"""
    coef(MLMNet)

Extract all coefficients from Mlmnet object

# Arguments 

- MLMNet = Mlmnet object

# Value

3d array of coefficients 

"""
function coef(MLMNet::Mlmnet)
    
    return MLMNet.B
end


"""
    coef(MLMNet, lambda)

Extract coefficients from Mlmnet object at a given lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar

# Value

2d array of coefficients 

"""
function coef(MLMNet::Mlmnet, lambda::Float64)
    
    # Find the index corresponding to the lambda of interest
    idx = findall([isapprox(lam, lambda) for lam in MLMNet.lambdas]) 
    if (length(idx) == 0)
        error("This lambda was not used in the MLMNet object.")
    end
    
    # Return the coefficient slice for the lambda of interest
    return MLMNet.B[idx[1],:,:]
end


"""
    coef_2d(MLMNet)

Extract coefficients from Mlmnet object as a flattened 2d array

# Arguments 

- MLMNet = Mlmnet object

# Value

2d array of flattened coefficients, where each column corresponds to a 
different lambda

"""
function coef_2d(MLMNet::Mlmnet)
    
    # Extract coefficients
    coeffs = coef(MLMNet)
    
    # Initialize 2d array to store flattened coefficients
    coeffs2d = Array{Float64}(undef, size(coeffs,2)*size(coeffs,3), 
                              size(coeffs,1))
    # Iterate through each coefficient slice
    for i in 1:size(coeffs,1)
        coeffs2d[:,i] = vec(coeffs[i,:,:])
    end
    
    return coeffs2d
end


"""
    predict(MLMNet, lambda, newPredictors)

Calculate new predictions based on Mlmnet object and given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- newPredictors = Predictors object. Defaults to the data.predictors field 
  in the MLM object used to fit the model. 

# Value

2d array of predicted values

"""
function predict(MLMNet::Mlmnet, lambda::Float64, 
                 newPredictors::Predictors=MLMNet.data.predictors)
    
    # Include X and Z intercepts in new predictors if necessary
    if MLMNet.data.predictors.isXIntercept==true && 
       newPredictors.isXIntercept==false
        newPredictors.X = add_intercept(newPredictors.X)
        newPredictors.isXIntercept = true
        println("Adding X intercept to newPredictors.")
  	end
    if MLMNet.data.predictors.isZIntercept==true && 
       newPredictors.isZIntercept==false
        newPredictors.Z = add_intercept(newPredictors.Z)
        newPredictors.isZIntercept = true
        println("Adding Z intercept to newPredictors.")
  	end

    # Remove X and Z intercepts in new predictors if necessary
    if MLMNet.data.predictors.isXIntercept==false && 
       newPredictors.isXIntercept==true
        newPredictors.X = remove_intercept(newPredictors.X)
        newPredictors.isXIntercept = false
        println("Removing X intercept from newPredictors.")
    end
    if MLMNet.data.predictors.isZIntercept==false && 
       newPredictors.isZIntercept==true
        newPredictors.Z = remove_intercept(newPredictors.Z)
        newPredictors.isZIntercept = false
        println("Removing Z intercept from newPredictors.")
    end
    
    # Calculate new predictions
    return calc_preds(newPredictors.X, newPredictors.Z, coef(MLMNet, lambda))
end 


"""
    predict(MLMNet, newPredictors)

Calculate new predictions based on Mlmnet object

# Arguments 

- MLMNet = Mlmnet object
- newPredictors = Predictors object. Defaults to the data.predictors field 
  in the MLM object used to fit the model. 

# Value

3d array of predicted values

"""
function predict(MLMNet::Mlmnet, 
                 newPredictors::Predictors=MLMNet.data.predictors)
    
    # Initialize 3d array for storing predictions
    all_preds = Array{Float64}(undef, length(MLMNet.lambdas), 
                               size(newPredictors.X,1), 
                               size(newPredictors.Z,1))
	# Calculate new predictions for each lambda
    for i = 1:length(MLMNet.lambdas)
        all_preds[i,:,:] = predict(MLMNet, MLMNet.lambdas[i], newPredictors)
    end
    
    return all_preds
end 


"""
    fitted(MLMNet, lambda)

Calculate fitted values of an Mlmnet object, given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar

# Value

2d array of fitted values

"""
function fitted(MLMNet::Mlmnet, lambda::Float64)
    
    return predict(MLMNet, lambda)
end


"""
    fitted(MLMNet)

Calculate fitted values of an Mlmnet object

# Arguments 

- MLMNet = Mlmnet object

# Value

3d array of fitted values

"""
function fitted(MLMNet::Mlmnet)
    
    return predict(MLMNet)
end


"""
    resid(MLMNet, lambda, newData)

Calculate residuals of an MLMNet object, given a lambda 

# Arguments 

- MLM = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- newData = RawData object. Defaults to the data field in the MLM object 
  used to fit the model. 

# Value

2d array of residuals

"""
function resid(MLMNet::Mlmnet, lambda::Float64, newData::RawData=MLMNet.data)
    
    # Include X and Z intercepts in new data if necessary
    if MLMNet.data.predictors.isXIntercept==true && 
       newData.predictors.isXIntercept==false
        newData.predictors.X = add_intercept(newData.predictors.X)
        newData.predictors.isXIntercept = true
        newData.p = newData.p + 1
        println("Adding X intercept to newData.")
    end
    if MLMNet.data.predictors.isZIntercept==true && 
       newData.predictors.isZIntercept==false
        newData.predictors.Z = add_intercept(newData.predictors.Z)
        newData.predictors.isZIntercept = true
        newData.q = newData.q + 1
        println("Adding Z intercept to newData.")
    end
    
    # Remove X and Z intercepts in new data if necessary
    if MLMNet.data.predictors.isXIntercept==false && 
       newData.predictors.isXIntercept==true
        newData.predictors.X = remove_intercept(newData.predictors.X)
        newData.predictors.isXIntercept = false
        newData.p = newData.p - 1
        println("Removing X intercept from newData.")
    end
    if MLMNet.data.predictors.isZIntercept==false && 
       newData.predictors.isZIntercept==true
        newData.predictors.Z = remove_intercept(newData.predictors.Z)
        newData.predictors.isZIntercept = false
        newData.q = newData.q - 1
        println("Removing Z intercept from newData.")
    end
    
    # Calculate residuals
    return calc_resid(get_X(newData), get_Y(newData), get_Z(newData), 
                      coef(MLMNet, lambda))
end


"""
    resid(MLMNet, newData)

Calculate residuals of an MLMNet object

# Arguments 

- MLMNet = Mlmnet object
- newData = RawData object. Defaults to the data field in the MLM object 
  used to fit the model. 

# Value

3d array of residuals

"""
function resid(MLMNet::Mlmnet, newData::RawData=MLMNet.data)
    
    # Initialize 3d array for storing residuals
    all_resid = Array{Float64}(undef, length(MLMNet.lambdas), 
                               newData.n, newData.m)
    
    # Calculate residuals for each lambda
    for i = 1:length(MLMNet.lambdas)
        all_resid[i,:,:] = resid(MLMNet, MLMNet.lambdas[i], newData)
    end
    
    return all_resid
end

############ Elastic Net ###############
"""
    coef(MLMNet)

Extract all coefficients from Mlmnet object

# Arguments 

- MLMNet = Mlmnet object

# Value

3d array of coefficients 

"""
function coef(MLMNet::MlmnetNet)
    
    return MLMNet.B
end


"""
    coef(MLMNet, lambda)

Extract coefficients from Mlmnet object at a given lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar

# Value

2d array of coefficients 

"""
function coef(MLMNet::MlmnetNet, lambdaL1::Float64, lambdaL2::Float64)
    
    # Find the index corresponding to the lambda of interest
    idx = findall([isapprox(lam1, lambdaL1) for lam1 in MLMNet.lambdasL1])
    idy = findall([isapprox(lam2, lambdaL2) for lam2 in MLMNet.lambdasL2])
    if (length(idx) == 0)
        error("This lambdaL1 was not used in the MLMNet object.")
    end

    if (length(idy) == 0)
        error("This lambdaL2 was not used in the MLMNet object.")
    end
    
    # Return the coefficient slice for the lambda of interest
    return MLMNet.B[idx[1],idy[1],:,:]
end


"""
    coef_2d(MLMNet)

Extract coefficients from Mlmnet object as a flattened 2d array

# Arguments 

- MLMNet = Mlmnet object

# Value

2d array of flattened coefficients, where each column corresponds to a 
different lambdaL1 and lambdaL2

"""
function coef_3d(MLMNet::MlmnetNet)
    
    # Extract coefficients
    coeffs = coef(MLMNet) # this is a 4-dimensional array
    
    # Initialize 2d array to store flattened coefficients
    coeffs3d = Array{Float64}(undef, size(coeffs,3)*size(coeffs,4), 
                              size(coeffs,1), size(coeffs,2))

    # Iterate through each coefficient slice
    for i in 1:size(coeffs, 1), j in size(coeffs, 2)
        coeffs3d[:,:,i,j] = vec(coeffs[i,j,:,:])
    end
    
    return coeffs3d
end


"""
    predict(MLMNet, lambdaL1, lambdaL2, newPredictors)

Calculate new predictions based on Mlmnet object and given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- newPredictors = Predictors object. Defaults to the data.predictors field 
  in the MLM object used to fit the model. 

# Value

2d array of predicted values

"""
function predict(MLMNet::MlmnetNet, lambdaL1::Float64, lambdaL2::Float64,
                 newPredictors::Predictors=MLMNet.data.predictors)
    
    # Include X and Z intercepts in new predictors if necessary
    if MLMNet.data.predictors.isXIntercept==true && 
       newPredictors.isXIntercept==false
        newPredictors.X = add_intercept(newPredictors.X)
        newPredictors.isXIntercept = true
        println("Adding X intercept to newPredictors.")
  	end
    if MLMNet.data.predictors.isZIntercept==true && 
       newPredictors.isZIntercept==false
        newPredictors.Z = add_intercept(newPredictors.Z)
        newPredictors.isZIntercept = true
        println("Adding Z intercept to newPredictors.")
  	end

    # Remove X and Z intercepts in new predictors if necessary
    if MLMNet.data.predictors.isXIntercept==false && 
       newPredictors.isXIntercept==true
        newPredictors.X = remove_intercept(newPredictors.X)
        newPredictors.isXIntercept = false
        println("Removing X intercept from newPredictors.")
    end
    if MLMNet.data.predictors.isZIntercept==false && 
       newPredictors.isZIntercept==true
        newPredictors.Z = remove_intercept(newPredictors.Z)
        newPredictors.isZIntercept = false
        println("Removing Z intercept from newPredictors.")
    end
    
    # Calculate new predictions
    return calc_preds(newPredictors.X, newPredictors.Z, coef(MLMNet, lambdaL1, lambdaL2))
end 


"""
    predict(MLMNet, newPredictors)

Calculate new predictions based on Mlmnet object

# Arguments 

- MLMNet = Mlmnet object
- newPredictors = Predictors object. Defaults to the data.predictors field 
  in the MLM object used to fit the model. 

# Value

4d array of predicted values

"""
function predict(MLMNet::MlmnetNet, 
                 newPredictors::Predictors=MLMNet.data.predictors)
    
    # Initialize 4d array for storing predictions
    all_preds = Array{Float64}(undef, length(MLMNet.lambdasL1), length(MLMNet.lambdasL2),
                               size(newPredictors.X,1), 
                               size(newPredictors.Z,1))
	# Calculate new predictions for each lambda
    for i = 1:length(MLMNet.lambdasL1), j = 1:length(MLMNet.lambdasL2)
        all_preds[i,j,:,:] = predict(MLMNet, MLMNet.lambdasL1[i], MLMNet.lambdasL2[j], 
                                        newPredictors)
    end
    
    return all_preds
end 


"""
    fitted(MLMNet, lambda)

Calculate fitted values of an Mlmnet object, given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar

# Value

2d array of fitted values

"""
function fitted(MLMNet::MlmnetNet, lambdaL1::Float64, lambdaL2::Float64)
    
    return predict(MLMNet, lambdaL1, lambdaL2)
end


"""
    fitted(MLMNet)

Calculate fitted values of an Mlmnet object

# Arguments 

- MLMNet = Mlmnet object

# Value

3d array of fitted values

"""
function fitted(MLMNet::MlmnetNet)
    
    return predict(MLMNet)
end


"""
    resid(MLMNet, lambdaL1, lambdaL2, newData)

Calculate residuals of an MLMNet object, given a lambda 

# Arguments 

- MLM = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- newData = RawData object. Defaults to the data field in the MLM object 
  used to fit the model. 

# Value

2d array of residuals

"""
function resid(MLMNet::MlmnetNet, lambdaL1::Float64, lambdaL2::Float64, newData::RawData=MLMNet.data)
    
    # Include X and Z intercepts in new data if necessary
    if MLMNet.data.predictors.isXIntercept==true && 
       newData.predictors.isXIntercept==false
        newData.predictors.X = add_intercept(newData.predictors.X)
        newData.predictors.isXIntercept = true
        newData.p = newData.p + 1
        println("Adding X intercept to newData.")
    end
    if MLMNet.data.predictors.isZIntercept==true && 
       newData.predictors.isZIntercept==false
        newData.predictors.Z = add_intercept(newData.predictors.Z)
        newData.predictors.isZIntercept = true
        newData.q = newData.q + 1
        println("Adding Z intercept to newData.")
    end
    
    # Remove X and Z intercepts in new data if necessary
    if MLMNet.data.predictors.isXIntercept==false && 
       newData.predictors.isXIntercept==true
        newData.predictors.X = remove_intercept(newData.predictors.X)
        newData.predictors.isXIntercept = false
        newData.p = newData.p - 1
        println("Removing X intercept from newData.")
    end
    if MLMNet.data.predictors.isZIntercept==false && 
       newData.predictors.isZIntercept==true
        newData.predictors.Z = remove_intercept(newData.predictors.Z)
        newData.predictors.isZIntercept = false
        newData.q = newData.q - 1
        println("Removing Z intercept from newData.")
    end
    
    # Calculate residuals
    return calc_resid(get_X(newData), get_Y(newData), get_Z(newData), 
                      coef(MLMNet, lambdaL1, lambdaL2))
end


"""
    resid(MLMNet, newData)

Calculate residuals of an MLMNet object

# Arguments 

- MLMNet = Mlmnet object
- newData = RawData object. Defaults to the data field in the MLM object 
  used to fit the model. 

# Value

3d array of residuals

"""
function resid(MLMNet::MlmnetNet, newData::RawData=MLMNet.data)
    
    # Initialize 4d array for storing residuals
    all_resid = Array{Float64}(undef, length(MLMNet.lambdasL1), length(MLMNet.lambdasL2), 
                               newData.n, newData.m)
    
    # Calculate residuals for each lambda
    for i = 1:length(MLMNet.lambdasL1), j = 1:length(MLMNet.lambdasL2)
        all_resid[i,j,:,:] = resid(MLMNet, MLMNet.lambdasL1[i], MLMNet.lambdasL2[j], newData)
    end
    
    return all_resid
end
