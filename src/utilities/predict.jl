"""
    coef(MLMNet)

Extract all coefficients from MlmnetDeprecated object

# Arguments 

- MLMNet = MlmnetDeprecated object

# Value

3d array of coefficients 

"""
function coef(MLMNet::Mlmnet)
    
    return MLMNet.B
end


"""
    coef(MLMNet, lambda, alpha)

Extract coefficients from MlmnetDeprecated object at a given lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- alpha = alpha penalty to determine the mix of penalties between L1 and L2
  a floating scalar

# Value

2d array of coefficients 

"""
function coef(MLMNet::Mlmnet, lambda::Float64, alpha::Float64)
    
    # Find the index corresponding to the lambda of interest
    idx = findall([isapprox(lam, lambda) for lam in MLMNet.lambdas])
    idy = findall([isapprox(alp, alpha) for alp in MLMNet.alphas])
    if (length(idx) == 0)
        error("This lambda was not used in the MLMNet object.")
    end

    if (length(idy) == 0)
        error("This alpha was not used in the MLMNet object.")
    end
    
    # Return the coefficient slice for the lambda of interest
    return MLMNet.B[:,:,idx[1],idy[1]] 
end


"""
    coef_3d(MLMNet)

Extract coefficients from MlmnetDeprecated object as a flattened 2d array

# Arguments 

- MLMNet = Mlmnet object

# Value

2d array of flattened coefficients, where each column corresponds to a 
different lambda and alpha

"""
function coef_3d(MLMNet::Mlmnet)
    
    # Extract coefficients
    coeffs = coef(MLMNet) # this is a 4-dimensional array
    
    # Initialize 2d array to store flattened coefficients
    coeffs3d = Array{Float64}(undef, size(coeffs,1)*size(coeffs,2), 
                              size(coeffs,3), size(coeffs,4))

    # Iterate through each coefficient slice
    for j in 1:size(coeffs, 4), i in size(coeffs, 3)
        coeffs3d[:,i,j] = vec(coeffs[:,:,i,j])
    end
    
    return coeffs3d
end


"""
    predict(MLMNet, lambda, alpha, newPredictors)

Calculate new predictions based on MlmnetDeprecated object and given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- alpha = alpha penalty to determine the mix of penalties between L1 and L2
  a floating scalar
- newPredictors = Predictors object. Defaults to the data.predictors field 
  in the MLM object used to fit the model. 

# Value

2d array of predicted values

"""
function predict(MLMNet::Mlmnet, lambda::Float64, alpha::Float64,
                 newPredictors::Predictors=MLMNet.data.predictors)
    
    # Include X and Z intercepts in new predictors if necessary
    if MLMNet.data.predictors.hasXIntercept==true && 
       newPredictors.hasXIntercept==false
        newPredictors.X = add_intercept(newPredictors.X)
        newPredictors.hasXIntercept = true
        println("Adding X intercept to newPredictors.")
  	end
    if MLMNet.data.predictors.hasZIntercept==true && 
       newPredictors.hasZIntercept==false
        newPredictors.Z = add_intercept(newPredictors.Z)
        newPredictors.hasZIntercept = true
        println("Adding Z intercept to newPredictors.")
  	end

    # Remove X and Z intercepts in new predictors if necessary
    if MLMNet.data.predictors.hasXIntercept==false && 
       newPredictors.hasXIntercept==true
        newPredictors.X = remove_intercept(newPredictors.X)
        newPredictors.hasXIntercept = false
        println("Removing X intercept from newPredictors.")
    end
    if MLMNet.data.predictors.hasZIntercept==false && 
       newPredictors.hasZIntercept==true
        newPredictors.Z = remove_intercept(newPredictors.Z)
        newPredictors.hasZIntercept = false
        println("Removing Z intercept from newPredictors.")
    end
    
    # Calculate new predictions
    return calc_preds(newPredictors.X, newPredictors.Z, coef(MLMNet, lambda, alpha))
end 


"""
    predict(MLMNet, newPredictors)

Calculate new predictions based on MlmnetDeprecated object

# Arguments 

- MLMNet = Mlmnet object
- newPredictors = Predictors object. Defaults to the data.predictors field 
  in the MLM object used to fit the model. 

# Value

4d array of predicted values

"""
function predict(MLMNet::Mlmnet, 
                 newPredictors::Predictors=MLMNet.data.predictors)
    
    # Initialize 4d array for storing predictions
    all_preds = Array{Float64}(undef, size(newPredictors.X,1), size(newPredictors.Z,1),
                               length(MLMNet.lambdas), length(MLMNet.alphas))

	# Calculate new predictions for each lambda
    for j = 1:length(MLMNet.alphas), i = 1:length(MLMNet.lambdas)
        all_preds[:,:,i,j] = predict(MLMNet, MLMNet.lambdas[i], MLMNet.alphas[j], 
                                        newPredictors)
    end
    
    return all_preds
end 


"""
    fitted(MLMNet, lambda, alpha)

Calculate fitted values of an MlmnetDeprecated object, given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- alpha = alpha penalty to determine the mix of penalties between L1 and L2
  a floating scalar

# Value

2d array of fitted values

"""
function fitted(MLMNet::Mlmnet, lambda::Float64, alpha::Float64)
    
    return predict(MLMNet, lambda, alpha)
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
    resid(MLMNet, lambda, alpha, newData)

Calculate residuals of an MLMNet object, given a lambda 

# Arguments 

- MLMNet = Mlmnet object
- lambda = lambda penalty to use, a floating scalar
- alpha = alpha penalty to determine the mix of penalties between L1 and L2
  a floating scalar
- newData = RawData object. Defaults to the data field in the MLM object 
  used to fit the model. 

# Value

2d array of residuals

"""
function resid(MLMNet::Mlmnet, lambda::Float64, alpha::Float64, newData::RawData=MLMNet.data)
    
    # Include X and Z intercepts in new data if necessary
    if MLMNet.data.predictors.hasXIntercept==true && 
       newData.predictors.hasXIntercept==false
        newData.predictors.X = add_intercept(newData.predictors.X)
        newData.predictors.hasXIntercept = true
        newData.p = newData.p + 1
        println("Adding X intercept to newData.")
    end
    if MLMNet.data.predictors.hasZIntercept==true && 
       newData.predictors.hasZIntercept==false
        newData.predictors.Z = add_intercept(newData.predictors.Z)
        newData.predictors.hasZIntercept = true
        newData.q = newData.q + 1
        println("Adding Z intercept to newData.")
    end
    
    # Remove X and Z intercepts in new data if necessary
    if MLMNet.data.predictors.hasXIntercept==false && 
       newData.predictors.hasXIntercept==true
        newData.predictors.X = remove_intercept(newData.predictors.X)
        newData.predictors.hasXIntercept = false
        newData.p = newData.p - 1
        println("Removing X intercept from newData.")
    end
    if MLMNet.data.predictors.hasZIntercept==false && 
       newData.predictors.hasZIntercept==true
        newData.predictors.Z = remove_intercept(newData.predictors.Z)
        newData.predictors.hasZIntercept = false
        newData.q = newData.q - 1
        println("Removing Z intercept from newData.")
    end
    
    # Calculate residuals
    return calc_resid(get_X(newData), get_Y(newData), get_Z(newData), 
                      coef(MLMNet, lambda, alpha))
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
    
    # Initialize 4d array for storing residuals
    all_resid = Array{Float64}(undef, newData.n, newData.m,
                                length(MLMNet.lambdas), length(MLMNet.alphas))
    
    # Calculate residuals for each lambda
    for j = 1:length(MLMNet.alphas), i = 1:length(MLMNet.lambdas)
        all_resid[:,:,i,j] = resid(MLMNet, MLMNet.lambdas[i], MLMNet.alphas[j], newData)
    end
    
    return all_resid
end
