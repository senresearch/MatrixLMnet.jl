"""
    make_folds(n, k, l)

Generate `k` non-overlapping folds. 

# Arguments

- n = Total number of observations to split into folds. 
- k = Number of folds to create. Defaults to 10. If `k=1`, then all the data 
  (along this dimension) will be included in each fold. 
- k2 = If `k=1`, then all the data (along this dimension) will be included in 
  each fold. `k2` specifies how many folds there are. Defaults to `k`, which 
  is kind of silly, but there needs to be a placeholder. 

# Value

1d array of length `k` of arrays of indices 

"""
function make_folds(n::Int64, k::Int64=10, k2::Int64=k)
    
    if k > 1
        # When k > 1, run the Kfolds function from MLBase to create folds of length ~ n*(1-1/k)
        return collect(Array{Int64,1}, Kfold(n, k))
    elseif k == 1 
        # When k = 1, repeat all indices 1:n k2 times
        return repeat([collect(1:n)], inner=k2)
    else 
        error("k must be strictly positive.")
    end
end


"""
    make_folds_conds(conds, k, prop)

Generate `k` folds for a set of conditions, making sure each level of each 
condition is represented in each fold. 

# Arguments

- conds = 1d array of conditions (strings)
- k = Number of folds to create. Defaults to 10. 
- prop = Proportion of each condition level's replicates to include in each 
  fold. Defaults to 1/k. Each fold will contain at least one replicate of 
  each condition level. 

# Value

1d array of length `k` of arrays of indices 

"""
function make_folds_conds(conds::AbstractArray{String,1}, 
                          k::Int64=10, prop::Float64=1/k)
						  
    if k == 1
        print("k is set to 1; may result in unexpected behavior")
    elseif k < 1
        error("k must be strictly positive.")
    end
    
    # All indices to split into folds
    idx = 1:length(conds)
    # 2d boolean array to store which observations to keep in each fold
    boolFolds = falses(length(conds), k)
    
    # Iterate through all the conditions
    for cond in unique(conds)
        # For each condition, calculate the number of replicates to sample
        numSamp = convert(Int64, ceil(sum(cond .== conds) * prop))
        # Iterate through all the folds
        for j in 1:k
            # For each fold, randomly keep a sample of each condition
            boolFolds[rand(idx[cond .== conds], numSamp) ,j] .= true
        end
    end
    
    # Initialze 1d array of arrays of indices
    folds = fill(convert(Array{Int64}, []), k)
    # Convert each column of booleans from boolFolds into indices
    for j in 1:k 
        folds[j] = findall(boolFolds[:,j])
    end
    
    return folds
end


"""
    findnotin(a, b)

Returns elements of `b` that are not present in `a`. 

# Arguments

- a = 1d array of integers
- b = 1d array of integers

# Value

1d array of integers

"""
function findnotin(a::AbstractArray{Int64,1}, b::AbstractArray{Int64,1})
    
    if a != b
        return [x for x=b if !(x in a)]
    else
        return a
    end
end


"""
    calc_mse(MLMNets, data, lambdas, rowFolds, colFolds)

Calculates test MSE for each of the CV folds for each lambda. 

# Arguments 

- MLMNets = 1d array of Mlmnet objects resulting from running cross validation
- data = RawData object used to generate MLMNets
- lambdas = 1d array of floats consisting of lambda penalties used to 
  generate MLMNets
- rowFolds = 1d array of arrays containing booleans for the row folds
- colFolds = 1d array of arrays containing booleans for the column folds

# Value

2d array of floats with dimensions equal to the number of lambdas by the 
number of folds. 

"""
function calc_mse(MLMNets::AbstractArray{Mlmnet,1}, data::RawData, 
                  lambdas::AbstractArray{Float64,1}, 
                  rowFolds::Array{Array{Int64,1},1}, 
                  colFolds::Array{Array{Int64,1},1})

    # Number of folds
    nFolds = length(MLMNets)

    # For each fold, generate a RawData object corresponding to the holdout set
    holdoutFolds = Array{RawData}(undef, nFolds)
    for i in 1:nFolds
        holdoutFolds[i] = 
          RawData(Response(get_Y(data)[findnotin(rowFolds[i], 
                                                 collect(1:data.n)), 
                                       findnotin(colFolds[i], 
                                                 collect(1:data.m))]), 
                  Predictors(get_X(data)[findnotin(rowFolds[i], 
                                                   collect(1:data.n)),:], 
                             get_Z(data)[findnotin(colFolds[i], 
                                                   collect(1:data.m)),:], 
                             data.predictors.isXIntercept, 
                             data.predictors.isZIntercept))
    end
  
    # Initialize array to store test MSEs 
    mse = Array{Float64}(undef, length(lambdas), nFolds)
    
    # Iterate through all folds and lambdas to calculate test MSE
    for j in 1:nFolds
        # Residuals for all lambdas at fold j
        resids = resid(MLMNets[j], holdoutFolds[j]) 
        for i in 1:length(lambdas)
            # MSE for lambda i and fold j   
            mse[i,j] = mean(resids[:,:,i].^2) 
        end
    end

    return mse
end


"""
    calc_prop_zero(MLMNets, lambdas, dig)

Calculates proportion of zero interaction coefficients for each of the CV 
folds for each lambda. 

# Arguments 

- MLMNets = 1d array of Mlmnet objects resulting from running cross validation
- lambdas = 1d array of floats consisting of lambda penalties used to 
  generate MLMNets

# Keyword arguments 

- dig = integer; digits of precision for zero coefficients. Defaults to 12. 

# Value

2d array of floats with dimensions equal to the number of lambdas by the 
number of folds. 

"""
function calc_prop_zero(MLMNets::AbstractArray{Mlmnet,1}, 
                        lambdas::AbstractArray{Float64,1}; 
                        dig::Int64=12)
    
    # Number of folds
    nFolds = length(MLMNets)
  
    # Initialize array to store proportions of zero interactions 
    propZero = Array{Float64}(undef, length(lambdas), nFolds)

    # Boolean arrays used to subset coefficients for interactions only
    xIdx = trues(MLMNets[1].data.p)
    xIdx[1] = MLMNets[1].data.predictors.isXIntercept==false
    zIdx = trues(MLMNets[1].data.q)
    zIdx[1] = MLMNets[1].data.predictors.isZIntercept==false 
  
    # Iterate through all folds and lambdas to calculate proportion of zero 
    # interaction coefficients 
    for j in 1:nFolds
        for i in 1:length(lambdas)
            # Proportion of zero interaction coefficients for lambda i and 
            # fold j  
            propZero[i,j] = mean(round.(coef(MLMNets[j])[xIdx,zIdx,i], 
                                        digits=dig) .== 0) 
        end
    end

    return propZero
end
############################# Elastic-net #############################
"""
    calc_mseNet(MLMNets, data, lambdas, rowFolds, colFolds)

Calculates test MSE for each of the CV folds for each lambda. 

# Arguments 

- MLMNets = 1d array of MlmnetNet objects resulting from running cross validation
- data = RawData object used to generate MLMNets
- lambdasL1 = 1d array of floats consisting of lambda penalties used to 
  generate MLMNets
- lambdasL2 = 1d array of floats consisting of lambda penalties used to 
  generate MLMNets
- rowFolds = 1d array of arrays containing booleans for the row folds
- colFolds = 1d array of arrays containing booleans for the column folds

# Value

2d array of floats with dimensions equal to the number of lambdas by the 
number of folds. 

"""
function calc_mseNet(MLMNets::AbstractArray{MlmnetNet,1}, data::RawData, 
                  lambdas::AbstractArray{Float64,1}, 
                  alphas::AbstractArray{Float64,1},
                  rowFolds::Array{Array{Int64,1},1}, 
                  colFolds::Array{Array{Int64,1},1})

    # Number of folds
    nFolds = length(MLMNets)

    # For each fold, generate a RawData object corresponding to the holdout set
    holdoutFolds = Array{RawData}(undef, nFolds)
    for i in 1:nFolds
        holdoutFolds[i] = 
          RawData(Response(get_Y(data)[findnotin(rowFolds[i], 
                                                 collect(1:data.n)), 
                                       findnotin(colFolds[i], 
                                                 collect(1:data.m))]), 
                  Predictors(get_X(data)[findnotin(rowFolds[i], 
                                                   collect(1:data.n)),:], 
                             get_Z(data)[findnotin(colFolds[i], 
                                                   collect(1:data.m)),:], 
                             data.predictors.isXIntercept, 
                             data.predictors.isZIntercept))
    end
  
    # Initialize array to store test MSEs 
    mse = Array{Float64}(undef, length(lambdas), length(alphas), nFolds)
    
    # Iterate through all folds and lambdas to calculate test MSE
    for j in 1:nFolds
        # Residuals for all lambdasL1 and lambdasL2 at fold j
        resids = resid(MLMNets[j], holdoutFolds[j]) 
        for i in 1:length(lambdas), k in 1:length(alphas)
            # MSE for (lambdaL1 i, lambdaL2 k) and fold j   
            mse[i,k,j] = mean(resids[:,:,i,k].^2) 
        end
    end

    return mse
end


"""
    calc_prop_zeroNet(MLMNets, lambdasL1, lambdasL2, dig)

Calculates proportion of zero interaction coefficients for each of the CV 
folds for each lambda. 

# Arguments 

- MLMNets = 1d array of Mlmnet objects resulting from running cross validation
- lambdas = 1d array of floats consisting of lambda penalties used to 
  generate MLMNets

# Keyword arguments 

- dig = integer; digits of precision for zero coefficients. Defaults to 12. 

# Value

2d array of floats with dimensions equal to the number of lambdas by the 
number of folds. 

"""
function calc_prop_zeroNet(MLMNets::AbstractArray{MlmnetNet,1}, 
                        lambdas::AbstractArray{Float64,1},
                        alphas::AbstractArray{Float64,1}; 
                        dig::Int64=12)
    
    # Number of folds
    nFolds = length(MLMNets)
  
    # Initialize array to store proportions of zero interactions 
    propZero = Array{Float64}(undef, length(lambdas), length(alphas), nFolds)

    # Boolean arrays used to subset coefficients for interactions only
    xIdx = trues(MLMNets[1].data.p)
    xIdx[1] = MLMNets[1].data.predictors.isXIntercept==false
    zIdx = trues(MLMNets[1].data.q)
    zIdx[1] = MLMNets[1].data.predictors.isZIntercept==false 
  
    # Iterate through all folds and lambdas to calculate proportion of zero 
    # interaction coefficients 
    for j in 1:nFolds
        for k in 1:length(alphas),i in 1:length(lambdas)
            # Proportion of zero interaction coefficients for lambda i and 
            # fold j  
            propZero[i,k,j] = mean(round.(coef(MLMNets[j])[xIdx,zIdx,i,k], 
                                        digits=dig) .== 0) 
        end
    end

    return propZero
end