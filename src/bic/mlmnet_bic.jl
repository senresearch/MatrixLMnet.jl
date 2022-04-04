"""
    Mlmnet_bic(MLMNet, lambdas, alphas, data)

Type for storing the results of running BIC validation for `mlmnet`

"""
mutable struct Mlmnet_bic 
    
    # Mlmnet object
    MLMNet::Mlmnet 
       
    # These will be generated and should not be supplied as input to the 
    # constructor.
    # Test MSE for each of model with respect to each lambda and alpha
    mse::Array{Float64,2} 
    # Proportion of zero interaction coefficients for each model 
    # with respect to each lambda and alpha
    propZero::Array{Float64,2} 
    # BIC for each of model with respect to each lambda and alpha
    bic::Array{Float64,2} 
    
    Mlmnet_bic(MLMNet, dig) = 
        new(MLMNet, 
            calc_mse(MLMNet),
            calc_prop_zero(MLMNet; dig=dig), 
            calc_bic(MLMNet))
end

"""
    mlmnet_bic(data, lambdas, alphas, rowFolds, colFolds; 
              method="ista", isNaive=false,
              hasXIntercept=true, hasZIntercept=true, 
              toXReg=trues(size(get_X(data), 2)), 
              toZReg=trues(size(get_Z(data), 2)), 
              toXInterceptReg=false, toZInterceptReg=false, 
              toNormalize=true, isVerbose=true, 
              stepsize=0.01, setStepsize=true, 
              dig=12, funArgs...)

Performs BIC validation for `mlmnet`. 

# Arguments

- data = RawData object
- lambdas = 1d array of floats consisting of the total penalties in descending 
  order. If they are not in descending order, they will be sorted. 
- alphas = 1d array of floats consisting of the penalty ratio that 
  determines the mix of penalties between L1 and L2

# Keyword arguments

- methods = function name that applies the Elastic-net penalty estimate method;
  default is `ista`, and the other methods are `fista`, `fista_bt`, `admm` and `cd`
- isNaive = boolean flag indicating whether to solve the Naive or non-Naive 
  Elastic-net problem
- hasXIntercept = boolean flag indicating whether or not to include an `X` 
  intercept (row main effects). Defaults to `true`. 
- hasZIntercept = boolean flag indicating whether or not to include a `Z` 
  intercept (column main effects). Defaults to `true`.
- toXReg = 1d array of bit flags indicating whether or not to regularize each 
  of the `X` (row) effects. Defaults to 2d array of `true`s with length 
  equal to the number of `X` effects (equivalent to `data.p`). 
- toZReg = 1d array of bit flags indicating whether or not to regularize each 
  of the `Z` (column) effects. Defaults to 2d array of `true`s with length 
  equal to the number of `Z` effects (equivalent to `data.q`). 
- toXInterceptReg = boolean flag indicating whether or not to regularize the 
  `X` intercept Defaults to `false`. 
- toZInterceptReg = boolean flag indicating whether or not to regularize the 
  `Z` intercept. Defaults to `false`. 
- toNormalize = boolean flag indicating if the columns of `X` and `Z` 
  should be standardized (to mean 0, standard deviation 1). Defaults to `true`.
- isVerbose = boolean flag indicating whether or not to print messages.  
  Defaults to `true`. 
- stepsize = float; step size for updates (irrelevant for coordinate 
  descent and when `setStepsize` is set to `true` for `ista!` and `fista!`). 
  Defaults to `0.01`. 
- setStepsize = boolean flag indicating whether the fixed step size should be 
  calculated (for `ista!` and `fista!`). Defaults to `true`.
- dig = integer; digits of precision for zero coefficients. Defaults to 12. 
- funArgs = variable keyword arguments to be passed into `fun`

# Value

An Mlmnet_bic object. 

# Some notes

This is the base `mlmnet_bic` function that all other variants call. 

"""
function mlmnet_bic(data::RawData, 
                   lambdas::AbstractArray{Float64,1},
                   alphas::AbstractArray{Float64,1}; 
                   method::String="ista", isNaive::Bool=false,
                   hasXIntercept::Bool=true, hasZIntercept::Bool=true, 
                   toXReg::BitArray{1}=trues(size(get_X(data), 2)), 
                   toZReg::BitArray{1}=trues(size(get_Z(data), 2)), 
                   toXInterceptReg::Bool=false, toZInterceptReg::Bool=false, 
                   toNormalize::Bool=true, isVerbose::Bool=true, 
                   stepsize::Float64=0.01, setStepsize::Bool=true, 
                   dig::Int64=12, funArgs...)
                   
    

    # Run mlmnet on each RawData object, in parallel when possible
    MLMNet = mlmnet(data, lambdas, alphas;
                      method=method, isNaive=isNaive, 
                      hasXIntercept=hasXIntercept, 
                      hasZIntercept=hasZIntercept, 
                      toXReg=toXReg, toZReg=toZReg, 
                      toXInterceptReg=toXInterceptReg, 
                      toZInterceptReg=toZInterceptReg, 
                      toNormalize=toNormalize, 
                      isVerbose=isVerbose, 
                      stepsize=stepsize, 
                      setStepsize=setStepsize, 
                      funArgs...)

    return Mlmnet_bic(MLMNet, dig)
end
