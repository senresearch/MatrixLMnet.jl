"""
    mlmnet_perms(fun, data, lambdas; 
                 permFun, isXIntercept, isZIntercept, isXReg, isZReg, 
                 isXInterceptReg, isZInterceptReg, isStandardize, isVerbose, 
                 stepsize, setStepsize, funArgs...)

Permutes response matrix Y in RawData object and then calls the mlmnet core 
function. 

# Arguments

- fun = function that applies an L1-penalty estimate method
- data = RawData object
- lambdas = 1d array of floats consisting of lambda penalties in descending 
  order. If they are not in descending order, they will be sorted. 

# Keyword arguments

- permFun = function used to permute `Y`. Defaults to `shuffle_rows` 
  (shuffles rows of `Y`). 
- isXIntercept = boolean flag indicating whether or not to include an `X` 
  intercept (row main effects). Defaults to `true`. 
- isZIntercept = boolean flag indicating whether or not to include a `Z` 
  intercept (column main effects). Defaults to `true`.
- isXReg = 1d array of bit flags indicating whether or not to regularize each 
  of the `X` (row) effects. Defaults to 2d array of `true`s with length 
  equal to the number of `X` effects (equivalent to `data.p`). 
- isZReg = 1d array of bit flags indicating whether or not to regularize each 
  of the `Z` (column) effects. Defaults to 2d array of `true`s with length 
  equal to the number of `Z` effects (equivalent to `data.q`). 
- isXInterceptReg = boolean flag indicating whether or not to regularize the 
  `X` intercept Defaults to `false`. 
- isZInterceptReg = boolean flag indicating whether or not to regularize the 
  `Z` intercept. Defaults to `false`. 
- isStandardize = boolean flag indicating if the columns of `X` and `Z` 
  should be standardized (to mean 0, standard deviation 1). Defaults to `true`.
- isVerbose = boolean flag indicating whether or not to print messages.  
  Defaults to `true`. 
- setStepsize = boolean flag indicating whether the fixed step size should be 
  calculated (for `ista!` and `fista!`). Defaults to `true`.
- stepsize = float; step size for updates (irrelevant for coordinate 
  descent and when `setStepsize` is set to `true` for `ista!` and `fista!`). 
  Defaults to `0.01`. 
- funArgs = variable keyword arguments to be passed into `fun`

# Value

An MLMnet object

# Some notes

The default method for choosing the fixed step size for `fista!` or `ista!` 
is to use the reciprocal of the product of the maximum eigenvalues of 
`X*transpose(X)` and `Z*transpose(Z)`. This is computed when `fista!` or 
`ista!` is passed into the `fun` argument and `setStepsize` is set to `true`. 
If `setStepsize` is set to `false`, the value of the `stepsize` argument will 
be used as the fixed step size. Note that obtaining the eigenvalues when `X` 
and/or `Z` are very large may exceed computational limitations. 

Specifying a good starting step size (`stepsize`) and multiplying factor 
(`gamma`) when `fista_bt!` is passed into the `fun` argument can be difficult. 
Shrinking the step size too gradually can result in slow convergence. Doing so 
too quickly can cause the criterion to diverge. We have found that setting 
`stepsize` to 0.01 often works well in practice; choice of `gamma` appears to 
be less consequential. 

"""
function mlmnet_perms(fun::Function, data::RawData, 
                      lambdas::AbstractArray{Float64,1}; 
                      permFun::Function=shuffle_rows, 
                      isXIntercept::Bool=true, isZIntercept::Bool=true, 
                      isXReg::BitArray{1}=trues(data.p), 
                      isZReg::BitArray{1}=trues(data.q), 
                      isXInterceptReg::Bool=false, 
                      isZInterceptReg::Bool=false, 
                      isStandardize::Bool=true, isVerbose::Bool=true, 
                      stepsize::Float64=0.01, setStepsize=true, funArgs...)
    
    # Create RawData object with permuted Y
    dataPerm = RawData(Response(permFun(get_Y(data))), data.predictors)
    
    # Run L1-penalty on the permuted data
    return mlmnet(fun, dataPerm, lambdas; 
                  isXIntercept=isXIntercept, isZIntercept=isZIntercept, 
                  isXReg=isXReg, isZReg=isZReg, 
                  isXInterceptReg=isXInterceptReg, 
                  isZInterceptReg=isZInterceptReg, 
                  isStandardize=isStandardize, isVerbose=isVerbose, 
                  stepsize=stepsize, setStepsize=setStepsize, funArgs...)
end
