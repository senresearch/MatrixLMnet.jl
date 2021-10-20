"""
    Mlmnet(B, lambdas, data)

Type for storing the results of an mlmnet model fit

"""
mutable struct Mlmnet 
    
    # Coefficient estimates
    B::Array{Float64,3} 
    # Lambda penalties
    lambdas::Array{Float64,1} 
    
    # Response and predictor matrices
    data::RawData 
end


"""
    mlmnet_pathwise(fun, X, Y, Z, lambdas, regXidx, regZidx, reg, norms; 
                    isVerbose, stepsize, funArgs...)

Performs the supplied method on a descending list of lambdas using ``warm 
starts''. 

# Arguments

- fun = function that applies an L1-penalty estimate method
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- lambdas = 1d array of floats consisting of lambda penalties in descending 
  order. If they are not in descending order, they will be sorted. 
- regXidx = 1d array of indices corresponding to regularized X covariates
- regZidx = 1d array of indices corresponding to regularized Z covariates
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient or `nothing`

# Keyword arguments

- isVerbose = boolean flag indicating whether or not to print messages.  
  Defaults to `true`. 
- stepsize = float; step size for updates. Defaults to `0.01`. 
- funArgs = variable keyword arguments to be passed into `fun`

# Value

A 3d array consisting of the coefficient estimates, with the different 
lambdas along the first dimension

# Some notes

Assumes that all necessary standardizations have been performed on X, Y, and 
Z, including adding on intercepts. To be called by `mlmnet`, which performs 
standardization and backtransforming. 

"""
function mlmnet_pathwise(fun::Function, X::AbstractArray{Float64,2}, 
                         Y::AbstractArray{Float64,2}, 
                         Z::AbstractArray{Float64,2}, 
                         lambdas::AbstractArray{Float64,1},  
                         regXidx::AbstractArray{Int64,1}, 
                         regZidx::AbstractArray{Int64,1}, 
                         reg::BitArray{2}, norms; isVerbose::Bool=true, 
                         stepsize::Float64=0.01, funArgs...)

    # Check that the lambdas are unique and in descending order. 
    if length(lambdas) != length(unique(lambdas))
        println_verbose("Dropping non-unique lambdas", isVerbose)
    end

    if any(lambdas .!= sort(lambdas, rev=true))
        println_verbose("Sorting lambdas into descending order.", isVerbose)
        lambdas .= sort(lambdas, rev=true)
    end 

    # Pre-allocate array for coefficients
    coeffs = Array{Float64}(undef, size(X,2), size(Z,2), length(lambdas)) # issue#10 ✓

    # Start with coefficients initalized at zero for the largest lambda value
    startB = zeros(size(X,2), size(Z,2))

    # Pre-compute eigenvalues and eigenvectors for ADMM
    if length(string(fun)) > 4 && (string(fun)[(end-4):end] == "admm!") 
        # Eigenfactorization of X
        XTX = transpose(X)*X
        eigX = eigen(XTX)
        Qx = eigX.vectors
        Lx = eigX.values
        
        # Eigenfactorization of Z
        ZTZ = transpose(Z)*Z
        eigZ = eigen(ZTZ)
		    Qz = eigZ.vectors
		    Lz = eigZ.values
    
        # Transformed Y
        X1 = X*Qx
        Z1 = Z*Qz
        U = transpose(X1) * Y * Z1

        # Kronecker product of eigenvalues of Z and X
        L = kron(Lx, transpose(Lz))
    end

    # Iterate through the path of lambdas
    for i=1:length(lambdas) 
        # Get L1-penalty estimates by updating the coefficients from previous 
        # iteration in place
        if length(string(fun)) <= 4 || (string(fun)[(end-4):end] != "admm!")
            fun(X, Y, Z, lambdas[i], startB, regXidx, regZidx, reg, norms; 
                isVerbose=isVerbose, stepsize=stepsize, funArgs...)
        else
            fun(X, Y, Z, lambdas[i], startB, regXidx, regZidx, reg, norms, 
                Qx, Qz, U, L; 
                isVerbose=isVerbose, stepsize=stepsize, funArgs...)
        end

        # Assign a slice of coeffs to the current coefficient estimates
        coeffs[:,:,i] = startB  # issue#10 ✓
    end

    return coeffs
end




"""
    mlmnet(fun, data, lambdas; 
           hasXIntercept, hasZIntercept, toXReg, toZReg, 
           toXInterceptReg, toZInterceptReg, toStandardize, isVerbose, 
           stepsize, setStepsize, funArgs...)

Standardizes X and Z predictor matrices, calculates fixed step size, performs 
the supplied method on a descending list of lambdas using ``warm starts'', 
and backtransforms resulting coefficients, as is deemed necessary by the user 
inputs.

# Arguments

- fun = function that applies an L1-penalty estimate method
- data = RawData object
- lambdas = 1d array of floats consisting of lambda penalties in descending 
order. If they are not in descending order, they will be sorted. 

# Keyword arguments

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
- toStandardize = boolean flag indicating if the columns of `X` and `Z` 
  should be standardized (to mean 0, standard deviation 1). Defaults to `true`.
- isVerbose = boolean flag indicating whether or not to print messages.  
  Defaults to `true`. 
- stepsize = float; step size for updates (irrelevant for coordinate 
  descent and when `setStepsize` is set to `true` for `ista!` and `fista!`). 
  Defaults to `0.01`. 
- setStepsize = boolean flag indicating whether the fixed step size should be 
  calculated (for `ista!` and `fista!`). Defaults to `true`.
- funArgs = variable keyword arguments to be passed into `fun`

# Value

An Mlmnet object

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
function mlmnet(fun::Function, data::RawData, lambdas::AbstractArray{Float64,1}; 
                hasXIntercept::Bool=true, hasZIntercept::Bool=true, 
                toXReg::BitArray{1}=trues(data.p), 
                toZReg::BitArray{1}=trues(data.q),     
                toXInterceptReg::Bool=false, toZInterceptReg::Bool=false, 
                toStandardize::Bool=true, isVerbose::Bool=true, 
                stepsize::Float64=0.01, setStepsize::Bool=true, funArgs...)
    
    # Ensure that toXReg and toZReg have same length as columns of X and Z
    if length(toXReg) != data.p
        error("toXReg does not have same length as number of columns in X.")
    end
    if length(toZReg) != data.q
        error("toZReg does not have same length as number of columns in Z.")
    end 
    
    # Add X and Z intercepts if necessary
    # Update toXReg and toZReg accordingly
    if hasXIntercept==true && data.predictors.hasXIntercept==false
        data.predictors.X = add_intercept(data.predictors.X)
        data.predictors.hasXIntercept = true
        data.p = data.p + 1
        toXReg = vcat(toXInterceptReg, toXReg)
    end
    if hasZIntercept==true && data.predictors.hasZIntercept==false
        data.predictors.Z = add_intercept(data.predictors.Z)
        data.predictors.hasZIntercept = true
        data.q = data.q + 1
        toZReg = vcat(toZInterceptReg, toZReg)
    end
    
    # Remove X and Z intercepts in new predictors if necessary
    # Update toXReg and toZReg accordingly
    if hasXIntercept==false && data.predictors.hasXIntercept==true
        data.predictors.X = remove_intercept(data.predictors.X)
        data.predictors.hasXIntercept = false
        data.p = data.p - 1
        toXReg = toXReg[2:end]
    end
    if hasZIntercept==false && data.predictors.hasZIntercept==true
        data.predictors.Z = remove_intercept(data.predictors.Z)
        data.predictors.hasZIntercept = false
        data.q = data.q - 1
        toZReg = toZReg[2:end]
    end
    
    # Update toXReg and toZReg accordingly when intercept is already included
    if hasXIntercept==true && data.predictors.hasXIntercept==true
        toXReg[1] = toXInterceptReg
    end
    if hasZIntercept==true && data.predictors.hasZIntercept==true
        toZReg[1] = toZInterceptReg
    end
	
    # Matrix to keep track of which coefficients to regularize.
    reg = toXReg.*transpose(toZReg) 
    # Indices corresponding to regularized X covariates. 
    regXidx = findall(toXReg) 
    # Indices corresponding to regularized Z covariates. 
    regZidx = findall(toZReg) 

    # Standardize predictors, if necessary. 
    if (toStandardize==true)
        # If predictors will be standardized, copy the predictor matrices.
        X = copy(get_X(data))
        Z = copy(get_Z(data))

        # Standardize predictors
        meansX, normsX, = standardize!(X, hasXIntercept) 
        meansZ, normsZ, = standardize!(Z, hasZIntercept)
        # If X and Z are standardized, set the norm to nothing
        norms = nothing 
    else 
        # If not standardizing, create pointers for the predictor matrices
        X = data.predictors.X
        Z = data.predictors.Z

        # Calculate the norm matrix
        # 2d array of norms corresponding to each coefficient
        norms = transpose(sum(X.^2, dims=1)).*sum(Z.^2, dims=1) 
    end

    # If chosen method is ista!/fista! with fixed step size and setStepsize is 
    # true, compute the step size. 
    if length(string(fun)) > 4 && ((string(fun)[(end-4):end] == "ista!")||(string(fun)[(end-4):end] == "istb!")) && 
       setStepsize == true
        # Calculate and store transpose(X)*X
        XTX = transpose(X)*X
        # Calculate and store transpose(Z)*Z
        ZTZ = transpose(Z)*Z 
        
        # Step size is the reciprocal of the maximum eigenvalue of kron(Z, X)
        if toStandardize==true
            # Standardizing X and Z results in complex eigenvalues
            # Hack is to square the singular values to get the eigenvalues (ZFY)
            eig_X = (svd(XTX).S).^2
            eig_Z = (svd(ZTZ).S).^2

            stepsize = 1/max(maximum(eig_X) * maximum(eig_Z), 
                             minimum(eig_X) * minimum(eig_Z))
  	    else 
            stepsize = 1/max(eigmax(XTX) * eigmax(ZTZ),
                             eigmin(XTX) * eigmin(ZTZ))
        end
        
        println_verbose(string("Fixed step size set to ", stepsize), 
                        isVerbose) 
    end

    # Run the specified L1-penalty method on the supplied inputs. 
    coeffs = mlmnet_pathwise(fun, X, get_Y(data), Z, lambdas, regXidx, 
                             regZidx, reg, norms; isVerbose=isVerbose, 
                             stepsize=stepsize, funArgs...)
  
    # Back-transform coefficient estimates, if necessary. 
    # Case if including both X and Z intercepts. 
    if toStandardize == true && (hasXIntercept==true) && (hasZIntercept==true)
        backtransform!(coeffs, meansX, meansZ, normsX, normsZ, get_Y(data), 
                       data.predictors.X, data.predictors.Z)
    elseif toStandardize == true # Otherwise
        backtransform!(coeffs, hasXIntercept, hasZIntercept, meansX, meansZ, 
                       normsX, normsZ)
    end
    
    return Mlmnet(coeffs, lambdas, data)
end
