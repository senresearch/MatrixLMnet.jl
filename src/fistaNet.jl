"""
    update_fistaNet!(B, B_prev, A, resid, resid_B, grad, X, Y, Z, norms, lambda, 
                  reg, iter, stepsize)

Updates coefficient estimates in place for each FISTA iteration based on the Elastic-net
silution, when `X` and `Z` are both standardized. 

# Arguments 

- B = 2d array of floats consisting of coefficient estimates
- B_prev = 2d array of floats consisting of coefficient estimates saved from 
  the previous iteration
- A = 2d array of floats consisting of extrapolated coefficients 
- resid = 2d array of floats consisting of the residuals calculated from the 
  extrapolated coefficients
- resid_B = 2d array of floats consisting of the residuals calculated from 
  the coefficient estimates
- grad = 2d array of floats consisting of the gradient
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response observations
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = `nothing`
- lambdaL1 = l1 penalty, a floating scalar
- lambdaL2 = l2 penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients
- iter = 1d array consisting of a single integer keeping track of how many 
  iterations have been computed
- stepsize = 1d array consisting of a float; step size of updates

# Value

None; updates coefficients in place

"""
function update_fista!(B::AbstractArray{Float64,2}, 
                       B_prev::AbstractArray{Float64,2}, 
                       A::AbstractArray{Float64,2}, 
                       resid::AbstractArray{Float64,2}, 
                       resid_B::AbstractArray{Float64,2}, 
                       grad::AbstractArray{Float64,2}, 
                       X::AbstractArray{Float64,2}, 
                       Y::AbstractArray{Float64,2}, 
                       Z::AbstractArray{Float64,2}, 
                       norms::Nothing, lambdaL1::Float64, lambdaL2::Float64,
                       reg::BitArray{2}, 
                       iter::AbstractArray{Int64,1}, 
                       stepsize::AbstractArray{Float64,1})
    
    # Update residuals based on extrapolated coefficients 
    calc_resid!(resid, X, Y, Z, A) 
    # Update gradient
    calc_grad!(grad, X, Z, resid) 

    # Cycle through all the coefficients to perform assignments
    for j = 1:size(B,2), i = 1:size(B,1)
        # Update coefficient estimate in the previous copy 
        if B_prev[i,j] != B[i,j] 
            B_prev[i,j] = B[i,j] 
        end
        
        B[i,j] = A[i,j] - stepsize[1] * grad[i,j] # L2 updates
        # Apply shrinkage to regularized coefficients
        if reg[i,j] 
            B[i,j] = prox(A[i,j], grad[i,j], sign(B[i,j]), lambdaL1*stepsize, norms, 
                          stepsize[1])/(1+lambdaL2*stepsize)
        end
        
        # Update extrapolated coefficient
        A[i,j] = B[i,j] + ((iter[1]-1.0)/(iter[1]+2.0))*(B[i,j] - B_prev[i,j]) 
    end 
    
    # Update residuals based on coefficient estimates
    calc_resid!(resid_B, X, Y, Z, B) 
end


"""
    update_fista!(B, B_prev, A, resid, resid_B, grad, X, Y, Z, norms, lambda, 
                  reg, iter, stepsize)

Updates coefficient estimates in place for each FISTA iteration when `X` and 
`Z` are not standardized. 

# Arguments 

- B = 2d array of floats consisting of coefficient estimates
- B_prev = 2d array of floats consisting of coefficient estimates saved from 
  the previous iteration
- A = 2d array of floats consisting of extrapolated coefficients 
- resid = 2d array of floats consisting of the residuals calculated from the 
  extrapolated coefficients
- resid_B = 2d array of floats consisting of the residuals calculated from 
  the coefficient estimates
- grad = 2d array of floats consisting of the gradient
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response observations
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient
- lambdaL1 = l1 penalty, a floating scalar
- lambdaL2 = l2 penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients
- iter = 1d array consisting of a single integer keeping track of how many 
  iterations have been computed
- stepsize = 1d array consisting of a float; step size of updates

# Value

None; updates coefficients in place

"""
function update_fista!(B::AbstractArray{Float64,2}, 
                       B_prev::AbstractArray{Float64,2}, 
                       A::AbstractArray{Float64,2}, 
                       resid::AbstractArray{Float64,2}, 
                       resid_B::AbstractArray{Float64,2}, 
                       grad::AbstractArray{Float64,2}, 
                       X::AbstractArray{Float64,2}, 
                       Y::AbstractArray{Float64,2}, 
                       Z::AbstractArray{Float64,2}, 
                       norms::AbstractArray{Float64,2}, 
                       lambdaL1::Float64, lambdaL2::Float64, 
                       reg::BitArray{2}, 
                       iter::AbstractArray{Int64,1}, 
                       stepsize::AbstractArray{Float64,1})
    
    # Update residuals based on extrapolated coefficients 
    calc_resid!(resid, X, Y, Z, A) 
    # Update gradient
    calc_grad!(grad, X, Z, resid) 

    # Cycle through all the coefficients to perform assignments
    for j = 1:size(B,2), i = 1:size(B,1)
        # Update coefficient estimate in the previous copy 
        if B_prev[i,j] != B[i,j] 
            B_prev[i,j] = B[i,j] 
        end
        
        B[i,j] = A[i,j] - stepsize[1] * grad[i,j]./norms[i,j] # L2 updates
        # Apply shrinkage to regularized coefficients
        if reg[i,j] 
            B[i,j] = prox(A[i,j], grad[i,j], sign(B[i,j]), lambdaL1*stepsize, norms[i,j], 
                          stepsize[1])/(1+lambdaL2*stepsize)
        end
        
        # Update extrapolated coefficient
        A[i,j] = B[i,j] + ((iter[1]-1.0)/(iter[1]+2.0))*(B[i,j] - B_prev[i,j])
    end 
    
    # Update residuals based on coefficient estimates
    calc_resid!(resid_B, X, Y, Z, B) 
end


"""
    fistaNet!(X, Y, Z, lambda, B, regXidx, regZidx, reg, norms; 
           isVerbose, stepsize, thresh, maxiter)

Performs the Elastic-net version FISTA with fixed step size.

# Arguments

- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- lambdaL1 = l1 penalty, a floating scalar
- lambdaL2 = l2 penalty, a floating scalar
- B = 2d array of floats consisting of starting coefficient estimates
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
- thresh = threshold at which the coefficients are considered to have 
  converged, a floating scalar. Defaults to `10^(-7)`. 
- maxiter = maximum number of update iterations. Defaults to `10^10`. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

The default method for choosing the fixed step size for `fista!` is to use the 
reciprocal of the product of the maximum eigenvalues of `X*transpose(X)` 
and `Z*transpose(Z)`. This is computed by the `mlmnet` function when `fista!` 
is passed into the `fun` argument and `setStepSize` is set to `true`. If 
`setStepSize` is set to `false`, the value of the `stepsize` argument will be 
used as the fixed step size. Note that obtaining the eigenvalues when `X` 
and/or `Z` are very large may exceed computational limitations. 

"""
function fistaNet!(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
                Z::AbstractArray{Float64,2}, 
                lambdaL1::Float64, lambdaL2::Float64,
                B::AbstractArray{Float64,2}, 
                regXidx::AbstractArray{Int64,1}, 
                regZidx::AbstractArray{Int64,1}, reg::BitArray{2}, norms; 
                isVerbose::Bool=true, stepsize::Float64=0.01, 
                thresh::Float64=10.0^(-7), maxiter::Int=10^10)
  
    # Pre-allocating arrays to store residuals for extrapolated coefficients 
    # and gradient. 
    resid = Array{Float64}(undef, size(Y)) 
    grad = Array{Float64}(undef, size(B))
    
    # Calculate residuals 
    calc_resid!(resid, X, Y, Z, B) 
    # Initialize residuals using coefficient estimates
    resid_B = copy(resid) 
    # Store coefficients from previous iteration
    B_prev = copy(B) 
    # Store extrapolated coefficients
    A = copy(B) 
    
    # Denominators of criterion
    crit_denom = [size(X,1)*size(Z,1), size(X,2)*size(Z,2)] 
    # Placeholder to store the old criterion 
    oldcrit = 1.0 
    # Calculate the current criterion
    crit = criterion(B[regXidx, regZidx], resid_B, lambdaL1, lambdaL2, crit_denom) 
    
    stepsize = [stepsize]
    iter = [0]
    # Iterate until coefficients converge or maximum iterations have been 
    # reached.
    while (abs(log(crit/oldcrit)) > thresh) && (iter[1] < maxiter)
        # Store the current criterion
        oldcrit = crit 
        # Update the coefficient estimates
        update_fistaNet!(B, B_prev, A, resid, resid_B, grad, X, Y, Z, norms, 
                      lambdaL1, lambdaL2, reg, iter, stepsize) 
        # Calculate the criterion after updating
        crit = criterion(B[regXidx, regZidx], resid_B, lambdaL1, lambdaL2, crit_denom) 
        
        iter[:] = iter .+ 1 # Increment the number of iterations
        # Warning message if coefficient estimates do not converge.
        if iter==(maxiter) 
            println("Estimates did not converge.")
        end
    end

    println_verbose(string("Criterion: ", crit), isVerbose)
    println_verbose(string("Number of iterations: ", iter[1]), isVerbose)
end