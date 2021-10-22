"""
    inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)

Updates a single coefficient estimate in place (to be called by 
`update_cd_cyclic!`, `update_cd_random!`, `update_cd_active_cyclic!`, or 
`update_cd_active_random!`) when `X` and `Z` are both standardized.

# Arguments 

- i = row index of the coefficient to update
- j = column index of the coefficient to update
- B = 2d array of floats consisting of coefficient estimates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = `nothing`
- lambda = lambda penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients

# Value

None; updates a coefficient in place

"""
function inner_update_cd!(i::Int64, j::Int64, B::AbstractArray{Float64,2}, 
                          resid::AbstractArray{Float64,2}, 
                          X::AbstractArray{Float64,2}, 
                          Z::AbstractArray{Float64,2}, 
                          norms::Nothing, lambda::Float64, reg::BitArray{2})

    gradient = calc_grad(X[:,i], Z[:,j], resid) # Calculate gradient
    b = B[i,j] # Store the current coefficient.
  
    b2update = b - gradient # L2 update  
    b2sign = sign(b2update) # L2 update sign
    
    # Update the current coefficient
    b = reg[i,j] ? prox(b, gradient, b2sign, lambda, norms) : b2update 
      
    if b != B[i,j] 
        ger!(B[i,j]-b, X[:,i], Z[:,j], resid) # Update residuals
        B[i,j] = b # Update coefficient estimates in the current copy
    end 
end


"""
    inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)

Updates a single coefficient estimate in place (to be called by 
`update_cd_cyclic!`, `update_cd_random!`, `update_cd_active_cyclic!`, or 
`update_cd_active_random!`) when `X` and `Z` are not standardized.

# Arguments 

- i = row index of the coefficient to update
- j = column index of the coefficient to update
- B = 2d array of floats consisting of coefficient estimates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient
- lambda = lambda penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients

# Value

None; updates a coefficient in place

"""
function inner_update_cd!(i::Int64, j::Int64, B::AbstractArray{Float64,2}, 
                          resid::AbstractArray{Float64,2},
                          X::AbstractArray{Float64,2}, 
                          Z::AbstractArray{Float64,2}, 
                          norms::AbstractArray{Float64,2}, lambda::Float64, 
                          reg::BitArray{2})

    gradient = calc_grad(X[:,i], Z[:,j], resid)  # Calculate gradient
    b = B[i,j] # Store the current coefficient.

    b2update = b - gradient/norms[i,j] # L2 update  
    b2sign = sign(b2update) # L2 update sign
    
    # Update the current coefficient
    b = reg[i,j] ? prox(b, gradient, b2sign, lambda, norms[i,j]) : b2update 
      
    if b != B[i,j] 
        ger!(B[i,j]-b, X[:,i], Z[:,j], resid) # Update residuals
        B[i,j] = b # Update coefficient estimates in the current copy
    end 
end


"""
    update_cd_cyclic!(B, resid, X, Z, norms, lambda, reg)

Cyclically updates coefficients in place for each coordinate descent iteration.

# Arguments 

- B = 2d array of floats consisting of coefficient estimates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient or `nothing`
- lambda = lambda penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients

# Value

None; updates coefficients in place

"""
function update_cd_cyclic!(B::AbstractArray{Float64,2}, 
                           resid::AbstractArray{Float64,2}, 
                           X::AbstractArray{Float64,2}, 
                           Z::AbstractArray{Float64,2}, 
                           norms, lambda::Float64, reg::BitArray{2})

    # Cycle through all the coefficients
    for j = 1:size(B,2), i = 1:size(B,1) 
        inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)
    end
end


"""
    update_cd_random!(B, resid, X, Z, norms, lambda, reg)

Randomly updates coefficients in place for each coordinate descent iteration.

# Arguments 

- B = 2d array of floats consisting of coefficient estimates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient or `nothing`
- lambda = lambda penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients

# Value

None; updates coefficients in place

"""
function update_cd_random!(B::AbstractArray{Float64,2}, 
                           resid::AbstractArray{Float64,2}, 
                           X::AbstractArray{Float64,2}, 
                           Z::AbstractArray{Float64,2}, 
                           norms, lambda::Float64, reg::BitArray{2})

    # Randomly shuffle the coefficient indices
    rand_idx = Random.shuffle(collect(zip(repeat(1:size(B,1), inner=size(B,2)), 
                                      repeat(1:size(B,2), outer=size(B,1))))) 

    # Update the coeffcients randomly
    for (i, j) in rand_idx 
        inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)
    end
end


"""
    update_cd_active_cyclic!(B, resid, X, Z, norms, lambda, reg, nonreg_idx, 
                             active_idx)

Cyclically updates active set of coefficients in place for each coordinate 
descent iteration.

# Arguments 

- B = 2d array of floats consisting of coefficient estimates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient or `nothing`
- lambda = lambda penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients
- nonreg_idx = tuple with the 2d indices of the non-regularized coefficients 
  as two 1d arrays of integers
- active_idx = tuple with the 2d indices of the active coefficients as two 1d 
  arrays of integers

# Value

None; updates coefficients in place

# Some notes

Given that you pass in the indices for the non-regularized and active 
(regularized) coefficients separately, this function can be further optimized 
so that you don't check for regularization when updating each coefficient 
with `inner_update!`.

"""
function update_cd_active_cyclic!(B::AbstractArray{Float64,2}, 
                                  resid::AbstractArray{Float64,2},
                                  X::AbstractArray{Float64,2}, 
                                  Z::AbstractArray{Float64,2}, 
                                  norms, lambda::Float64, reg::BitArray{2}, 
                                  nonreg_idx::Tuple{AbstractArray{Int64,1},
                                                    AbstractArray{Int64,1}}, 
                                  active_idx::Tuple{AbstractArray{Int64,1},
                                                    AbstractArray{Int64,1}})

  # Cycle through the non-regularized coefficients
  for (i,j) in zip(nonreg_idx[1], nonreg_idx[2]) 
    inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)
  end

  # Cycle through the active set
  for (i,j) in zip(active_idx[1], active_idx[2]) 
    inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)
  end
end


"""
    update_cd_active_random!(B, resid, X, Z, norms, lambda, reg, nonreg_idx, 
                             active_idx)

Randomly updates active set of coefficients in place for each coordinate 
descent iteration.

# Arguments 

- B = 2d array of floats consisting of coefficient estimates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- norms = 2d array of floats consisting of the norms corresponding to each 
  coefficient or `nothing`
- lambda = lambda penalty, a floating scalar
- reg = 2d array of bits, indicating whether or not to regularize each of the 
  coefficients
- nonreg_idx = tuple with the 2d indices of the non-regularized coefficients 
  as two 1d arrays of integers
- active_idx = tuple with the 2d indices of the active coefficients as two 1d 
  arrays of integers

# Value

None; updates coefficients in place

"""
function update_cd_active_random!(B::AbstractArray{Float64,2}, 
                                  resid::AbstractArray{Float64,2},
                                  X::AbstractArray{Float64,2}, 
                                  Z::AbstractArray{Float64,2}, 
                                  norms, lambda::Float64, reg::BitArray{2}, 
                                  nonreg_idx::Tuple{AbstractArray{Int64,1},
                                                    AbstractArray{Int64,1}}, 
                                  active_idx::Tuple{AbstractArray{Int64,1},
                                                    AbstractArray{Int64,1}})

    # Randomly shuffle the coefficient indices for non-regularized 
    # coefficients and those in the active set
    rand_idx = Random.shuffle(collect(zip(vcat(nonreg_idx[1], active_idx[1]), 
                                          vcat(nonreg_idx[2], active_idx[2]))))

    # Update the non-regularized coefficients and active set randomly
    for (i, j) in rand_idx 
        inner_update_cd!(i, j, B, resid, X, Z, norms, lambda, reg)
    end
end


"""
    cd!(X, Y, Z, lambda, B, regXidx, regZidx, reg, norms; 
        isVerbose, stepsize, isRandom, thresh, maxiter)

Performs coordinate descent using either random or cyclic updates. Does NOT 
take advantage of the active set; see `cd_active!`. 

# Arguments

- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- lambda = lambda penalty, a floating scalar
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
- stepsize = float; step size for updates 
  (irrelevant for coordinate descent). Defaults to `0.01`. 
- isRandom = boolean flag indicating whether to use random or cyclic updates. 
  Defaults to `true`. 
- thresh = threshold at which the coefficients are considered to have 
  converged, a floating scalar. Defaults to `10^(-7)`. 
- maxiter = maximum number of update iterations. Defaults to `10^10`. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

"""
function cd!(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
             Z::AbstractArray{Float64,2}, lambda::Float64, 
             B::AbstractArray{Float64,2}, 
             regXidx::AbstractArray{Int64,1}, 
             regZidx::AbstractArray{Int64,1}, reg::BitArray{2}, norms; 
             isVerbose::Bool=true, stepsize::Float64=0.01, 
             isRandom::Bool=true, thresh::Float64=10.0^(-7), 
             maxiter::Int=10^10)

    # Set the updates to random or cyclic. 
    if isRandom == true
        update! = update_cd_random!
        update_active! = update_cd_active_random!
    else
        update! = update_cd_cyclic!
        update_active! = update_cd_active_cyclic!
    end
    
    # Calculate residuals 
    resid = calc_resid(X, Y, Z, B)
    # Denominators of criterion
    crit_denom = [size(X,1)*size(Z,1), size(X,2)*size(Z,2)] 
    # Placeholder to store the old criterion 
    oldcrit = 1.0 
    # Calculate the current criterion
    crit = criterion(B[regXidx, regZidx], resid, lambda, crit_denom) 

    iter = 0
    # Iterate until coefficients converge or maximum iterations have been 
    # reached.
    while (abs(log(crit/oldcrit)) > thresh) && (iter < maxiter)
        # Store the current criterion
        oldcrit = crit 
        # Update the coefficient estimates
        update!(B, resid, X, Z, norms, lambda, reg) 
        # Calculate the criterion after updating
        crit = criterion(B[regXidx, regZidx], resid, lambda, crit_denom) 

        iter += 1 # Increment the number of iterations
        # Print warning message if coefficient estimates do not converge.
        if iter==(maxiter) 
            println("Estimates did not converge.")
        end
    end

    println_verbose(string("Criterion: ", crit), isVerbose)
    println_verbose(string("Number of iterations: ", iter), isVerbose)
end


"""
    cd_active!(X, Y, Z, lambda, B, regXidx, regZidx, reg, norms; 
               isVerbose, stepsize, isRandom, thresh, maxiter)

Performs coordinate descent, taking advantage of the active set, using either 
random or cyclic updates. 

# Arguments

- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- lambda = lambda penalty, a floating scalar
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
- stepsize = float; step size for updates 
  (irrelevant for coordinate descent). Defaults to `0.01`. 
- isRandom = boolean flag indicating whether to use random or cyclic updates. 
  Defaults to `true`. 
- thresh = threshold at which the coefficients are considered to have 
  converged, a floating scalar. Defaults to `10^(-7)`. 
- maxiter = maximum number of update iterations. Defaults to `10^10`. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

"""
function cd_active!(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
                    Z::AbstractArray{Float64,2}, lambda::Float64, 
                    B::AbstractArray{Float64,2}, 
                    regXidx::AbstractArray{Int64,1}, 
                    regZidx::AbstractArray{Int64,1}, reg::BitArray{2}, norms; 
                    isVerbose::Bool=true, stepsize::Float64=0.01, 
                    isRandom::Bool=true, thresh::Float64=10.0^(-7), 
                    maxiter::Int=10^10)

    # Set the updates to random or cyclic. 
    if isRandom == true
        update! = update_cd_random!
        update_active! = update_cd_active_random!
    else
        update! = update_cd_cyclic!
        update_active! = update_cd_active_cyclic!
    end

    # Calculate residuals 
    resid = calc_resid(X, Y, Z, B)
    # Denominators of criterion
    crit_denom = [size(X,1)*size(Z,1), size(X,2)*size(Z,2)] 
    # Placeholder to store the old criterion 
    oldcrit = 1.0 

    # Initial iteration of updates
    iter = 1 
    # Update the coefficient estimates
    update!(B, resid, X, Z, norms, lambda, reg) 
    # Calculate the criterion after the first update
    crit = criterion(B[regXidx, regZidx], resid, lambda, crit_denom) 
    
    # Indices of non-regularized coefficients
    nonreg_cart_idx = findall(reg.==false)
    nonreg_idx = Tuple([map(i->i[1], nonreg_cart_idx), 
                           map(i->i[2], nonreg_cart_idx)])
    # Identify active set
    active_cart_idx = findall((B.!=0.0) .& (reg.==true))
    active_idx = Tuple([map(i->i[1], active_cart_idx), 
                           map(i->i[2], active_cart_idx)])

    # Iterate until coefficients converge or maximum iterations have been 
    # reached.
    while (abs(log(crit/oldcrit)) > thresh) && (iter < maxiter)
        println_verbose(string("Iterating through active set at iteration ", 
                               iter, "."), isVerbose)
        # Iterate over active set
        while (abs(log(crit/oldcrit)) > thresh) && (iter < maxiter)
            # Store the current criterion
            oldcrit = crit 
            # Update the non-regularized and active coefficients
            update_active!(B, resid, X, Z, norms, lambda, reg, nonreg_idx, 
                           active_idx) 
            # Calculate the criterion after updating
            crit = criterion(B[regXidx, regZidx], resid, lambda, crit_denom) 
            
            iter += 1 # Increment the number of iterations
            # Warning message if coefficient estimates do not converge.
            if iter==(maxiter) 
                println("Estimates did not converge.")
            end
        end
        
        # Store the current criterion
        oldcrit = crit 
        # Update the coefficient estimates
        update!(B, resid, X, Z, norms, lambda, reg) 
        # Calculate the criterion after updating
        crit = criterion(B[regXidx, regZidx], resid, lambda, crit_denom) 
        
        # Re-identify active set
        active_cart_idx = findall((B.!=0.0) .& (reg.==true))
        active_idx = Tuple([map(i->i[1], active_cart_idx), 
                           map(i->i[2], active_cart_idx)])
        iter += 1 # Increment the number of iterations
        # Warning message if coefficient estimates do not converge.
        if iter==(maxiter) 
            println("Estimates did not converge.")
        end
    end

    println_verbose(string("Criterion: ", crit), isVerbose)
    println_verbose(string("Number of iterations: ", iter), isVerbose)
end
