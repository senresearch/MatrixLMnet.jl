"""
    prox1(V, lambda)

Proximal operator for the L1 norm updates in ADMM. 

# Arguments 

- V = 2d array of floats consisting of the values to update
- lambda = lambda penalty, a floating scalar

# Value 

2d array of floats

"""
function prox1(V::Array{Float64,2}, lambda::Float64)

    return max.(0.0, abs.(V).-lambda) .* sign.(V)
end    


"""
    prox1(V, lambda, Qx, Qz)

Proximal operator for the L1 norm updates in ADMM when Qx and Qz are 
orthogonal matrices. 

# Arguments 

- V = 2d array of floats consisting of the values to update
- lambda = lambda penalty, a floating scalar
- Qx = 2d array of floats consisting of the eigenvectors of X
- Qz = 2d array of floats consisting of the eeigenvectors of Z

# Value 

2d array of floats

"""
function prox1(V::Array{Float64,2}, lambda::Float64,
               Qx::Array{Float64,2}, Qz::Array{Float64,2})
    
    return transpose(Qx) * prox1(Qx*V*transpose(Qz), lambda) * Qz
end    


"""
    prox2(V, rho, U, L)

Proximal operator for the L2 norm updates in ADMM. 

# Arguments 

- V = 2d array of floats consisting of the values to update
- rho = float; parameter thata controls ADMM tuning. 
- U = 2d array of floats consisting of the transformed Y matrix
- L = 2d array of floats consisting of the kronecker product of the 
  eigenvalues of X and Z

# Value 

2d array of floats

"""
function prox2(V::Array{Float64,2}, rho::Float64, 
               U::Array{Float64,2}, L::Array{Float64,2})

    return (V .+ rho.*U) ./ (1.0 .+ rho.*L)
end


"""
    update_admm!(B, B0, B2, QxBQz, resid, X, Y, Z, Qx, Qz, U, L, 
                 lambda, regXidx, regZidx, rho)

Updates coefficient estimates in place for each ADMM iteration. 

# Arguments

- B = 2d array of floats consisting of coefficient estimates for L1 updates
- B0 = 2d array of floats consisting of coefficient estimates for L2 updates
- B2 = 2d array of floats consisting of coefficient estimates for dual updates
- QxBQz = 2d array of floats consisting of coefficient estimates transformed 
  to original space
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- Qx = 2d array of floats consisting of the eigenvectors of X
- Qz = 2d array of floats consisting of the eeigenvectors of Z
- U = 2d array of floats consisting of the transformed Y matrix
- L = 2d array of floats consisting of the kronecker product of the 
  eigenvalues of X and Z
- lambda = lambda penalty, a floating scalar
- regXidx = 1d array of indices corresponding to regularized X covariates
- regZidx = 1d array of indices corresponding to regularized Z covariates
- rho = float; parameter thata controls ADMM tuning. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

`rho` controls ADMM tuning and can be specified by the user. 

"""
function update_admm!(B::AbstractArray{Float64,2}, 
                      B0::AbstractArray{Float64,2}, 
                      B2::AbstractArray{Float64,2}, 
                      QxBQz::AbstractArray{Float64,2}, 
                      resid::AbstractArray{Float64,2}, 
                      X::AbstractArray{Float64,2}, 
                      Y::AbstractArray{Float64,2}, 
                      Z::AbstractArray{Float64,2}, 
                      Qx::AbstractArray{Float64,2}, 
                      Qz::AbstractArray{Float64,2}, 
                      U::AbstractArray{Float64,2}, 
                      L::AbstractArray{Float64,2}, 
                      lambda::Float64, 
                      regXidx::AbstractArray{Int64,1}, 
                      regZidx::AbstractArray{Int64,1}, rho::Float64)
    
    # L2 updates
    B0 .= B .- B2
    B0 = prox2(B0, rho, U, L)
    
    # L1 updates
    B .= B0 .+ B2
    B[regXidx,regZidx] = prox1(B, rho*lambda, Qx, Qz)[regXidx,regZidx]
    
    # Dual updates
    B2 .+= B0 .- B
    
    # Transform coefficients back to original space
    LinearAlgebra.mul!(QxBQz, Qx*B, transpose(Qz)) 
    # Update residuals
    calc_resid!(resid, X, Y, Z, QxBQz) 
end


"""
    admm!(X, Y, Z, lambda, B, regXidx, regZidx, reg, norms; 
          isVerbose, stepsize, rho, thresh, maxiter)

Performs ADMM.  

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
- stepsize = float; step size for updates (irrelevant for ADMM). 
  Defaults to `0.01`. 
- rho = float; parameter thata controls ADMM tuning. Defaults to `1.0`. 
- thresh = threshold at which the coefficients are considered to have 
  converged, a floating scalar. Defaults to `10^(-7)`. 
- maxiter = maximum number of update iterations. Defaults to `10^10`. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

`rho` controls ADMM tuning and can be specified by the user. 

"""
function admm!(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
               Z::AbstractArray{Float64,2}, lambda::Float64, 
               B::AbstractArray{Float64,2}, 
               regXidx::AbstractArray{Int64,1}, 
               regZidx::AbstractArray{Int64,1}, reg::BitArray{2}, norms; 
               isVerbose::Bool=true, stepsize::Float64=0.01, rho::Float64=1.0, 
               thresh::Float64=10.0^(-7), maxiter::Int=10^10)
    
    # Eigenfactorization of X
    eigX = eigen(transpose(X)*X)
    Qx = eigX.vectors
    Lx = eigX.values
    
    # Eigenfactorization of Z
    eigZ = eigen(transpose(Z)*Z)
    Qz = eigZ.vectors
    Lz = eigZ.values
    
    # Transformed Y
    X1 = X*Qx
    Z1 = Z*Qz
    U = transpose(X1) * Y * Z1
    # Kronecker product of eigenvalues of X and Z
    L = kron(Lx, transpose(Lz))
    
    # Calculate residuals 
    resid = resid = calc_resid(X, Y, Z, B)
    # Store coefficients from previous iteration ??
    B0 = prox2(B, rho, U, L) # copy(B) 
    # Store extrapolated coefficients ??
    B2 = B0 .- B # copy(B) 
    
    # Transform coefficients back to original space
    QxBQz = Qx * B* transpose(Qz)
    
    # Denominators of criterion
    crit_denom = [size(X,1)*size(Z,1), size(X,2)*size(Z,2)] 
    # Placeholder to store the old criterion 
    oldcrit = 1.0 
    # Calculate the current criterion
    crit = criterion(QxBQz[regXidx, regZidx], resid, lambda, crit_denom) 
    
    iter = 0
    # Iterate until coefficients converge or maximum iterations have been 
    # reached.
    while (abs(log(crit/oldcrit)) > thresh) && (iter < maxiter)
        # Store the current criterion
        oldcrit = crit 

        # Update the coefficient estimates
        update_admm!(B, B0, B2, QxBQz, resid, X, Y, Z, Qx, Qz, U, L, lambda, 
                     regXidx, regZidx, rho)

        # Calculate the criterion after updating
        crit = criterion(QxBQz[regXidx, regZidx], resid, lambda, crit_denom) 

        iter += 1 # Increment the number of iterations
        # Print warning message if coefficient estimates do not converge.
        if iter==(maxiter) 
            println("Estimates did not converge.")
        end
    end

    # Transform coefficients back to original space
    B .= QxBQz

    println_verbose(string("Criterion: ", crit), isVerbose)
    println_verbose(string("Number of iterations: ", iter), isVerbose)
end