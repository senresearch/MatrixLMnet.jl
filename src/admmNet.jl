"""
    proxNet1(v, lambda)

Proximal operator for the Elastic-net penalisation component updates in ADMM. 

# Arguments 

- v = float; value to update
- lambdaL1 = l1 penalty, a floating scalar
- lambdaL2 = l1 penalty, a floating scalar

# Value 

2d array of floats

"""

function proxNet1(v::Float64, lambda::Float64)

    return max(0.0, abs(v)-lambda) * sign(v)
end

"""
    proxNet2(v, rho, u, l)

Proximal operator for the residual sum of squares component updates in ADMM. 

# Arguments 

- v = float; value to update
- rho = float; parameter that controls ADMM tuning. 
- u = float; corresponding transformed Y
- l = float; corresponding eigenvalue

# Value 

2d array of floats

"""

function proxNet2(v::Float64, rho::Float64, 
               u::Float64, l::Float64)
    
    return (u + rho*v) / (l + rho)
end


"""
    update_admmNet!(B, B0, B2, resid, X, Y, Z, Qx, Qz, U, L, lambdaL1, lambdaL2, 
                 regXidx, regZidx, rho, r, s, tau_incr, tau_decr, mu)

Updates coefficient estimates in place for each ADMM iteration. 

# Arguments

- B = 2d array of floats consisting of coefficient estimates for L1 updates
- B0 = 2d array of floats consisting of coefficient estimates for L2 updates
- B2 = 2d array of floats consisting of coefficient estimates for dual updates
- resid = 2d array of floats consisting of the residuals
- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- Qx = 2d array of floats consisting of the eigenvectors of X
- Qz = 2d array of floats consisting of the eigenvectors of Z
- U = 2d array of floats consisting of the transformed Y matrix
- L = 2d array of floats consisting of the kronecker product of the 
  eigenvalues of X and Z
- lambdaL1 = l1 penalty, a floating scalar
- lambdaL2 = l2 penalty, a floating scalar
- regXidx = 1d array of indices corresponding to regularized X covariates
- regZidx = 1d array of indices corresponding to regularized Z covariates
- rho = float; parameter that controls ADMM tuning. 
- r = 2d array of floats consisting of the primal residuals. 
- s = 2d array of floats consisting of the dual residuals. 
- tau_incr = float; parameter that controls the factor at which rho increases. 
  Defaults to 2.0. 
- tau_decr = float; parameter that controls the factor at which rho decreases. 
  Defaults to 2.0. 
- mu = float; parameter that controls the factor at which the primal and dual 
  residuals should be within each other. Defaults to 10.0. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

`rho` controls ADMM tuning and can be specified by the user. 

"""
function update_admmNet!(B::AbstractArray{Float64,2}, 
                      B0::AbstractArray{Float64,2}, 
                      B2::AbstractArray{Float64,2}, 
                      resid::AbstractArray{Float64,2}, 
                      X::AbstractArray{Float64,2}, 
                      Y::AbstractArray{Float64,2}, 
                      Z::AbstractArray{Float64,2}, 
                      Qx::AbstractArray{Float64,2}, 
                      Qz::AbstractArray{Float64,2}, 
                      U::AbstractArray{Float64,2}, 
                      L::AbstractArray{Float64,2}, 
                      lambdaL1::Float64, lambdaL2::Float64, 
                      regXidx::AbstractArray{Int64,1}, 
                      regZidx::AbstractArray{Int64,1}, 
                      rho::AbstractArray{Float64,1}, 
                      r::AbstractArray{Float64,2}, 
                      s::AbstractArray{Float64,2}, 
                      tau_incr::Float64, tau_decr::Float64, mu::Float64)
    
    # Save a copy of the old values of B
    s .= copy(B) 

    # L2 updates
    B0 .= B .- B2
    # Transform B0
    mul!(B0, transpose(Qx), B0*Qz) 
    # Perform L2 updates and transform B0 back
    mul!(B0, Qx * proxNet2.(B0, rho[1], U, L), transpose(Qz))
    
    # L1 updates
    B .= B0 .+ B2
    B[regXidx,regZidx] .= proxNet1.(B[regXidx,regZidx], lambdaL1/rho[1])/(1+lambdaL2/rho[1])

    # Primal residuals 
    r .= B0 .- B
    # Dual residuals 
    s .-= B

    # Dual updates
    B2 .+= r
  
    # Update residuals
    calc_resid!(resid, X, Y, Z, B) 

    # Update rho according to the relative size of the primal and dual residuals
    # Then re-scale B2 accordingly
    if sqrt(sum(r.^2)) > mu * rho[1] * sqrt(sum(s.^2))
        rho[:] .*= tau_incr
        B2 ./= tau_incr
    elseif rho[1] * sqrt(sum(s.^2)) > mu * sqrt(sum(r.^2))
        rho[:] ./= tau_decr
        B2 .*= tau_decr
    end
end


"""
    admmNet!(X, Y, Z, lambdaL1, lambdaL2, B, regXidx, regZidx, reg, norms, Qx, Qz, U, L; 
          isVerbose, stepsize, rho, setRho, thresh, maxiter, 
          tau_incr, tau_decr, mu)

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
- Qx = 2d array of floats consisting of the eigenvectors of X
- Qz = 2d array of floats consisting of the eigenvectors of Z
- U = 2d array of floats consisting of the transformed Y matrix
- L = 2d array of floats consisting of the kronecker product of the 
  eigenvalues of X and Z

# Keyword arguments

- isVerbose = boolean flag indicating whether or not to print messages.  
  Defaults to `true`. 
- stepsize = float; step size for updates (irrelevant for ADMM). 
  Defaults to `0.01`. 
- rho = float; parameter that controls ADMM tuning. Defaults to `1.0`. 
- setRho = boolean flag indicating whether the ADMM tuning parameter `rho` 
  should be calculated. Defaults to `true`.
- thresh = threshold at which the coefficients are considered to have 
  converged, a floating scalar. Defaults to `10^(-7)`. 
- maxiter = maximum number of update iterations. Defaults to `10^10`. 
- tau_incr = float; parameter that controls the factor at which rho increases. 
  Defaults to 2.0. 
- tau_decr = float; parameter that controls the factor at which rho decreases. 
  Defaults to 2.0. 
- mu = float; parameter that controls the factor at which the primal and dual 
  residuals should be within each other. Defaults to 10.0. 

# Value

None; updates coefficients in place

# Some notes

Convergence is determined as when the log ratio of the current and previous 
criteria is less than the threshold `thresh`. 

`rho` controls ADMM tuning and can be specified by the user. 

"""
function admmNet!(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
               Z::AbstractArray{Float64,2}, lambdaL1::Float64, lambdaL2::Float64,
               B::AbstractArray{Float64,2}, 
               regXidx::AbstractArray{Int64,1}, 
               regZidx::AbstractArray{Int64,1}, reg::BitArray{2}, norms, 
               Qx::AbstractArray{Float64,2}, Qz::AbstractArray{Float64,2}, 
               U::AbstractArray{Float64,2}, L::AbstractArray{Float64,2}; 
               isVerbose::Bool=true, stepsize::Float64=0.01, 
               rho::Float64=1.0, setRho::Bool=true, 
               thresh::Float64=10.0^(-7), maxiter::Int=10^10, 
               tau_incr::Float64=2.0, tau_decr::Float64=2.0, mu::Float64=10.0)
    
    # Set the ADMM tuning parameter, rho
    if setRho == true 
        # Get smallest and largest eigenvalues
        mineig = minimum(L)
        maxeig = maximum(L)
        
        gamma = lambdaL1/sqrt(1+lambdaL2)
        # Set the value of rho
        if gamma < mineig
            rho = mineig
        elseif gamma > maxeig
            rho = gamma
        else
            rho = maxeig
        end
    end
    rho = [rho]

    # Calculate residuals 
    resid = calc_resid(X, Y, Z, B)
    # Initialize values for L2 updates
    B0 = copy(B)
    # Initialize values for dual updates
    B2 = copy(B) 
  
    # Initial primal residuals
    r = copy(B)
    # Initialize dual residuals
    s = copy(B)
  
    # Denominators of criterion
    crit_denom = [size(X,1)*size(Z,1), size(X,2)*size(Z,2)] 
    # Placeholder to store the old criterion 
    oldcrit = 1.0 
    # Calculate the current criterion
    crit = criterionNet(B[regXidx, regZidx], resid, lambdaL1, lambdaL2, crit_denom) 
    
    iter = 0
    # Iterate until coefficients converge or maximum iterations have been 
    # reached.
    while (abs(log(crit/oldcrit)) > thresh) && (iter < maxiter)
        # Store the current criterion
        oldcrit = crit 

        # Update the coefficient estimates
        update_admmNet!(B, B0, B2, resid, X, Y, Z, Qx, Qz, U, L, lambdaL1, lambdaL2,
                     regXidx, regZidx, rho, r, s, tau_incr, tau_incr, mu)

        # Calculate the criterion after updating
        crit = criterionNet(B[regXidx, regZidx], resid, lambdaL1, lambdaL2, crit_denom) 

        iter += 1 # Increment the number of iterations
        # Print warning message if coefficient estimates do not converge.
        if iter==(maxiter) 
            println("Estimates did not converge.")
        end
    end

    println_verbose(string("Criterion: ", crit), isVerbose)
    println_verbose(string("Number of iterations: ", iter), isVerbose)
end
