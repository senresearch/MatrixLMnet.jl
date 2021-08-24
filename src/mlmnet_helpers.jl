"""
    println_verbose(x, isVerbose)

Version of println that only prints when isVerbose flag is true

# Arguments 

- x = something that can be printed
- isVerbose = boolean flag indicating whether or not to print messages. 
  Defaults to `true`. 
	
# Value 

None; prints to console

"""
function println_verbose(x, isVerbose::Bool=true)
    
    if isVerbose == true
        println(x)
    end
end


"""
    criterion(B, resid, lambda, crit_denom)

Calculate the criterion for the L1 penalty

# Arguments 

- B = 2d array of floats consisting of regularized coefficient estimates
- resid = 2d array of floats consisting of the residuals
- lambda = lambda penalty, a floating scalar
- crit_denom = 1d array of 2 integers, the denominators of the criterion

# Value 

A floating scalar

"""
function criterion(B::AbstractArray{Float64,2}, 
                   resid::AbstractArray{Float64,2}, 
                   lambda::Float64, crit_denom::AbstractArray{Int64,1})
    
    return 0.5 * sum(abs2, resid)/crit_denom[1] + 
             lambda * sum(abs, B)/crit_denom[2]
end

"""
    criterionNet(B, resid, lambdaL1, lambdaL2, crit_denom)

Calculate the criterion for the Elastic-net penalty

# Arguments 

- B = 2d array of floats consisting of regularized coefficient estimates
- resid = 2d array of floats consisting of the residuals
- lambdaL1 = l1 penalty, a floating scalar
- lambdaL2 = l2 penalty, a floating scalar
- crit_denom = 1d array of 2 integers, the denominators of the criterion

# Value 

A floating scalar

"""
function criterionNet(B::AbstractArray{Float64,2}, 
                   resid::AbstractArray{Float64,2}, 
                   lambdaL1::Float64, lambdaL2::Float64, crit_denom::AbstractArray{Int64,1})
    
    return 0.5 * sum(abs2, resid)/crit_denom[1] + 
             lambdaL1 * sum(abs, B)/crit_denom[2] + 
             0.5 * lambdaL2 * sum(abs2, B)/crit_denom[2]
end


"""
    prox(b, gradient, b2sign, lambda, norm, stepsize)

Proximal (soft-thresholding) operator when not incorporating the norms 
(norms=1)

# Arguments 

- b = coefficient to update, a float
- gradient = gradient of b, a float
- b2sign = sign of b + stepsize*gradient, a float
- lambda = lambda penalty , a float
- norm = Nothing
- stepsize = step size to multiply updates, a float

# Value 

A floating scalar

"""
function prox(b::Float64, gradient::Float64, b2sign::Float64, 
              lambda::Float64, norm::Nothing, stepsize::Float64)
    
    return max(0.0, b2sign*b - stepsize * (b2sign*gradient + lambda)) * b2sign
end


"""
    prox(b, gradient, b2sign, lambda, norm)

Proximal (soft-thresholding) operator when not incorporating the norms 
(norms=1) and step size is 1

# Arguments 

- b = coefficient to update, a float
- gradient = gradient of b, a float
- b2sign = sign of b + stepsize*gradient, a float
- lambda = lambda penalty , a float
- norm = Nothing

# Value 

A floating scalar

"""
function prox(b::Float64, gradient::Float64, b2sign::Float64, 
              lambda::Float64, norm::Nothing)
    
    return max(0.0, b2sign*b - (b2sign*gradient + lambda)) * b2sign
end


"""
    prox(b, gradient, b2sign, lambda, norm, stepsize)

Proximal (soft-thresholding) operator

# Arguments 

- b = coefficient to update, a float
- gradient = gradient of b, a float
- b2sign = sign of b + stepsize*gradient, a float
- lambda = lambda penalty , a float
- norm = norm corresponding to b, a float
- stepsize = step size to multiply updates, a float

# Value 

A floating scalar

"""
function prox(b::Float64, gradient::Float64, b2sign::Float64, 
              lambda::Float64, norm::Float64, stepsize::Float64)
    
    return max(0.0, b2sign*b - stepsize * (b2sign*gradient + lambda)/norm) *
             b2sign
end


"""
    prox(b, gradient, b2sign, lambda, norm)

Proximal (soft-thresholding) operator when step size is 1

# Arguments 

- b = coefficient to update, a float
- gradient = gradient of b, a float
- b2sign = sign of b + stepsize*gradient, a float
- lambda = lambda penalty , a float
- norm = norm corresponding to b, a float

# Value 

A floating scalar

"""
function prox(b::Float64, gradient::Float64, b2sign::Float64, 
              lambda::Float64, norm::Float64)
    
    return max(0.0, b2sign*b - (b2sign*gradient + lambda)/norm) * b2sign
end


"""
    calc_grad!(grad, X, Z, resid)

Calculate gradient in place

# Arguments 

- gradient = 2d array of floats consisting of the gradient, to be updated in 
  place
- X = 2d array of floats consisting of the row covariates, standardized as 
  necessary
- Z = 2d array of floats consisting of the column covariates, standardized 
  as necessary
- resid = 2d array of floats consisting of the residuals

# Value 

None; updates gradient in place. 

"""
function calc_grad!(grad::AbstractArray{Float64,2}, 
                    X::AbstractArray{Float64,2}, 
                    Z::AbstractArray{Float64,2}, 
                    resid::AbstractArray{Float64,2})
    
    LinearAlgebra.mul!(grad, -transpose(X), resid*Z) 
end


"""
    calc_grad!(Xi, Zj, resid)

Calculate gradient at a single coefficient

# Arguments 

- Xi = 1d array of floats consisting of the row covariates for the 
  coefficient, standardized as necessary
- Zj = 1d array of floats consisting of the column covariates for the 
  coefficient, standardized as necessary
- resid = 2d array of floats consisting of the residuals

# Value 

A floating scalar

"""
function calc_grad(Xi::AbstractArray{Float64,1}, Zj::AbstractArray{Float64,1}, 
                   resid::AbstractArray{Float64,2})
    
    return -sum(transpose(Xi)*resid*Zj)
end
