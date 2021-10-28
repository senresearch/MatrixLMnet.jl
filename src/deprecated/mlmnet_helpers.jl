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
