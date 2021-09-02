
"""
simulateCorrelatedData(ρMat, n; σₙ, σ, μ)

Generate a matrix of random values from a normal distribution, 
where columns are correlated according to the `ρMat` correlation matrix.  

# Arguments
- `ρMat::Matrix{Real}`: coefficients of the correlation matrix
- `n::Int64`: number of samples, default `n` is `100`
- `σₙ::Real`: noise standard deviation, default `σₙ` is `0` 
- `σ::Real`: simulated data standard deviation, default `σ` is `1` 
- `μ::Real`: simulated data mean, default `μ` is `0` 

# Examples
```julia
simulateCorrelatedData([1.0   0.5  0.25;
                        0.5   1.0  0.5 ;
                        0.25  0.5  1.0])
```
"""
     
function simulateCorrelatedData(ρMat::Matrix{Float64}, n::Int64=100; 
                                σₙ::Real=0, σ::Real = 1, μ::Real= 0)

    # Get number of variables
    p = size(ρMat)[2]

    # Whitening transformation of the random samples matrix 
    X = standardize(ZScoreTransform, rand(Normal(0, 1), n, p), dims = 1)
    X = X / LinearAlgebra.cholesky(cov(X)).U
  
    # Apply correlation matrix
    X = X * LinearAlgebra.cholesky(ρMat).U
    X = X *  σ .+ μ .+ rand(Normal(0, σₙ), n)
  
    return X
    
end