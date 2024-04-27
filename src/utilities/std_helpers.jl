"""
    normalize!(A::AbstractArray{Float64,2}, hasIntercept::Bool)

Centers and normalizes the columns of A in place

# Arguments 

- A = 2d array of floats
- hasIntercept = boolean flag indicating whether the first column of A is the 
  intercept

# Value 

Centers and normalizes A in place and returns 2d arrays of the column means and L2 
norms of A before standardization. 

"""
function normalize!(A::AbstractArray{Float64,2}, hasIntercept::Bool)
    
    # If including intercept, subtract the column means from all other columns
    if hasIntercept == true 
        means = mean(A, dims=1)
        A[:,2:end] = A[:,2:end].-transpose(means[2:end]) 
    else # If no intercept do not center.
        means = Array{Float64}(undef, 1, size(A,2))
    end
    
    # Obtain and divide by the column L2 norms
    norms = Array{Float64}(undef, 1, size(A,2)) 
    for j in 1:size(A,2)
        norms[j] = norm(A[:,j])
        A[:,j] = A[:,j]/norms[j]
    end
    
    # Return the column means and norms
    return means, norms
end

"""
    backtransform!(B::AbstractArray{Float64,4}, 
                        addXIntercept::Bool, addZIntercept::Bool, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2})

Back-transform coefficient estimates B in place if X and Z were standardized 
prior to the estimation-- when not including intercept columns for either X 
or Z. 

# Arguments 

- B = 4d array of coefficient estimates
- addXIntercept = boolean flag indicating whether or not to X has an 
  intercept column
- addZIntercept = boolean flag indicating whether or not to Z has an 
  intercept column
- meansX = 2d array of column means of X, obtained prior to standardizing X
- meansZ = 2d array of column means of Z, obtained prior to standardizing Z
- normsX = 2d array of column norms of X, obtained prior to standardizing X
- normsZ = 2d array of column norms of Z, obtained prior to standardizing Z

# Value 

None; back-transforms B in place. 

# Some notes

B is a 4d array in which each coefficient matrix is stored along the third and fourth 
dimension. 

"""
function backtransform!(B::AbstractArray{Float64,4}, 
                        addXIntercept::Bool, addZIntercept::Bool, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2})

    # Iterate through the first dimension of B to back-transform each 
	# coefficient matrix: 
    for j in 1:size(B,4) 
        for i in 1:size(B,3) 

            # reverse scale from X and Z normalization
            B[:,:,i,j] = (B[:,:,i,j]./permutedims(normsX))./normsZ 

            # Back transform the X intercepts (row main effects)
            if addXIntercept == true
                # B̂intercept = Ȳ - X̄ B̂coef. Since X centered, B̂intercept|Xcentered = Ȳ      
                B[1,:,i,j] = B[1,:,i,j] - vec(meansX[:,2:end]*B[2:end,:,i,j])
            end

            # Back transform the Z intercepts (column main effects)
            if addZIntercept == true
                # B̂intercept = Ȳ - Z̄ B̂coef. Since X centered, B̂intercept|Xcentered = Ȳ      
                B[:,1,i,j] = B[:,1,i,j] - B[:,2:end,i,j]*permutedims(meansZ[:,2:end])
            end
            
        end
    end
end