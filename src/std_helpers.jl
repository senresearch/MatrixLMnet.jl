"""
    standardize!(A, isIntercept)

Standardize the columns of A in place

# Arguments 

- A = 2d array of floats
- isIntercept = boolean flag indicating whether the first column of A is the 
  intercept

# Value 

Standardizes A in place and returns 2d arrays of the column means and L2 
norms of A before standardization. 

"""
function standardize!(A::AbstractArray{Float64,2}, isIntercept::Bool)
    
    # If including intercept, subtract the column means from all other columns
    if isIntercept == true 
        means = mean(A, dims=1)
        A[:,2:end] = A[:,2:end].-transpose(means[2:end]) 
    else # Otherwise, subtract the column means from all columns
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
    backtransform!(B, meansX, meansZ, normsX, normsZ, Y, Xold, Zold)

Back-transform coefficient estimates B in place if X and Z were standardized 
prior to the estimation-- when both X and Z include intercept columns. 

# Arguments 

- B = 2d array of coefficient estimates B
- meansX = 2d array of column means of X, obtained prior to standardizing X
- meansZ = 2d array of column means of Z, obtained prior to standardizing Z
- normsX = 2d array of column norms of X, obtained prior to standardizing X
- normsZ = 2d array of column norms of Z, obtained prior to standardizing Z
- Y = 2d array of response matrix Y
- Xold = 2d array row covariates X prior to standardization
- Zold = 2d array column covariates Z prior to standardization

# Value 

None; back-transforms B in place

"""
function backtransform!(B::AbstractArray{Float64,2}, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2}, 
                        Y::AbstractArray{Float64,2}, 
                        Xold::AbstractArray{Float64,2}, 
                        Zold::AbstractArray{Float64,2})

    # Back transform the X intercepts (row main effects)
    prodX = (meansX[:,2:end]./normsX[:,2:end])*B[2:end, 2:end]
    B[1,2:end] = (B[1,2:end]-vec(prodX))./vec(normsZ[:,2:end])/normsX[1,1]
    
    # Back transform the Z intercepts (column main effects)
    prodZ = B[2:end, 2:end]*transpose(meansZ[:,2:end]./normsZ[:,2:end])
    B[2:end,1] = (B[2:end,1]-prodZ)./transpose(normsX[:,2:end])/normsZ[1,1]
    
    # Back transform the interactions
    B[2:end, 2:end] = B[2:end, 2:end]./transpose(normsX[:,2:end])./
                                       normsZ[:,2:end]
    
    # Re-estimate intercept
    B[1,1] = 0
    B[1,1] = mean(Y-Xold*B*transpose(Zold))
end


"""
    backtransform!(B, isXIntercept, isZIntercept, 
	               meansX, meansZ, normsX, normsZ)

Back-transform coefficient estimates B in place if X and Z were standardized 
prior to the estimation-- when not including intercept columns for either X 
or Z. 

# Arguments 

- B = 2d array of coefficient estimates B
- isXIntercept = boolean flag indicating whether or not to X has an 
  intercept column
- isZIntercept = boolean flag indicating whether or not to Z has an 
  intercept column
- meansX = 2d array of column means of X, obtained prior to standardizing X
- meansZ = 2d array of column means of Z, obtained prior to standardizing Z
- normsX = 2d array of column norms of X, obtained prior to standardizing X
- normsZ = 2d array of column norms of Z, obtained prior to standardizing Z

# Value 

None; back-transforms B in place

"""
function backtransform!(B::AbstractArray{Float64,2}, 
                        isXIntercept::Bool, isZIntercept::Bool, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2})

    # Back transform the X intercepts (row main effects), if necessary
    if isXIntercept == true 
        prodX = (meansX[:,2:end]./normsX[:,2:end])*B[2:end, 2:end]
        B[1,2:end] = (B[1,2:end]-vec(prodX))./vec(normsZ[:,2:end])/normsX[1,1]
    end
    
    # Back transform the Z intercepts (column main effects), if necessary
    if isZIntercept == true 
        prodZ = B[2:end, 2:end]*transpose(meansZ[:,2:end]./normsZ[:,2:end])
        B[2:end,1] = (B[2:end,1]-prodZ)./transpose(normsX[:,2:end])/
                                         normsZ[1,1]
    end
    
    # Back transform the interactions, if necessary
    if (isXIntercept == true) || (isZIntercept == true) 
        B[2:end, 2:end] = B[2:end, 2:end]./transpose(normsX[:,2:end])./
                                           normsZ[:,2:end]
    end
    
    # Back transform the interactions if not including any main effects
    if (isXIntercept == false) && (isZIntercept == false) 
        B = B./transpose(normsX)./normsZ
    end
end


"""
    backtransform!(B, meansX, meansZ, normsX, normsZ, Y, Xold, Zold)

Back-transform coefficient estimates B in place if X and Z were standardized 
prior to the estimation-- when both X and Z include intercept columns. 

# Arguments 

- B = 3d array of coefficient estimates B
- meansX = 2d array of column means of X, obtained prior to standardizing X
- meansZ = 2d array of column means of Z, obtained prior to standardizing Z
- normsX = 2d array of column norms of X, obtained prior to standardizing X
- normsZ = 2d array of column norms of Z, obtained prior to standardizing Z
- Y = 2d array of response matrix Y
- Xold = 2d array row covariates X prior to standardization
- Zold = 2d array column covariates Z prior to standardization

# Value 

None; back-transforms B in place

# Some notes

B is a 3d array in which each coefficient matrix is stored along the first 
dimension. 

"""
function backtransform!(B::AbstractArray{Float64,3}, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2}, 
                        Y::AbstractArray{Float64,2}, 
                        Xold::AbstractArray{Float64,2}, 
                        Zold::AbstractArray{Float64,2})

    # Iterate through the first dimension of B to back-transform each 
	# coefficient matrix. 
    for i in 1:size(B,1)  # issue#10 ✓
        # Back transform the X intercepts (row main effects)
        prodX = (meansX[:,2:end]./normsX[:,2:end])*B[2:end, 2:end,i]
        B[1,2:end,i] = (B[1,2:end,i]-vec(prodX))./vec(normsZ[:,2:end])/
                                                  normsX[1,1]
        
        # Back transform the Z intercepts (column main effects)
        prodZ = B[2:end,2:end,i]*transpose(meansZ[:,2:end]./normsZ[:,2:end])
        B[2:end,1,i] = (B[2:end,1,i]-prodZ)./transpose(normsX[:,2:end])/
                                             normsZ[1,1]
        
        # Back transform the interactions
        B[2:end,2:end,i] = B[2:end,2:end,i]./transpose(normsX[:,2:end])./
                                             normsZ[:,2:end]
        
        # Re-estimate intercept
        B[1,1,i] = 0
        B[1,1,i] = mean(Y-Xold*B[:,:,i]*transpose(Zold))
    end
end


"""
    backtransform!(B, isXIntercept, isZIntercept, 
                   meansX, meansZ, normsX, normsZ)

Back-transform coefficient estimates B in place if X and Z were standardized 
prior to the estimation-- when not including intercept columns for either X 
or Z. 

# Arguments 

- B = 3d array of coefficient estimates
- isXIntercept = boolean flag indicating whether or not to X has an 
  intercept column
- isZIntercept = boolean flag indicating whether or not to Z has an 
  intercept column
- meansX = 2d array of column means of X, obtained prior to standardizing X
- meansZ = 2d array of column means of Z, obtained prior to standardizing Z
- normsX = 2d array of column norms of X, obtained prior to standardizing X
- normsZ = 2d array of column norms of Z, obtained prior to standardizing Z

# Value 

None; back-transforms B in place. 

# Some notes

B is a 3d array in which each coefficient matrix is stored along the first 
dimension. 

"""
function backtransform!(B::AbstractArray{Float64,3}, 
                        isXIntercept::Bool, isZIntercept::Bool, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2})

    # Iterate through the first dimension of B to back-transform each 
	# coefficient matrix. 
    for i in 1:size(B,3)  # issue#10 ✓
        # Back transform the X intercepts (row main effects), if necessary 
        if isXIntercept == true 
            prodX = (meansX[:,2:end]./normsX[:,2:end])*B[2:end,2:end,i]
            B[1,2:end,i] = (B[1,2:end,i]-vec(prodX))./vec(normsZ[:,2:end])/
                                                      normsX[1,1]
        end
        
        # Back transform the Z intercepts (column main effects), if necessary
        if isZIntercept == true 
            prodZ = B[2:end,2:end,i]*transpose(meansZ[:,2:end]./
                    normsZ[:,2:end])
            B[2:end,1,i] = (B[2:end,1,i]-prodZ)./transpose(normsX[:,2:end])/
                                                 normsZ[1,1]
        end
        
        # Back transform the interactions, if necessary
        if (isXIntercept == true) || (isZIntercept == true) 
            B[2:end,2:end,i] = B[2:end,2:end,i]./transpose(normsX[:,2:end])./
                                                 normsZ[:,2:end]
        end
        
        # Back transform the interactions if not including any main effects
        if (isXIntercept == false) && (isZIntercept == false) 
            B[:,:,i] = B[:,:,i]./transpose(normsX)./normsZ
        end
    end
end

"""
    backtransformNet!(B, isXIntercept, isZIntercept, 
                   meansX, meansZ, normsX, normsZ)

Back-transform coefficient estimates B in place if X and Z were standardized 
prior to the estimation-- when not including intercept columns for either X 
or Z. 

# Arguments 

- B = 3d array of coefficient estimates
- isXIntercept = boolean flag indicating whether or not to X has an 
  intercept column
- isZIntercept = boolean flag indicating whether or not to Z has an 
  intercept column
- meansX = 2d array of column means of X, obtained prior to standardizing X
- meansZ = 2d array of column means of Z, obtained prior to standardizing Z
- normsX = 2d array of column norms of X, obtained prior to standardizing X
- normsZ = 2d array of column norms of Z, obtained prior to standardizing Z

# Value 

None; back-transforms B in place. 

# Some notes

B is a 3d array in which each coefficient matrix is stored along the first 
dimension. 

"""
function backtransformNet!(B::AbstractArray{Float64,4}, 
                        isXIntercept::Bool, isZIntercept::Bool, 
                        meansX::AbstractArray{Float64,2}, 
                        meansZ::AbstractArray{Float64,2}, 
                        normsX::AbstractArray{Float64,2}, 
                        normsZ::AbstractArray{Float64,2})

    # Iterate through the first dimension of B to back-transform each 
	# coefficient matrix: 
    for j in 1:size(B,4)  # issue#10 ✓
        for i in 1:size(B,3) 
            # Back transform the X intercepts (row main effects), if necessary 
            if isXIntercept == true 
                prodX = (meansX[:,2:end]./normsX[:,2:end])*B[2:end,2:end,i,j]
                B[1,2:end,i,j] = (B[1,2:end,i,j]-vec(prodX))./vec(normsZ[:,2:end])/
                                                        normsX[1,1]
            end
        
            # Back transform the Z intercepts (column main effects), if necessary
            if isZIntercept == true 
                prodZ = B[2:end,2:end,i,j]*transpose(meansZ[:,2:end]./
                    normsZ[:,2:end])
                B[2:end,1,i,j] = (B[2:end,1,i,j]-prodZ)./transpose(normsX[:,2:end])/
                                                    normsZ[1,1]
            end
        
            # Back transform the interactions, if necessary
            if (isXIntercept == true) || (isZIntercept == true) 
                B[2:end,2:end,i,j] = B[2:end,2:end,i,j]./transpose(normsX[:,2:end])./
                                                    normsZ[:,2:end]
            end
        
            # Back transform the interactions if not including any main effects
            if (isXIntercept == false) && (isZIntercept == false) 
                B[:,:,i,j] = B[:,:,i,j]./transpose(normsX)./normsZ
            end
        end
    end
end