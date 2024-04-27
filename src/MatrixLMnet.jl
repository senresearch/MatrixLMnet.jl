module MatrixLMnet

using Reexport

using MatrixLM
import MatrixLM.calc_preds, MatrixLM.calc_preds!, 
    MatrixLM.calc_resid, MatrixLM.calc_resid!
@reexport using MatrixLM: design_matrix, @mlmformula

using DataFrames
using Distributed
using LinearAlgebra
using LinearAlgebra.BLAS
import LinearAlgebra.mul!, LinearAlgebra.norm, LinearAlgebra.dot, 
     LinearAlgebra.diagm, LinearAlgebra.eigmin, LinearAlgebra.eigmax
using Random
using Statistics
using MLBase


export Response, Predictors, RawData, get_X, get_Z, get_Y, contr, 
    add_intercept, remove_intercept, shuffle_rows, shuffle_cols
    



# Helper functions
include("utilities/std_helpers.jl")
include("mlmnet/mlmnet_helpers.jl")

# Optimizing algorithms
include("methods/cd.jl")
include("methods/ista.jl")
include("methods/fista.jl")
include("methods/fista_bt.jl")
include("methods/admm.jl")
export cd!, cd_active!, ista!, fista!, fista_bt!, admm!

# Top level functions that call Elastic Net algorithms using warm starts
include("mlmnet/mlmnet.jl")
export mlmnet, Mlmnet   

# Predictions and residuals
include("utilities/predict.jl")
export coef, predict, fitted, resid#, coef_2d,

# Permutations
include("mlmnet/mlmnet_perms.jl")
export mlmnet_perms

# Cross-validation
include("crossvalidation/mlmnet_cv_helpers.jl")
export make_folds, make_folds_conds
include("crossvalidation/mlmnet_cv.jl")
export mlmnet_cv, Mlmnet_cv
include("crossvalidation/mlmnet_cv_summary.jl")
export calc_avg_mse, lambda_min, calc_avg_prop_zero, mlmnet_cv_summary 

# BIC validation
include("bic/mlmnet_bic_helpers.jl")
export calc_bic
include("bic/mlmnet_bic.jl")
export mlmnet_bic, Mlmnet_bic
include("bic/mlmnet_bic_summary.jl")
export mlmnet_bic_summary



end
