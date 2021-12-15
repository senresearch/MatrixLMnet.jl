module MatrixLMnet


using MatrixLM
import MatrixLM.calc_preds, MatrixLM.calc_preds!, 
    MatrixLM.calc_resid, MatrixLM.calc_resid!

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
    add_intercept, remove_intercept, shuffle_rows, shuffle_cols,
    cd!, cd_active!, ista!, fista!, fista_bt!, admm!, 
    mlmnet, Mlmnet, coef, coef_2d, predict, fitted, resid, 
    mlmnet_perms, 
    make_folds, make_folds_conds, mlmnet_cv, Mlmnet_cv, mlmnet_cv,
    avg_mse, lambda_min, avg_prop_zero, mlmnet_cv_summary


# Helper functions
include("utilities/std_helpers.jl")
include("mlmnet/mlmnet_helpers.jl")

# Deprecated L1 algorithms
# include("deprecated/ista.jl")
# include("deprecated/fista.jl")
# include("deprecated/fista_bt.jl")
# include("deprecated/admm.jl")
# include("deprecated/mlmnet.jl")
# include("deprecated/mlmnet_helpers.jl")

# include("deprecated/mlmnet_cv.jl")
# include("deprecated/mlmnet_cv_summary.jl")

# Optimizing algorithms
include("methods/cd.jl")
include("methods/ista.jl")
include("methods/fista.jl")
include("methods/fista_bt.jl")
include("methods/admm.jl")

# Top level functions that call L1 algorithms using warm starts
include("mlmnet/mlmnet.jl")

# Predictions and residuals
include("utilities/predict.jl")

# Permutations
include("mlmnet/mlmnet_perms.jl")

# Cross-validation
include("crossvalidation/mlmnet_cv_helpers.jl")
include("crossvalidation/mlmnet_cv.jl")
include("crossvalidation/mlmnet_cv_summary.jl")

end
