module MatrixLMnet2


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
    istaNet!, fistaNet!, fistaNet_bt!, admmNet!,
    Mlmnet, mlmnet, MlmnetNet, mlmnetNet, coef, coef_2d, predict, fitted, resid, 
    mlmnet_perms, 
    make_folds, make_folds_conds, Mlmnet_cv, mlmnet_cv, MlmnetNet_cv, mlmnetNet_cv,
    avg_mse, lambda_min, lambdaNet_min, avg_prop_zero, mlmnet_cv_summary, mlmnetNet_cv_summary


# Helper functions
include("std_helpers.jl")
include("mlmnet_helpers.jl")

# L1 algorithms
include("cd.jl")
include("ista.jl")
include("fista.jl")
include("fista_bt.jl")
include("admm.jl")

# Elastic-net algorithms
include("istaNet.jl")
include("fistaNet.jl")
include("fistaNet_bt.jl")
include("admmNet.jl")

# Top level functions that call L1 algorithms using warm starts
include("mlmnet.jl")
include("mlmnetNet.jl")
# Predictions and residuals
include("predict.jl")

# Permutations
include("mlmnet_perms.jl")

# Cross-validation
include("mlmnet_cv_helpers.jl")
include("mlmnet_cv.jl")
include("mlmnet_cv_summary.jl")

# Cross-validation for Elastic-net
include("mlmnetNet_cv.jl")
include("mlmnetNet_cv_summary.jl")


end
