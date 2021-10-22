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
    istaNet!, fistaNet!, fistaNet_bt!, admmNet!, istb!,
    Mlmnet, mlmnet, MlmnetNet, mlmnetNet, coef, coef_2d, predict, fitted, resid, 
    mlmnet_perms, 
    make_folds, make_folds_conds, Mlmnet_cv, mlmnet_cv, MlmnetNet_cv, mlmnetNet_cv,
    avg_mse, lambda_min, lambdaNet_min, avg_prop_zero, mlmnet_cv_summary, mlmnetNet_cv_summary


# Helper functions
include("utilities\std_helpers.jl")
include("mlmnet\mlmnet_helpers.jl")

# L1 algorithms
include("methods\cd.jl")
include("methods\ista.jl")
include("methods\istb.jl")
include("methods\fista.jl")
include("methods\fista_bt.jl")
include("methods\admm.jl")

# Elastic-net algorithms
include("methods\istaNet.jl")
include("methods\fistaNet.jl")
include("methods\fistaNet_bt.jl")
include("methods\admmNet.jl")

# Top level functions that call L1 algorithms using warm starts
include("mlmnet\mlmnet.jl")
include("mlmnet\mlmnetNet.jl")
# Predictions and residuals
include("utilities\predict.jl")

# Permutations
include("mlmnet\mlmnet_perms.jl")

# Cross-validation
include("mlmnet_cv_helpers.jl")
include("mlmnet_cv.jl")
include("mlmnet_cv_summary.jl")

# Cross-validation for Elastic-net
include("mlmnetNet_cv.jl")
include("mlmnetNet_cv_summary.jl")


end
