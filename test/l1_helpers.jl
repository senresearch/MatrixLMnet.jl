#=
Description:
-----------

'lasso_helpers.jl' sources the binaries files of the lasso 'MatrixLMnet' package version 0.1.0.
This script is used to test the Elastic Net version of the package with 𝛼 = 1, bycomparing its results
with lasso's results of the version 0.1.0.
=#

###########
# Library #
###########
# using MatrixLM
####################
# External sources #
####################
# include("sim_helpers.jl")

# Get list of the sources files containing the functions of the lasso version
dirLasso = realpath(joinpath(@__DIR__, "lasso"))
filesList = readdir(dirLasso)
filesLasso = joinpath.(dirLasso, filesList)
include.(filesLasso)
