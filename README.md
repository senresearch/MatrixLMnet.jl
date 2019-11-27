# matrixLMnet

Core functions to obtain L<sub>1</sub>-penalized estimates for matrix linear models. 

`matrixLMnet` is an extension of the [`matrixLM`](https://github.com/janewliang/matrixLM.jl) package, which provides core functions for closed-form least squares estimates for matrix linear models. 

## Installation 

The `matrixLMnet` package can be installed by running: 

```
using Pkg
# Install matrixLM dependency first
Pkg.add(PackageSpec(url="https://github.com/janewliang/matrixLM.jl", rev="master")) 
Pkg.add(PackageSpec(url="https://github.com/janewliang/matrixLMnet.jl", rev="master"))
```

`matrixLMnet` was developed in [Julia v1.3](https://julialang.org/downloads/). 

## Usage 

```
using matrixLMnet
```

First, construct a `RawData` object consisting of the response variable `Y` and row/column covariates `X` and `Z`. All three matrices must be passed in as 2-dimensional arrays. Note that the `contr` function can be used to set up treatment and/or sum contrasts for categorical variables stored in a DataFrame. By default, `contr` generates treatment contrasts for all specified categorical variables (`"treat"`). Other options include `"sum"` for sum contrasts, `"noint"` for treatment contrasts with no intercept, and `"sumnoint"` for sum contrasts with no intercept. 

```
using DataFrames
using Random

# Dimensions of matrices 
n = 100
m = 250
# Number of column covariates
q = 20

# Randomly generate an X matrix of row covariates with 2 categorical variables
# and 4 continuous variables
Random.seed!(1)
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n)), 
            DataFrame(rand(n,4)))
# Use the contr function to get contrasts for the two categorical variables 
# (treatment contrasts for catvar1 and sum contrasts for catvar2).
# contr returns a DataFrame, so X needs to be converted into a 2d array.
X = convert(Array{Float64,2}, contr(X_df, [:catvar1, :catvar2], 
                                    ["treat", "sum"]))
# Number of row covariates
p = size(X)[2]

# Randomly generate some data for column covariates Z and response variable Y
Z = rand(m,q)
B = rand(vcat(-5:5, zeros(19)),p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E

# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z))
```

Create a 1d array of lambda penalty values to fit the estimates. If the lambdas are not in descending order, they will be sorted by `mlmnet`. 

```
lambdas = reverse(1.8.^(1:10))
```

L<sub>1</sub>-penalized estimates for matrix linear models can be obtained by running `mlmnet`. In addition to the `RawData` object and `lambdas`, `mlmnet` requires as an argument the function name for an algorithm used to fit L<sub>1</sub>-penalized estimates. Options are currently:`cd!` (coordinate descent), `cd_active!` (active coordinate descent), `ista!` (ISTA with fixed step size), `fista!` (FISTA with fixed step size), `fista_bt!` (FISTA with backtracking), and `admm!` (ADMM). 

An object of type `Mlmnet` will be returned, with variables for the penalized coefficient estimates (`B`) and the lambda penalty values used (`lambdas`). By default, `mlmnet` estimates both row and column main effects (X and Z intercepts), but this behavior can be suppressed by setting `isXIntercept=false` and/or `isZntercept=false`; the intercepts will be regularized unless `isXInterceptReg=false` and/or `isZInterceptReg=false`. Individual `X` (row) and `Z` (column) effects can be left unregularized by manually passing in 1d boolean arrays of length `p` and `q` to indicate which effects should be regularized (`true`) or not (`false`) into `isXReg` and `isZreg`. By default, `mlmnet` standardizes the columns of `X` and `Z` to have mean 0 and standard deviation 1 (`isStandardize=true`). Additional keyword arguments include `isVerbose`, which controls message printing; `thresh`, the threshold at which the coefficients are considered to have converged; and `maxiter`, the maximum number of iterations. 

```
est = mlmnet(fista_bt!, dat, lambdas)
```

The functions for the algorithms used to fit the L<sub>1</sub>-penalized estimates have keyword arguments that can be passed into `mlmnet` when non-default behavior is desired. Irrelevant keyword arguments will be ignored. 

`cd!` (coordinate descent)
- `isRandom= true`: Bool; whether to use random or cyclic updates

`cd_active` (active coordinate descent)
- `isRandom = true`: Bool; whether to use random or cyclic updates

`ista!` (ISTA with fixed step size)
- `stepsize = 0.01`: Float64; fixed step size for updates
- `setStepsize = true`: Bool; whether the fixed step size should be calculated, overriding `stepsize`

`fista!` (FISTA with fixed step size)
- `stepsize = 0.01`: Float64; fixed step size for updates
- `setStepsize = true`: Bool; whether the fixed step size should be calculated, overriding `stepsize`

`fista_bt!` (FISTA with backtracking)
- `stepsize = 0.01`: Float64; initial step size for updates
- `gamma = 0.5`: Float64; multiplying factor for step size backtracking/line search

`admm!` (ADMM)
- `rho = 1.0`: Float64; parameter that controls ADMM tuning
- `setRho = true`: Float64; whether the ADMM tuning parameter should be calculated, overriding `rho`
- `tau_incr = 2.0`: Float64; parameter that controls the factor at which the ADMM tuning parameter increases
- `tau_decr = 2.0`: Float64; parameter that controls the factor at which the ADMM tuning parameter decreases
- `mu = 10.0`: Float64; parameter that controls the factor at which the primal and dual residuals should be within each other

The 3d array of coefficient estimates can be accessed using `coef(est)`. Predicted values and residuals can be obtained by calling `predict` and `resid`. By default, both of these functions use the same data used to fit the model. However, a new `Predictors` object can be passed into `predict` as the `newPredictors` argument and a new `RawData` object can be passed into `resid` as the `newData` argument. For convenience, `fitted(est)` will return the fitted values by calling `predict` with the default arguments. 

```
preds = predict(est)
resids = resid(est)
```

All four of these functions take an optional `lambda` argument, in which case only the 2d array corresponding to that value of lambda will be returned, e.g. `coef(est, lambdas[1])`. (If a lambda value that was not used in the fitting of the `Mlmnet` object is passed in, an error will be rasied.) One can also extract coefficients as a flattened 2d array by calling `coef_2d(est)`, for convenience when writing the results to flat files. 

`mlmnet_perms` permutes the response matrix `Y` in a `RawData` object and then calls `mlmnet`. By default, the function used to permute `Y` is `shuffle_rows`, which shuffles the rows for `Y`. Alternative functions for permuting `Y`, such as `shuffle_cols`, can be passed into the argument `permFun`. Non-default behavior for `mlmnet` can be specified by passing its keyword arguments into `mlmnet_perms`. 

```
estPerms = mlmnet_perms(fista_bt!, dat, lambdas)
```

Cross-validation for `mlmnet` can be run using `mlmnet_cv`. The user can either manually specify the row/column folds of `Y` as a 1d array of 1d arrays of row indices, or specify the number of folds that should be used. If the number of folds is specified, disjoint folds of approximately equal size will be generated from a call to `make_folds`. Passing in `1` for the number of row (or column) folds indicates that all of the rows (or columns) of `Y` should be used in each fold. The advantage of manually passing in the row and/or column folds is that it allows the user to stratify or otherwise control the nature of the folds. For example, `make_folds_conds` will generate folds for a set of categorical conditions and ensure that each condition is represented in each fold. Cross validation is computed in parallel when possible. Non-default behavior for `mlmnet` can be specified by passing its keyword arguments into `mlmnet_cv`. 

In the call below, `mlmnet_cv` generates 10 disjoint row folds but uses all columns of `Y` in each fold (indicated by the `1`). The function returns an `Mlmnet_cv` object, which contains an array of the Mlmnet objects for each fold (`MLMNets`); the lambda penalty values used (`lambdas`); the row and column folds (`rowFolds` and `colFolds`); an array of the mean-squared error for each fold (`mse`); and an array of the proportion of zero interaction effects for each fold (`propZero`). The keyword argument `dig` in `mlmnet_cv` adjusts the level of precision when calculating the percent of zero coefficients. It defaults to `12`. 

```
Random.seed!(120)
estCVObjs = mlmnet_cv(fista_bt!, dat, lambdas, 10, 1)
```

`mlmnet_cv_summary` displays a table of the average mean-squared error and proportion of zero coefficients across the folds for each value of lambda. The optimal lambda might be the one that minimizes the mean-squared error (MSE), or can be chosen based on a pre-determined proportion of zeros that is desired in the coefficient estimates. 

```
println(mlmnet_cv_summary(estCVObjs))
```

The `lambda_min` function returns the summary information for the lambdas that correspond to the minimum average test MSE across folds and the MSE that is one standard error greater. 

```
lambda_min(estCVObjs)
```

Additional details can be found in the documentation for specific functions. 
