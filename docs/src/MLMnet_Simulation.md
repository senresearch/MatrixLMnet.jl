## Overview 

`MatrixLMnet.jl` is a comprehensive package for both $L_1$ and $L_2$ penalized estimation in matrix linear models. 
  It offers efficient and versatile methods for fitting sparse matrix linear models, particularly for analyzing structured high-throughput data. In this demonstration, we will explore the functionalities of this package through an easy-to-follow simulation study, showcasing its practical applications and user-friendly features.    
      
Within the scope of the matrix linear model framework, the model is articulated as follows:   
   
$$Y = XBZ^T+E$$   
    
Where     
- ``Y_{n \times m}`` is the response matrix,   
- ``X_{n \times p}`` is the matrix for main predictors,   
- ``Z_{m \times q}`` denote the response attributes matrix based on a supervised knowledge,   
- ``E_{n \times m}`` is the error term,   
- ``B_{p \times q}`` is the matrix for main and interaction effects.,


## Data Generation


```julia
using MatrixLMnet
using DataFrames
using Plots
```

In this example,  we set up a simulation for matrix linear models. 
We generate a dataset containing two categorical variables and four numerical variables.


```julia
# Define number of samples in the predictors dataset
n = 100

# Generate data with one categorical variables and 3 numerical variables.
dfX = hcat(DataFrame(
    catvar1=rand(["A", "B", "C", "D"], n)), 
    DataFrame(rand(n,3), ["numvar1", "numvar2", "numvar3"]));

first(dfX, 3)
```



```@raw html
<div><div style = "float: left;"><span>3×4 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">catvar1</th><th style = "text-align: left;">numvar1</th><th style = "text-align: left;">numvar2</th><th style = "text-align: left;">numvar3</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "String" style = "text-align: left;">String</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">A</td><td style = "text-align: right;">0.0272858</td><td style = "text-align: right;">0.880826</td><td style = "text-align: right;">0.284163</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">B</td><td style = "text-align: right;">0.571666</td><td style = "text-align: right;">0.414586</td><td style = "text-align: right;">0.406994</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">A</td><td style = "text-align: right;">0.151188</td><td style = "text-align: right;">0.680321</td><td style = "text-align: right;">0.424177</td></tr></tbody></table></div>
```


We need to create a design matrix **X** form the dataframe *dfX*. We can use the `design_matrix()` function from the package `MatrixLM.jl`. For more information about its usage and the macro `@mlmformula` please refer to its [documentation](https://senresearch.github.io/MatrixLM.jl/stable/).      
First, we need to select the contrast coding of our categorical variable.


```julia
# Define constrast coding for catvar 
levels_catvar1 = sort(unique(dfX.catvar1));
X_ctrst = Dict(
             :catvar1 => MatrixLMnet.MatrixLM.StatsModels.DummyCoding(levels = levels_catvar1),
           );
```

Let convert the dataframe *dfX* to the predictor matrix **X** using `design_matrix()`:


```julia
X = design_matrix(@mlmformula(1 + catvar1 + numvar1 + numvar2 + numvar3), dfX, X_ctrst);
```

 We also define **Z**, a random matrix with dimensions *m* by *q*, and **B**, a matrix of random integers between -5 and 5 with dimensions matching **X** and **Z**. According the matrix linear models framework, the response matrix **Y** is calculated as the product of **X**, **B**, and the transpose of **Z**, with an added noise matrix **E**.   
 Finally, we construct a `RawData` object named `dat``, which comprises the response matrix **Y** and the predictor matrices **X** and **Z**.


```julia
# Number of predictors
p = size(X)[2]
# Number of column responses
m = 250
# Number of column covariates
q = 20

Z = rand(m,q)
B = rand(-5:5,p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E

# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z));
```

## Model Estimation

### Hyper parameters

Create a 1d array of lambda penalties values to fit the estimates. If the lambdas are not in descending order, they will be automatically sorted by `mlmnet`.


```julia
lambdas = reverse(1.5.^(1:3:10))
```




    4-element Vector{Float64}:
     57.6650390625
     17.0859375
      5.0625
      1.5



Create a 1d array of alpha parameter penalties values that determine the penalties mix between $L_1$ and $L_2$ to fit the estimates according to the Elastic Net penalization method.  In the case of Lasso regression ($L_1$ regularization), alpha should be 1, and 0 for Ridge regression ($L_2$ regularization). If the alphas are not in descending order, they will be automatically sorted by `mlmnet`.


```julia
alphas = reverse(collect(0:0.5:1));
```

If `alphas` argument is omitted, a Lasso regression will be applied which is equivalent to `alphas = [1]`.  

### Elastic Net penalization algorithms selection

The algorithms available for fitting Elastic Net penalized estimates in `mlmnet` function come with customizable keyword arguments for fine-tuning. 
The `method` keyword argument selects the function implementing the Elastic-net penalty estimation method. The default method is **ista**; alternative options include **fista**, **fista_bt**, **admm**, and **cd**.    
*Note: Any irrelevant arguments will simply be disregarded.*

|Algorithm                     | Methods     | Parameter     | Default  | Description                                                                     |
|------------------------------|-------------|---------------|----------|--------------------------------------------------------------------------------:|
|**Coordinate Descent**        | "cd"        | `isRandom`    | true     | Determines the use of either random or cyclic updates                           |
|**Active Coordinate Descent** | "cd_active" | `isRandom`    | true     | Specifies the choice between random and cyclic updates                          |
|**ISTA** with fixed step size | "ista"      | `stepsize`    | 0.01     | Sets a fixed step size for updates                                              |
|                              |             | `setStepsize` | true     | Decides if the fixed step size is to be computed, overriding `stepsize`         |
|**FISTA** with fixed step size| "fista"     | `stepsize`    | 0.01     | Establishes a fixed step size for updates                                       |
|                              |             | `setStepsize` | true     | Determines if the fixed step size should be recalculated, overriding `stepsize` |
|**FISTA** with backtracking   | "fista_bt"  | `stepsize`    | 0.01     | Indicates the initial step size for updates                                     |
|                              |             | `gamma`       | 0.5      | The multiplier for step size during backtracking/line search                    |
|**ADMM**                      | "admm"      | `rho`         | 1.0      | The parameter influencing ADMM tuning                                           |
|                              |             | `setRho`      | true     | Decides whether the ADMM tuning parameter `rho` is to be auto-calculated        |
|                              |             | `tau_incr`    | 2.0      | The factor for increasing the ADMM tuning parameter                             |
|                              |             | `tau_decr`    | 2.0      | The factor for decreasing the ADMM tuning parameter                             |
|                              |             | `mu`          | 10.0     | The parameter influencing the balance between primal and dual residuals         |


### Estimation

In this example, we will use the 'fista' method for our estimation process. 
Given that our design matrix already incorporates an intercept, we specify that there is no need to add an additional intercept to the design matrices **X** and **Z**.


```julia
est_fista = mlmnet(
    dat, 
    [lambdas[1]], [alphas[1]], 
    method = "fista", 
    addZIntercept = false, addXIntercept = false, isVerbose = false);
```

Let's delve into the structure returned by the `mlmnet()` function. The result, `est_fista`, is of the `Mlmnet` type, a structured data type encompassing the next key fields:

- `B`: This is the estimated coefficient matrix. It is a four-dimensional matrix where the dimensions represent rows, columns, lambda indices, and alpha indices.
- `lambdas`: A vector containing the lambda values utilized in the estimation.
- `alphas`: A vector of the alpha values used.
- `data`: A `RawData` object that holds the original dataset used for the estimation.

For instance, let execute the command with all lambdas and alphas values:


```julia
est_fista = mlmnet(
    dat, 
    lambdas, alphas, 
    method = "fista", 
    addZIntercept = false, addXIntercept = false, isVerbose = false);
```

In this case, the matrix `B` would have dimensions of (7, 20, 4, 3). These dimensions correspond to the estimated coefficients across each of the 4 lambda values and 3 alpha values, offering a comprehensive view of the coefficient variations influenced by different regularization parameters.

### Cross Validation 

Cross-validation within the `mlmnet` framework is implemented by the `mlmnet_cv` function. This function offers flexibility in defining the folds for cross-validation. We can either specify the row and column folds of `Y` manually as 1D arrays containing row or column indices, or we can simply specify the desired number of folds. When a specific number of folds is indicated, `mlmnet_cv` will generate disjoint folds of approximately equal size using the `make_folds` function. Setting the number of row or column folds to `1` implies the use of all rows or columns of `Y` in each fold.

One key benefit of manually inputting row and/or column folds is the user's ability to stratify or control the nature of the folds. For instance, `make_folds_conds` can be employed to create folds based on a set of categorical conditions, ensuring each condition is equally represented across folds. Additionally, cross-validation computations are executed in parallel whenever feasible, enhancing efficiency. To incorporate non-standard behaviors in `mlmnet`, we can pass the relevant keyword arguments directly into `mlmnet_cv`.

In the following call, `mlmnet_cv` is configured to create 10 disjoint row folds while utilizing all columns of `Y` in each fold, as denoted by the 1:


```julia
est_fista_cv = mlmnet_cv(
    dat, 
    lambdas, alphas,
    10, 1, 
    method = "fista",  
    addZIntercept = false, addXIntercept = false, isVerbose = false);
```

*Note: The `dig` keyword argument in `mlmnet_cv` is used to set the precision level for computing the percentage of zero coefficients. Its default value is `12`, allowing for detailed precision in the calculations.*

The function returns an `Mlmnet_cv` object, which contains an array of the Mlmnet objects for each fold (`MLMNets`); the lambda penalty values used (`lambdas`); the row and column folds (`rowFolds` and `colFolds`); an array of the mean-squared error for each fold (`mse`); and an array of the proportion of zero interaction effects for each fold (`propZero`). The keyword argument `dig` in `mlmnet_cv` adjusts the level of precision when calculating the percent of zero coefficients. It defaults to `12`.

The output from the `mlmnet_cv()` function yields an `Mlmnet_cv` object. This object encompasses the following components:

- `MLMNets`: An array consisting of the `Mlmnet` objects corresponding to each fold.
- `lambdas`: The array of lambda penalty values applied during the estimati
- `alphas`: The array of alpha penalty values applied during the estimation.on.
- `rowFolds` and `colFolds`: Arrays representing the row and column folds used in cross-validation.
- `mse`: An array detailing the mean-squared error for each individual fold.
- `propZero`: An array capturing the proportion of zero interaction effects in each ft `12`."

`mlmnet_cv_summary` presents a table summarizing the average mean-squared error (MSE) and the proportion of zero coefficients for each pair of (lambda, alpha) values across all folds. The optimal (lambda, alpha) could be selected as the one that minimizes the MSE. Alternatively, it might be chosen according to a specific, pre-defined proportion of zeros desired in the coefficient estimates.


```julia
mlmnet_cv_summary(est_fista_cv)
```



```@raw html
<div><div style = "float: left;"><span>12×6 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">𝜆_𝛼_Index</th><th style = "text-align: left;">Lambda</th><th style = "text-align: left;">Alpha</th><th style = "text-align: left;">AvgMSE</th><th style = "text-align: left;">StdMSE</th><th style = "text-align: left;">AvgPropZero</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Tuple{Int64, Int64}" style = "text-align: left;">Tuple…</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">(1, 1)</td><td style = "text-align: right;">57.665</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">18.7416</td><td style = "text-align: right;">1.87917</td><td style = "text-align: right;">0.679286</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">(2, 1)</td><td style = "text-align: right;">17.0859</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">6.46943</td><td style = "text-align: right;">0.57586</td><td style = "text-align: right;">0.42</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">(3, 1)</td><td style = "text-align: right;">5.0625</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">2.49412</td><td style = "text-align: right;">0.290205</td><td style = "text-align: right;">0.178571</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">(4, 1)</td><td style = "text-align: right;">1.5</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">1.22069</td><td style = "text-align: right;">0.071585</td><td style = "text-align: right;">0.0671429</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: left;">(1, 2)</td><td style = "text-align: right;">57.665</td><td style = "text-align: right;">0.5</td><td style = "text-align: right;">32006.4</td><td style = "text-align: right;">3798.02</td><td style = "text-align: right;">0.0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">6</td><td style = "text-align: left;">(2, 2)</td><td style = "text-align: right;">17.0859</td><td style = "text-align: right;">0.5</td><td style = "text-align: right;">6609.07</td><td style = "text-align: right;">862.572</td><td style = "text-align: right;">0.0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">7</td><td style = "text-align: left;">(3, 2)</td><td style = "text-align: right;">5.0625</td><td style = "text-align: right;">0.5</td><td style = "text-align: right;">829.012</td><td style = "text-align: right;">123.063</td><td style = "text-align: right;">0.005</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">8</td><td style = "text-align: left;">(4, 2)</td><td style = "text-align: right;">1.5</td><td style = "text-align: right;">0.5</td><td style = "text-align: right;">92.8234</td><td style = "text-align: right;">14.5585</td><td style = "text-align: right;">0.005</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">9</td><td style = "text-align: left;">(1, 3)</td><td style = "text-align: right;">57.665</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">61511.1</td><td style = "text-align: right;">7044.65</td><td style = "text-align: right;">0.0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">10</td><td style = "text-align: left;">(2, 3)</td><td style = "text-align: right;">17.0859</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">18196.8</td><td style = "text-align: right;">2234.07</td><td style = "text-align: right;">0.0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">11</td><td style = "text-align: left;">(3, 3)</td><td style = "text-align: right;">5.0625</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">2839.32</td><td style = "text-align: right;">390.58</td><td style = "text-align: right;">0.0</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">12</td><td style = "text-align: left;">(4, 3)</td><td style = "text-align: right;">1.5</td><td style = "text-align: right;">0.0</td><td style = "text-align: right;">320.418</td><td style = "text-align: right;">49.9818</td><td style = "text-align: right;">0.0</td></tr></tbody></table></div>
```
   
       
       
The `lambda_min` function returns the summary information for the lambdas that correspond to the minimum average test MSE across folds and the MSE that is one standard error greater.


```julia
lambda_min(est_fista_cv)
```


```@raw html
<div><div style = "float: left;"><span>2×6 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">Name</th><th style = "text-align: left;">Index</th><th style = "text-align: left;">Lambda</th><th style = "text-align: left;">Alpha</th><th style = "text-align: left;">AvgMSE</th><th style = "text-align: left;">AvgPropZero</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "String" style = "text-align: left;">String</th><th title = "Tuple{Int64, Int64}" style = "text-align: left;">Tuple…</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">(𝜆, 𝛼)_min</td><td style = "text-align: left;">(4, 1)</td><td style = "text-align: right;">1.5</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">1.22069</td><td style = "text-align: right;">0.0671429</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">(𝜆, 𝛼)_min1se</td><td style = "text-align: left;">(4, 1)</td><td style = "text-align: right;">1.5</td><td style = "text-align: right;">1.0</td><td style = "text-align: right;">1.22069</td><td style = "text-align: right;">0.0671429</td></tr></tbody></table></div>
```

## Model predictions and residuals

The 4D array of coefficient estimates is returned by the function `coef(est)`, where `est` is the results of the function `mlmnet`. 

The command `est_fista.B` or `coef(est_fista)` retrieves the full array containing the estimated coefficients.    
Let display the residual errors heatmap based on our estimation.

To compare the estimated coefficients with the original matrix **B**, we will visualize the matrices using heatmaps. This graphical representation allows us to readily see differences and similarities between the two.


```julia
plot(
    heatmap(B[end:-1:1, :], 
            size = (800, 300)),     
    heatmap(est_fista.B[end:-1:1, :, 4, 1], 
            size = (800, 300)), 
            # clims = (-5, 5)),     
    title = ["\$ \\mathbf{B}\$" "\$ \\mathbf{\\hat{B}}\$"]
)
```

![svg](images/output_B_est.svg)

To obtain predicted values and residuals, one can use `predict` and `resid` respectively. 
By default, these functions use the data from the model fit, but alternative data can be supplied: `newPredictors` (a `Predictors` object) for `predict`, and `newData` (a `RawData` object) for `resid`. For added convenience, `fitted(est)` returns the fitted values by default when calling `predict`.

Let's employ the same visualization method to compare the predicted values with the original **Y** response matrix. This allows us to gauge the accuracy of our model predictions.


```julia
preds = predict(est_fista, lambdas[4], alphas[1]); # Prediction value
```


```julia
plot(
    heatmap(Y[end:-1:1, :], 
            size = (800, 300)),     
    heatmap(preds[end:-1:1, :], 
            size = (800, 300), 
            clims = (-25, 45)
            ),     
    title = ["\$ \\mathbf{Y}\$" "\$ \\mathbf{\\hat{Y}}\$"],
)
```

![svg](images/output_Y_pred.svg)

The `resid()` function provides us with the ability to compute residuals for each observation, helping you evaluate the discrepancy between the model's predictions and the actual data.


```julia
resids= resid(est_fista, lambdas[4], alphas[1]); 
```


```julia
plot(
    heatmap(
        resids[end:-1:1, :], 
        color = cgrad(:bluesreds,[0.1, 0.3, 0.7, 0.9], alpha = 0.8),
        title = "Residual errors",
        titlefontsize = 12, grid = false,
        xlabel = "Responses",
        ylabel = "Samples",
        size = (500, 250),
        left_margin = (5,:mm), bottom_margin = (5,:mm),
    ),
    histogram(
        (reshape(resids,250*100,1)),
        grid  = false,
        label = "",
        size = (800, 300)
    ),     
    title = ["Residuals" "Distribution of the residuals"]
)
```

![svg](images/output_resid_hist.svg)

Additional details can be found in the documentation for specific functions.


