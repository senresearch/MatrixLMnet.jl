# MatrixLMnet: Core functions for penalized estimation for matrix linear models.

[![CI](https://github.com/senresearch/MatrixLMnet.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/senresearch/MatrixLMnet.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/senresearch/MatrixLMnet.jl/branch/main/graph/badge.svg?token=uHM6utUQoi)](https://codecov.io/gh/GregFa/MatrixLMnet.jl)

Package for $L_1$ and $L_2$ penalized estimation of
 matrix linear models (bilinear models for matrix-valued data).

`MatrixLMnet` depends on the [`MatrixLM`](https://github.com/senresearch/MatrixLM.jl) package, 
which provides core functions for closed-form least squares estimates for matrix linear models. 

See the paper, ["Sparse matrix linear models for structured high-throughput data"](https://arxiv.org/abs/1712.05767), and its [reproducible code](https://github.com/senresearch/mlm_l1_supplement) for details on the L1 penalized estimation.

## Installation 

The `MatrixLMnet` package can be installed by running: 

```
using Pkg
Pkg.add("MatrixLMnet")
```

For the most recent version, use:
```
using Pkg
Pkg.add(url = "https://github.com/senresearch/MatrixLMnet.jl", rev="main")
```

Alternatively, you can also install `MatrixLMnet` from the julia REPL. Press `]` to enter pkg mode again, and enter the following:

```
add MatrixLMnet
```

## Contributing

We appreciate contributions from users including reporting bugs, fixing
issues, improving performance and adding new features.


## Questions

If you have questions about contributing or using `MatrixLMnet` package, please communicate with authors form github.

## Citing `MatrixLMnet`

If you use `MatrixLMnet` in a scientific publication, please consider citing following paper:

Jane W. Liang. Śaunak Sen. "Sparse matrix linear models for structured high-throughput data." Ann. Appl. Stat. 16 (1) 169 - 192, March 2022. https://doi.org/10.1214/21-AOAS1444

```
@article{10.1214/21-AOAS1444,
author = {Jane W. Liang and Śaunak Sen},
title = {{Sparse matrix linear models for structured high-throughput data}},
volume = {16},
journal = {The Annals of Applied Statistics},
number = {1},
publisher = {Institute of Mathematical Statistics},
pages = {169 -- 192},
keywords = {ADMM, FISTA, gradient descent, Julia, Lasso, proximal gradient algorithms},
year = {2022},
doi = {10.1214/21-AOAS1444},
URL = {https://doi.org/10.1214/21-AOAS1444}
}
```