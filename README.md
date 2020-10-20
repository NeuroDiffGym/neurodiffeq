# neurodiffeq

[![Build Status](https://travis-ci.org/odegym/neurodiffeq.svg?branch=master)](https://travis-ci.org/odegym/neurodiffeq)
[![codecov](https://codecov.io/gh/odegym/neurodiffeq/branch/master/graph/badge.svg)](https://codecov.io/gh/odegym/neurodiffeq)
[![Documentation Status](https://readthedocs.org/projects/neurodiffeq/badge/?version=latest)](https://neurodiffeq.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.01931/status.svg)](https://doi.org/10.21105/joss.01931)

# Introduction

`neurodiffeq` is a package for solving differential equations with neural networks. Differential equations are equations that relate some function with its derivatives. They emerge in various scientific and engineering domains. Traditionally these problems can be solved by numerical methods (e.g. finite difference, finite element). While these methods are effective and adequate, their expressibility is limited by their function representation. It would be interesting if we can compute solutions for differential equations that are continuous and differentiable.

As universal function approximators, artificial neural networks have been shown to have the potential to solve ordinary differential equations (ODEs) and partial differential equations (PDEs) with certain initial/boundary conditions. The aim of `neurodiffeq` is to implement these existing techniques of using ANN to solve differential equations in a way that allow the software to be flexible enough to work on a wide range of user-defined problems.

# Installation

## Install via pip

Like most standard libraries, `neurodiffeq` in hosted on PyPI. To install the latest stable relesase, simply use the `pip` (or `pip3`) tool.

```sh
pip install neurodiffeq 
# or try: pip3 install neurodiffeq
```

## Manual Install

Alternatively, you can install the library manually to get early access to our new features. This is the recommended way for developers who want to contribute to the library.

1. (optional) Create a new environment. With `conda`: `conda create --name [name of the new environment] python=3.7` and activate the enviroment by `conda activate  [name of the new environment]`; With `venv`: `python3 -m venv temp [path to the new environment]` and activate the environment by `source [path to the new environment]/bin/activate`

2. Clone the repo by `git clone https://github.com/odegym/neurodiffeq.git` and `cd` into the root directory of the repo by `cd neurodiffeq`

3. Install the dependencies by `pip install -r requirements.txt` and install `neurodiffeq` by `pip install .`

4. (optional) Run tests by 
```bash
for file in tests/test*py; do 
    pytest $file;
done
```
Note: running all tests together using `pytest` or `pytest tests/test*py`  will not pass due to [this issue](https://github.com/odegym/neurodiffeq/issues/58).

5. (optional) Deactivate the environment. With `conda`: `conda deactivate`; With `venv`: `deactivate`

# Getting Started

For basic use of `neurodiffeq`, please check the [documentation](https://neurodiffeq.readthedocs.io/en/latest/) hosted on [ReadTheDocs](https://readthedocs.org)

# Contributing

Thanks for your interest to contribute! 

When contributing to this repository, we consider the following process:

1. Open an issue to discuss the change you are planning to make

2. Make the change on a forked repository and update the README.md if changes are made to the interface

3. Open a pull request

