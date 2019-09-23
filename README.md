# neurodiffeq

[![Build Status](https://travis-ci.org/odegym/neurodiffeq.svg?branch=master)](https://travis-ci.org/odegym/neurodiffeq)

[![codecov](https://codecov.io/gh/feiyu-chen96/neurodiffeq/branch/master/graph/badge.svg)](https://codecov.io/gh/feiyu-chen96/neurodiffeq)

# Introduction

`neurodiffeq` is a package for solving differential equations with neural networks. Differential equations are equations that relate some function with its derivatives. They emerge in various scientific and engineering domains. Traditionally these problems can be solved by numerical methods (e.g. finite difference, finite element). While these methods are effective and adequate, their solutions are discrete. It would be interesting if we can compute solutions for differential equations that are continuous and differentiable.

As universal function approximators, artificial neural networks have been shown to have the potential to solve ordinary differential equations (ODEs) and partial differential equations (PDEs) with certain initial/boundary conditions. The aim of `neurodiffeq` is to implement these existing techniques of using ANN to solve differential equations in a way that allow the software to be flexible enough to work on a wide range of user-defined problems.

# Installation

Currently `neurodiffeq` is not in PyPI, so it needs to be installed from this repo. To install `neurodiffeq`, use `pip install git+https://github.com/odegym/neurodiffeq.git`

# Getting Started

For basic use of `neurodiffeq`, please check the [User Guide](https://feiyu-chen96.github.io/neurodiffeq_User_Guide.html)

# Contributing

Thanks for your interest to contribute! 

When contributing to this repository, we consider the following process:

1. Open an issue to discuss the change you are planning to make

2. Make the change on a forked repository and update the README.md if changes are made to the interface

3. Open a pull request

