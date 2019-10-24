---
title: 'NeuroDiffEq: A Python package for solving differential equations with neural networks'
tags:
  - Python
  - differential equations
  - neural networks
authors:
  - name: Feiyu Chen
    affiliation: 1
  - name: David Sondak
    affiliation: 1
  - name: Pavlos Protopapas
    affiliation: 1
affiliations:
 - name: SEAS, Harvard University, Cambridge, MA, United States
   index: 1
date: 29 October 2019
bibliography: paper.bib
---

# Summary

Differential equations emerge in various scientific and engineering domains. Traditionally these problems can be solved by numerical methods like finite difference method (FDM), finite volume method (FVM), and finite element method (FEM). While these methods are effective and adequate, the numerical solutions are usually discrete and not differentiable. Artificial neural networks (ANN) are a framework of machine learning algorithms that use a collection of connected units to learn function mappings. As the most basic form of ANN, multilayer perceptrons are proven to be universal function approximators. This suggests the possibility of using ANN to solve differential equations. Previous studies have demonstrated that ANN has the potential to solve ordinary differential equations (ODEs) and partial differential equations (PDEs) with certain initial/boundary conditions[@lagaris1998artificial]. These methods demonstrate nice properties include: 1) continuous and differentiable solution 2) better interpolation properties 3) smaller number of parameters. ``NeuroDiffEq`` is a Python package built with ``PyTorch`` that uses ANNs to solve ODEs and PDEs. It is not the only software package that aims to encapsulate this technique. Unbeknownst to us, two other similar packages ``DeepXDE``[@lu2019deepxde] and ``PyDEns``[@koryagin2019pydens] are also developed this year.

``NeuroDiffEq`` is designed to encourage the user focus more on the problem domain (What is the differential equation we need to solve? What are the initial/boundary conditions?) and at the same time allow them to dig into solution domain (What ANN architecture and loss function should be used? What are the training hyperparameters?) when they want to.  ``NeuroDiffEq`` is already being used to study the convergence properties of ANN for solving differential equations. It is also used in another ongoing project for solving equations in the field of general relativity. 

# Methods

The key idea of solving differential equations with ANN is to reformulate the problem as an optimization problem, in which we minimize the difference between two sides of the equation. For example, if we are solving 
$$\frac{dx}{dt} - x = 0$$
and we choose L2-loss as our loss function, then we can reformulate the differential equation as the following optimization problem: 
$$
\min_{\vec{p}}(\frac{dNN(\vec{p}, t)}{dt} - NN(\vec{p}, t))^2
$$
where $\vec{p}$ are the weights of the ANN and $NN(\vec{p}, t)$ is the output of the ANN. We can see that when this objective function is driven to 0, the original equation is satisfied. 

One twist is that a differential equation typically have inifite number of solutions. We reach a particular solution only when some initial/boundary conditions are imposed. Since $NN(\vec{p}, t)$ will not automatically satisfy the initial/boundary conditions, we need to 'constrain' the solution. This constrain can be done in 2 ways: (1) We can add the initial/boundary conditions to the objective function. If we have a initial condition $x(t)\bigg|_{t = t_0} = x_0$, we can change our objective function so our problem becomes: 
$$
\min_{\vec{p}}\left[(\frac{dNN(\vec{p}, t)}{dt} - NN(\vec{p}, t))^2 + \lambda(NN(\vec{p}, t_0) - x_0)^2\right]
$$
We can see that the larger the $\lambda$, the stricter we satisfy the initial/boundary conditions;(2) We can transform the $NN(\vec{p}, t)$ in a way such that the initial/boundary conditions are bound to be satisfied. If we have a initial condition $x(t)\bigg|_{t = t_0} = x_0$, we can let
$$
\tilde{NN}(\vec{p}, t) = (1-e^{t_0-t})NN(\vec{p}, t) + x_0
$$
so our problem becomes: 
$$
\min_{\vec{p}}(\frac{d\tilde{NN}(\vec{p}, t)}{dt} - \tilde{NN}(\vec{p}, t))^2
$$
Both these two methods have their advantages. The first way is simpler and more elegant from a software perspective, and can be more easily adapted to be used on high-dimensional PDEs. The second way assures that the initial/boundary conditions are met exactly, considering that differential equations can be sensitive to initial/boundary conditions, this would be desirable. Another advantage of the second method is that fixing these conditions can reduce the effort required during training of the ANN[@mcfall2009artificial]. ``DeepXDE`` uses the first way to impose initial/boundary conditions. ``PyDEns`` uses the second way to impose initial/boundary conditions. In ``NeuroDiffEq``, we choose the second way.  

# References