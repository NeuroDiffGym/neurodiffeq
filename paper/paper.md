---
title: 'NeuroDiffEq: A Python package for solving differential equations with neural networks'
tags:
  - Python
  - differential equations
  - neural networks
authors:
  - name: Feiyu Chen
    orcid: 0000-0002-0476-9998
    affiliation: 1
  - name: David Sondak
    affiliation: 1
  - name: Pavlos Protopapas
    affiliation: 1
affiliations:
 - name: Institute for Applied Computational Science, Harvard University, Cambridge, MA, United States
   index: 1
date: 29 October 2019
bibliography: paper.bib
---

# Summary

Differential equations emerge in various scientific and engineering domains for modeling physical phenomena.  Most
differential equations of practical interest are analytically intractable.  Traditionally, differential equations are solved
by numerical methods.  Sophistical algorithms exist to integrate differential equations in time and space.  Time integration
techniques continue to be an active area of research and include backward difference formulas and Runge-Kutta methods.
Common spatial discretization approaches include the finite difference method (FDM), finite volume method (FVM), and finite
element method (FEM) as well as spectral methods such as the Fourier-spectral method.  These classical methods have been
studied in detail and much is known about their convergence properties.  Moreover, highly optimized codes exist for solving
differential equations of practical interest with these techniques.  While these methods are efficient and well-studied,
their expressibility is limited by their function representation.  For example, piecewise linear finite element methods
represent complex dynamics as piecewise linear functions.  Mesh adaptivity can provide more fidelity for complicated physics
but terms involving higher-order derivatives are still neglected.  This difficulty can be offset to some degree by increasing
the order of the basis, but the piecewise nature of the expansion does ultimately lead to non-differentiablity at element
boundaries.  Fourier-spectral methods are very high-order methods with excellent convergence properties, but they
suffer from limited expressibility near boundaries.

Artificial neural networks (ANN) are a framework of machine learning algorithms that use a collection of connected units to
learn function mappings. The most basic form of ANNs, multilayer perceptrons, have been proven to be universal function 
approximators[@hornik1989multilayer]. This suggests the possibility of using ANN to solve differential equations. Previous 
studies have demonstrated that ANNs have the potential to solve ordinary differential equations (ODEs) and partial
differential equations (PDEs) with certain initial/boundary conditions[@lagaris1998artificial]. These methods show nice
properties including: (1) continuous and differentiable solutions (2) good interpolation properties (3) [DLS:  I don't
understand this point] smaller number of parameters thus less memory intensive.  Given the interest in developing neural
networks for solving differential equations, it would be extremely beneficial to have an easy-to-use software project that
allows researchers to quickly set up and solve problems.

``NeuroDiffEq`` is a Python package built with ``PyTorch`` that uses ANNs to solve ordinary and partial differential
equations (ODEs and PDEs).  During the release of ``NeuroDiffEq`` we discovered that two other groups had simultaneously
released their own software packages for solving differential equations with neural networks:  ``DeepXDE``[@lu2019deepxde]
and ``PyDEns``[@koryagin2019pydens]. [DLS:  Say something about each of these projects.  How are they different / similar?]
``NeuroDiffEq`` is designed to encourage the user to focus more on the problem domain (What is the differential equation we
need to solve? What are the initial/boundary conditions?) and at the same time allow them to dig into solution domain (What
ANN architecture and loss function should be used? What are the training hyperparameters?) when they want to.  ``NeuroDiffEq`` 
is already currently being  used to study the convergence properties of ANNs for solving differential equations as well as
solving the equations in the field of general relativity (Schwarzchild and Kerr black holes). 

# Methods

The key idea of solving differential equations with ANN is to reformulate the problem as an optimization problem in which we minimize the difference between two sides of the equation. For example, if we are solving 
$$\frac{dx}{dt} - x = 0$$
we can choose to use L2-loss and reformulate the differential equation as the following optimization problem: 
$$
\min_{\vec{p}}(\frac{dNN(\vec{p}, t)}{dt} - NN(\vec{p}, t))^2
$$
where $\vec{p}$ are the weights of the ANN and $NN(\vec{p}, t)$ is the output of the ANN given input $t$. We can see that when this objective function is driven to 0, the original equation is satisfied. 

One additional problem we need to take care of is that a differential equation typically have inifite number of solutions. We reach a particular solution only when some initial/boundary conditions are imposed. Since $NN(\vec{p}, t)$ will not automatically satisfy the initial/boundary conditions, we need to 'constrain' the solution. This constrain can be done in 2 ways: (1) We can add the initial/boundary conditions to the objective function. If we have a initial condition $x(t)\bigg|_{t = t_0} = x_0$, we can change our objective function into
$$
\min_{\vec{p}}\left[(\frac{dNN(\vec{p}, t)}{dt} - NN(\vec{p}, t))^2 + \lambda(NN(\vec{p}, t_0) - x_0)^2\right]
$$
where the second term penalize solutions that don't satisfy the initial condition. We can see that the larger the $\lambda$, the stricter we satisfy the initial condition; (2) We can transform the $NN(\vec{p}, t)$ in a way such that the initial/boundary conditions are bound to be satisfied. If we have a initial condition $x(t)\bigg|_{t = t_0} = x_0$, we can let
$$
\widetilde{NN}(\vec{p}, t) = (1-e^{t_0-t})NN(\vec{p}, t) + x_0
$$
so that when $t = t_0$, $\widetilde{NN}(\vec{p}, t)$ will always be $x_0$. Accordingly, we change our objective function into 
$$
\min_{\vec{p}}(\frac{d\widetilde{NN}(\vec{p}, t)}{dt} - \widetilde{NN}(\vec{p}, t))^2
$$
Both these two methods have their advantages. The first way is simpler and more elegant to implement, and can be more easily extended to be used on high-dimensional PDEs. The second way assures that the initial/boundary conditions are met exactly, considering that differential equations can be sensitive to initial/boundary conditions, this would be desirable. Another advantage of the second method is that fixing these conditions can reduce the effort required during training of the ANN[@mcfall2009artificial]. ``DeepXDE`` uses the first way to impose initial/boundary conditions. ``PyDEns`` uses the second way to impose initial/boundary conditions. In ``NeuroDiffEq``, we choose the second way.  

# References
