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

Differential equations emerge in various scientific and engineering domains. Traditionally these problems can be solved by numerical methods like finite difference method (FDM), finite volume method (FVM), and finite element method (FEM). While these methods are effective and adequate, the numerical solution are usually discrete and not differentiable. Artificial neural networks (ANN) are a framework of machine learning algorithms that use a collection of connected units to learn function mappings. As the most basic form of ANN, multilayer perceptrons are proven to be universal function approximators. This suggests the possibility of using ANN to solve differential equations. Previous studies have demonstrated that ANN has the potential to solve ordinary differential equations (ODEs) and partial differential equations (PDEs) with certain initial/ boundary conditions[`@lagaris1998artificial`]. These method demonstrate nice properties include: 1) continuous and differentiable solution 2) better interpolation properties 3) smaller number of parameters.

``NeuroDiffEq`` is a Python package built with ``PyTorch`` that uses ANNs to solve ODEs and PDEs. It is designed to encourage the user focus more on the problem domain (What is the differential equation we need to solve? What are the initial/boundary value conditions?) and at the same time allow them to dig into solution domain (What neural network architecture and loss function should be used? What are the training hyperparameters?) when they want to. 

# Statement of Need

# References