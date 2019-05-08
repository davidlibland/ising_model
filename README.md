# Polynomial interactions
In this project, we consider the an Ising model of 
the form
 
H[x] = \sum_{i,j} A_{i,j} x_i x_j + \sum_{i,j} B_{i,j,k} x_i x_j x_k

where A and B denote a symmetric 2-tensor and a 
symmetric 3-tensor (respectively), and $x\in \{-1,1\}^N$ is a Ising 
vector.
The energy function of this Ising model is 

p(x) = 1/Z exp(-H[x])

where Z is a normalizing constant.

We consider two sampling strategies:
1. Exact sampling
2. Approximate sampling (via MCMC)

A writeup can be found in [this notebook](polynomial_interactions.ipynb).