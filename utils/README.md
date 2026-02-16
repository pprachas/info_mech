# Files

All scripts here contains functions used to reproduce the work found in the paper.

|File|Description|
|----|-----------|
|analytical.py|(legacy) functions to output analytical solutions of Flamant problem for point load and uniform load|
|custom_ee.py|function to compute LDDP differential entropy. More details on the estimation can be found in the ``test`` folder|
|fea.py|collections of functions to run FEniCSx (i.e., import mesh, running fea with same factorization, etc.)|
|grad_check.py|(legacy) compares finite difference of a function to a automatic differentiation|
|jax_mi.py| (legacy) DIfferentiable KSG estimator written in JAX. This estomator is coordinate dependent and need futher validation. |
|meshing.py|functions to generate structured meshing for rectangle and plate-in-hole.|
|postprocess.py|functions for signal pruning before mutual information estimation and greedy sensor selection|
|symbolic.py|functions to aid in solving elastic halfspace problems. Also contains functions to generate Legendre polynomials and efficient usage of mpmath.|
