# Code Repository for: "Information Propagation and Encoding in Solids: A Quantitative Approach Towards Mechanical Intelligence"
This repository accompanies this preprint:
[Information Propagation and Encoding in Solids: A Quantitative Approach Towards Mechanical Intelligence](https://arxiv.org/abs/2602.00140)

Here is quick overview of each directory. Directories marked with legacy code are not used for the paper. 
Further details about each script and siubdirectory can be found in the READMEs located in each directory.

|Directory|Description|
|---------|-----------|
|fenicsx_mi|directory for all the FEniCSx code used in the publication. This directory includes code for parameter space exploration and Bayesian Optimization. Both the FEA components and mutual information computation is found in this directory.|
|fenicsx_scripts|scripts for simple problem in linear elastic FEA in FEniCSx (all legacy code)|
|half_space|directory containing code for elastic halfspace. This includes code for both symbolic integration of Flamant's solution and mutual information computations.|
|utils|scripts for all functions used in this work|
|validation|code validation for analytical solution and information estimates (all legacy code)|


##  Pytests

We have provided pytests for our entropy estimator based the KSG estimator implement with NPEET, the relative entropy based on [Accurate estimation of the normalized mutual information of multidimensional data](https://pubs.aip.org/aip/jcp/article/161/5/054108/3306182/Accurate-estimation-of-the-normalized-mutual). 

More details on the tests implement and expected test failures can be found inside the directory.

to run pytest with complete information on skips and xfails (expected failures) run:
```bash
pytest -rxs tests/
```



