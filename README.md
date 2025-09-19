# Code Repository for Information Propagation in Elastic Solids

|Directory|Description|
|---------|-----------|
|fenicsx_scripts|folder for all the FEniCSx code|
|utils|scripts for all functions used in this work|
|validation|code validation for analytical solution and information estimates|

# Pytests

We have provided pytests for our entropy estimator based the KSG estimator implement with NPEET, the relative entropy based on `` ``. 

More details on the tests implement and expected test failures can be found here.

to run pytest with complete information on skips and xfails (expected failures) run:
```bash
pytest -rxs tests/
```
