# Mutual Information Estimators

* Used KSG estimator -- good balance between accuracy, complexity and number of data needed
  * Things about ksg to keep in mind:
  * uses knn to estimate distributions
    * number of neighbhors will affect the MI value but the general trend os the same
    * Need normalization due to knn --
      * Shouldn't affect MI due to diffeomorphic (smooth invertible) transformation
      * In practivee standard scaling works better than min-max scaling (deals with truncation error better)  
