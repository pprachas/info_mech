# Entropy & Volume Estimator Tests

This test suite validates the behavior of the custom entropy estimator `entropy_r` implemented in `utils/custom_ee.py`.  
It uses [pytest](https://docs.pytest.org/) for automated testing.

---

## Test Functions

1. **Entropy identity test (`test_entropy`)**  
   - Verifies  
     $$
     I(X;Y) \approx H(X) + H(Y) - H(X,Y)
     $$  
     within a small tolerance.

2. **Volume test (`test_volume`)**  
   - Compares estimated volumes against ground truth.  
   - Skips when the volume is not mathematically well-defined.  
   - Marks as expected failure (xfail) when known estimator limitations apply (e.g., `uniform_0_2`).

3. **Convergence test (`test_volume_convergence_highdim`)**  
   - Runs only for `uniform_0_1_3D`.  
   - Checks whether estimated volumes approach the true value (1) as sample size increases.  

---

## Test Cases

We test `entropy_r` on a collection of synthetic datasets that cover simple, structured, and degenerate scenarios:

### 1. `uniform_0_1_1D`
- **Setup:**  
  $x \sim \text{Uniform} (0, 1)$ in 1D, $y \sim \text{Uniform}(0, 1)$.  
- **Ground truth:**  
  - Support volume of marginal space x: 1  
  - Support volume of marginal space y: 1  
  - Support volume of joint space: 1.  

  **Expectation of test results:**
  - All tests should pass



---

### 2. `uniform_0_2`
- **Setup:**  
  $x \sim \text{Uniform}(0, 2)$ in 1D, $y \sim \text{Uniform}(0, 1)$.  
- **Ground truth:**  
  - Support volume of marginal space x: 2  
  - Support volume of marginal space y: 1
  - Support volume of joint space: 2.

- **Expectation of test results:**
  - Entropy and mutual information relationship should hold
  - Volume estimation in the marginal spaces should fail but volume estimation in the joint space should pass

 **Notes:** For results obtained in the paper, we circumvent this issue by scaling both x and y with sklearn's ``StandardScaler`` which will make support sizes for x and y relatively similar. A ``MinMaxScaler`` can also work since it will enforce support size [0,1] for x and y, but in our experience is less numerically stable for knn-based estimation for both entropy and mutual information.

---

### 3. `uniform_0_1_3D`
- **Setup:**  
  $x \sim \text{Uniform} (0, 1)^3$ in 3D, $y = x$.  
- **Ground truth:**  
  - Support volume of marginal space x: 2  
  - Support volume of marginal space y: 1
  - Support volume of joint space: 1.
- **Expectation of test results:**  
  - Entropy and mutual information relationship should hold
  - Volume estimation is bad for both the marginal spaces due to curse of dimensionality.
  - **Note:** This is the dataset used in the **convergence test** to check if estimated volumes approach 1 as the sample size increases.

---

### 4. `bijective_1D`
- **Setup:**  
  $x \sim \text{Uniform}(0, 1)$, $y = x$ 
- **Ground truth:**  
  - Support volumes are well-defined individually, but the joint support is measure-zero.  
- **Expectation of test results:**  
  - Entropy identity should hold.  
  - Volume test is skipped.  

---

### 5. `bijective_highdim`
- **Setup:**  
  $x ~ \text{Uniform}(0, 1)^d$, $y = x$.  
- **Ground truth:**  
  - Same as case 4, but in higher dimension.  
  - Joint support is a diagonal set of measure-zero in \([0,1]^{2d}\).  
- **Expectation of test result:**  
  - Entropy identity should hold.  
  - Volume test is skipped.  

---

##  Running the Tests

To run all tests:

```bash
pytest
```
To also see the xfail and skip messgaes run 

```bash
pytest -rxs
```
