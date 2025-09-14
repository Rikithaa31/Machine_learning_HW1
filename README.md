# Machine_learning_HW1

## Student Info

* **Name:** Rikithaa Sai D
* **Id**: 700
* **Course:** CS5710 Machine Learning (Fall 2025)

## How to run

1. Create/activate a Python 3.9+ environment.
2. Install: `pip install numpy matplotlib jupyter`
3. Launch: `jupyter notebook` and open any `.ipynb` under `/notebooks`.
4. Run cells top-to-bottom.

## Results summary

### 1 — Function Approximation by Hand

Dataset: (1,1), (2,2), (3,2), (4,5)

* Model θ=(1,0): **MSE = 0.5000**
* Model θ=(0.5,1): **MSE = 1.1250**
  **Better fit:** θ=(1,0)

### 2 — Random Guessing Practice

Cost: $J(\theta_1,\theta_2)=8(\theta_1-0.3)^2+4(\theta_2-0.7)^2$

* $J(0.1,0.2)=1.3200$
* $J(0.5,0.9)=0.4800$
  **Closer to optimum (0.3,0.7):** (0.5, 0.9)
  **Why random guessing is inefficient (2–3 lines):** It explores the space blindly, so the chance of landing near the optimum in a continuous domain is very small. Without gradient/curvature information, guesses don’t move systematically toward lower cost; many evaluations are wasted. Gradient-based methods use the slope to reliably reduce $J$.

### 3 — First Gradient Descent Iteration

Dataset: (1,3), (2,4), (3,6), (4,5), start θ⁽⁰⁾=(0,0), α=0.01

* θ⁽¹⁾ = (0.24500, 0.09000), **J(θ⁽⁰⁾)=21.500000 > J(θ⁽¹⁾)=15.256037**
* θ⁽²⁾ = (0.44875, 0.16595), **J(θ⁽¹⁾)=15.256037 > J(θ⁽²⁾)=10.922289**
  **Conclusion:** Error strictly decreases across the first two updates.

### 4 — Compare Random Guessing vs Gradient Descent

Dataset: (1,2), (2,2), (3,4), (4,6)

* Random guesses:

  * $J(0.2,0.5)=8.3500$
  * $J(0.9,0.1)=1.9350$
* After one GD step from θ=(0,0) with α=0.01 → θ=(0.21, 0.07): **J = 10.50915**
  **Lower error:** the random guess (0.9, 0.1). It happens to be closer to the least-squares line than a single small GD step from the origin.

### 5 — Recognizing Underfitting and Overfitting (short answers)

* **Scenario:** Training error is high, test error is also high → **Underfitting**.
* **Why:** The model is too simple or constrained to capture the signal in the data.
* **Fixes (two+):** Increase model capacity/features; reduce regularization; train longer/tune hyperparameters; feature engineering.

### 6 — Comparing Models (short answers)

* **Overfitting vs underfitting:** Model A (perfect train, poor test) → **Overfitting**; Model B (poor train & test) → **Underfitting**.
* **Bias–variance tradeoff:** Overfitting = low bias, high variance; Underfitting = high bias, low variance.
* **Improvements:** For A, add regularization/early-stopping, simplify model, get more data. For B, increase capacity/features and/or reduce regularization.

### 7 — Programming Problem (Linear Regression: Normal Equation vs Gradient Descent)

* Implemented per instructions in a separate notebook/script (generate data $y=3+4x+\epsilon$, 200 samples, $x\in[0,5]$).
* Reported the closed-form solution, GD trajectory and loss curve, and the two fitted lines on the same plot.
* **Comment:** GD converges to the closed-form solution (within tolerance) with an appropriate learning rate and iterations.




