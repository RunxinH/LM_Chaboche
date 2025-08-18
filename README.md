- [Introduction](#introduction)
  * [Chaboche Model with Error Analysis](#chaboche-model-with-error-analysis)
- [Latest version](#latest-version)
- [Requirements](#requirements)
- [Installation](#installation)
- [Modules](#modules)
- [Function Descriptions](#function-descriptions)
- [Contributions and extensions](#contributions-and-extensions)
- [Acknowledgements](#acknowledgements)

# Introduction
This repository provides an open-source implementation of the **Chaboche cyclic plasticity model** with integrated **error analysis**, using the **Levenberg-Marquardt (LM) optimization algorithm** to fit experimental stress-strain data. 

The Chaboche model captures the nonlinear kinematic and isotropic hardening behavior under cyclic loading, which is essential for modeling fatigue and plasticity in metals. This combined workflow allows users to fit experimental data, visualize results, and quantitatively assess the fitting performance using error metrics.

## Chaboche Model with Error Analysis
The unified script `LM_Chaboche.py` implements:
* Multi-backstress Chaboche constitutive modeling.
* LM optimization to calibrate material parameters.
* SSE, R² and RMSE error evaluation.
* Visualization of experimental vs predicted curves.
* Export of fitted results and performance metrics.

# Latest version
v1.0.1 features:
* Merged Chaboche model and error analysis into one script.
* Improved modularity and code readability.
* Single-step execution for both fitting and evaluation.

# Requirements
Python ≥ 3.8 is recommended. Required packages:
* numpy
* pandas
* matplotlib
* scipy

# Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/chaboche-lm.git
cd chaboche-lm
pip install -r requirements.txt
```

Run the model fitting and evaluation:
```bash
python LM_Chaboche.py
```

# Modules
* `LM_Chaboche.py` - Core script containing the Chaboche model, LM optimizer, visualization, and error metrics.

# Function Descriptions

Function | Description
---  |---
`load_real_data(filepath)` | Loads experimental CSV data with columns: `Strain 1 (%)`, `Stress`. Returns strain and stress arrays.
`chaboche_model(strain, params)` | Computes stress predictions based on Chaboche model and parameter set.
`cal_Jacobian(params, input_data)` | Returns the Jacobian matrix with partial derivatives of stress w.r.t. parameters.
`cal_residual(params, input_data, output_data)` | Returns residuals between experimental and model-predicted stress.
`cal_Hessian_LM(Jacobian, u, num_params)` | Constructs the damped Hessian matrix used in LM updates.
`cal_g(Jacobian, residual)` | Calculates the gradient vector.
`cal_step(Hessian_LM, g)` | Solves for parameter update step using Hessian and gradient.
`LM(num_iter, params, input_data, output_data)` | Main optimization loop for parameter fitting.
`calculate_sse(actual, predicted)` | Computes sum of squared errors.
`calculate_r_squared(actual, predicted)` | Computes coefficient of determination (R²).

# Contributions and extensions
This project is released under the MIT License and welcomes contributions. Fork, file issues, or open PRs. For major changes, please open a discussion beforehand.

# Acknowledgements
This work builds upon foundational studies in cyclic plasticity and numerical optimization. We acknowledge the following key references:

- Chaboche JL. *A review of some plasticity and viscoplasticity constitutive theories*. International Journal of Plasticity, 2008.
- Levenberg K. *A method for the solution of certain non-linear problems in least squares*, Quarterly of Applied Mathematics, 1944.
- Marquardt DW. *An algorithm for least-squares estimation of nonlinear parameters*, SIAM Journal on Applied Mathematics, 1963.

We also thank the open-source scientific Python ecosystem for enabling this work.
