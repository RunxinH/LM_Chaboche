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
v1.0.2 features:
* Added **bounded LM optimization** with customizable parameter constraints.
* Improved numerical stability with safeguard checks.
* Enhanced plotting with error metrics overlay.
* New GUI interface file added (basic launch, no detailed description included here).

# Requirements
Python ≥ 3.8 is recommended. Required packages:
* numpy
* os
* sys
* PyQt6
* qtrangeslider
* matplotlib
* scipy

# Installation
Clone the repository and install dependencies in requirements
```

Run an example script-based fit:
```bash
python LM_Chaboche.py
```

Run the GUI
```bash
python GUI.py
```


# Modules
* `LM_Chaboche.py` - Core script containing the Chaboche model, LM optimizer, visualization, and error metrics.

# Function Descriptions

Function | Description
---  |---
`load_real_data(filepath)` | Loads experimental CSV data in the same format as in `sample_data`. 
`apply_bounds(params, bounds)` | Applies lower/upper bounds to parameters based on user-defined dictionary of ranges. Returns bounded parameter array.
`chaboche_model(strain, params, E, sigmay)` | Computes stress predictions using the Chaboche constitutive model with three backstresses and isotropic hardening. Returns predicted stress array.
`cal_Jacobian(params, input_data, E, sigmay, h=1e-6)` | Calculates the numerical Jacobian matrix using finite differences. Returns Jacobian (array shape: [num_data, num_params]).
`cal_residual(params, input_data, output_data, E, sigmay)` | Computes residuals between experimental data (`output_data`) and model-predicted stress. Returns residual array.
`LM_bounded(num_iter, params, input_data, output_data, E, sigmay, bounds=None, h=1e-6)` | Performs **bounded Levenberg–Marquardt optimization**. Iteratively updates parameters until convergence. Returns optimized parameter set.
`main()` | Example workflow: loads sample data, runs optimization, prints parameters, calculates error metrics (SSE, R², RMSE), and plots fitted vs. experimental stress–strain curves.


# Contributions and extensions
This project is released under the MIT License and welcomes contributions. Fork, file issues, or open PRs. For major changes, please open a discussion beforehand.

# Acknowledgements
This work builds upon foundational studies in cyclic plasticity and numerical optimization. We acknowledge the following key references:

- Chaboche JL. *A review of some plasticity and viscoplasticity constitutive theories*. International Journal of Plasticity, 2008.
- Levenberg K. *A method for the solution of certain non-linear problems in least squares*, Quarterly of Applied Mathematics, 1944.
- Marquardt DW. *An algorithm for least-squares estimation of nonlinear parameters*, SIAM Journal on Applied Mathematics, 1963.

We also thank the open-source scientific Python ecosystem for enabling this work.
