- [Introduction](#introduction)
  * [Chaboche Model](#chaboche-model)
  * [Error Analysis](#error-analysis)
  * [Moving forward](#moving-forward)
- [Latest version](#latest-version)
- [Requirements](#requirements)
- [Installation](#installation)
- [Modules](#modules)
- [Function Descriptions](#function-descriptions)
- [Contributions and extensions](#contributions-and-extensions)
- [Acknowledgements](#acknowledgements)

# Introduction
This repository provides an open-source implementation of the **Chaboche cyclic plasticity model** using the **Levenberg-Marquardt (LM) optimization algorithm** to fit experimental stress-strain data. 
It also includes a simple **error analysis script** to evaluate the fitting performance.

The Chaboche model describes the nonlinear kinematic and isotropic hardening behavior under cyclic loading, widely applied in fatigue and plasticity research.

## Chaboche Model
The `LM_Model.py` script implements:
* The multi-backstress Chaboche constitutive model.
* Levenberg-Marquardt optimization to fit material parameters.
* Visualization of experimental vs fitted data.
* Saving fitted results as CSV.

## Error Analysis
The `Error_cal.py` script calculates:
* Sum of Squared Errors (SSE)
* R-squared (R²) values

These metrics provide quantitative insight into the quality of the model fitting.

## Moving forward
Future improvements may include:
* Adding support for viscoplasticity.
* Automating parameter sensitivity analysis.
* Expanding to temperature-dependent behavior.

# Latest version
v1.0.0 is the first release of this repository, featuring:
* Fully functional Chaboche model with LM fitting.
* Basic error analysis scripts.
* Example workflow for fitting experimental CSV data.

# Requirements
Python ≥ 3.8 is recommended. The following Python libraries are required:
* numpy
* pandas
* matplotlib
* scipy

# Installation

Clone this repository and install dependencies:
```
git clone https://github.com/yourusername/chaboche-lm.git
cd chaboche-lm
pip install -r requirements.txt
```

Run the model fitting and error analysis:
```
python LM_Model.py
python Error_cal.py
```

# Modules

* `LM_Model.py` - Chaboche model, LM optimization, and data fitting.
* `Error_cal.py` - Calculate SSE and R² for fitted vs experimental data.

# Function Descriptions

Function | Description
---  |---
`load_real_data(filepath)` | *Parameters*: `filepath`: path to the experimental data CSV file. Expected columns: `Strain 1 (%)`, `Stress`. <br>*Returns*: `strain_data`: numpy array of strain values (converted from percent to decimal), `stress_data`: numpy array of stress values.
`chaboche_model(strain, params)` | *Parameters*: `strain`: numpy array of input strain data; `params`: list of model parameters `[C1, r1, C2, r2, C3, r3, Q, b]`. <br>*Returns*: `sigma`: numpy array of predicted stress values using the Chaboche model.
`cal_deriv(params, strain, param_index)` | *Parameters*: `params`: list of model parameters; `strain`: numpy array of strain values; `param_index`: index of the parameter for derivative calculation. <br>*Returns*: Numerical derivative (numpy array) of stress with respect to the selected parameter.
`cal_Hessian_LM(Jacobian, u, num_params)` | *Parameters*: `Jacobian`: Jacobian matrix of residuals; `u`: damping factor for LM algorithm; `num_params`: total number of parameters. <br>*Returns*: `H`: Hessian matrix with damping applied.
`cal_g(Jacobian, residual)` | *Parameters*: `Jacobian`: Jacobian matrix of residuals; `residual`: difference between predicted and experimental stress values. <br>*Returns*: `g`: Gradient vector.
`cal_step(Hessian_LM, g)` | *Parameters*: `Hessian_LM`: Hessian matrix; `g`: Gradient vector. <br>*Returns*: `step`: parameter update step (numpy array). Uses pseudo-inverse if the Hessian is singular.
`cal_Jacobian(params, input_data)` | *Parameters*: `params`: list of model parameters; `input_data`: strain data. <br>*Returns*: `J`: Jacobian matrix (numpy array), containing derivatives of stress with respect to each parameter.
`cal_residual(params, input_data, output_data)` | *Parameters*: `params`: list of model parameters; `input_data`: strain data; `output_data`: experimental stress data. <br>*Returns*: `residual`: difference between experimental and predicted stress (numpy array).
`LM(num_iter, params, input_data, output_data)` | *Parameters*: `num_iter`: maximum number of LM iterations; `params`: initial parameter guess (numpy array); `input_data`: strain data; `output_data`: stress data. <br>*Returns*: `fitted_params`: optimized model parameters (numpy array).
`calculate_sse(actual, predicted)` | *Parameters*: `actual`: numpy array of experimental stress; `predicted`: numpy array of model-predicted stress. <br>*Returns*: `sse`: sum of squared errors as a float.
`calculate_r_squared(actual, predicted)` | *Parameters*: `actual`: numpy array of experimental stress; `predicted`: numpy array of model-predicted stress. <br>*Returns*: `r_squared`: coefficient of determination (R²) as a float.

# Contributions and extensions
This project is open for improvements and collaborative development. Feel free to fork, extend, and submit pull requests. Please contact me first if you plan substantial changes to the model core.

# Acknowledgements
This work draws on foundational theories in plasticity and optimization algorithms. Thanks to the broader computational mechanics community for advancing these fields.
