- [Introduction](#introduction)
  * [Chaboche Model](#chaboche-model)
  * [Error Analysis](#error-analysis)
  * [Moving forward](#moving-forward)
- [Latest version](#latest-version)
- [Requirements](#requirements)
- [Installation](#installation)
- [Modules](#modules)
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

# Usage example
* Place your experimental data CSV in the working directory. Required columns: `Strain 1 (%)`, `Stress`.
* Update the file paths in the scripts (`filepath`, `save_file_path`, etc.).
* Run the fitting and error calculation scripts.

# Contributions and extensions
This project is open for improvements and collaborative development. Feel free to fork, extend, and submit pull requests. Please contact me first if you plan substantial changes to the model core.

# Acknowledgements
This work draws on foundational theories in plasticity and optimization algorithms. Thanks to the broader computational mechanics community for advancing these fields.
