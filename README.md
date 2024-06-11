# LM_Chaboche
Scripts for fitting Chaboche model using the Levenberg-Marquardt method with error analysis

# Overview

This Python script is designed to fit the Chaboche material model to experimental stress-strain data using the Levenberg-Marquardt optimization algorithm. The Chaboche model, a robust constitutive model, is widely used to describe the cyclic behavior and hardening characteristics of materials under plastic deformation.

# Features included
* **Data Loading**: Reads experimental strain and stress data from a CSV file.
* **Chaboche Model implementation**: Implements the Chaboche material model to predict stress based on given strain data and material parameters.
* **Optimization**: Uses the Levenberg-Marquardt algorithm to iteratively adjust model parameters to minimize the difference between predicted and observed stresses.
* **Visualization**: Plots the fitted model against the experimental data for visual validation.
* **Output**: Saves the fitted parameters and modeled stress-strain curve to a CSV file.

# Dependencies
* 'numpy'
* 'matplotlib'
* 'pandas'

# File structure
LM_Model.py: Contains the main script to load data, fitting process, and save results.
Error_cal.py: Contains the script to calculate the SSE and R square.
data.csv: Example CSV file containing 'Strain (%)' and 'Stress' columns.
