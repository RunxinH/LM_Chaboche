import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Calculate SSE
def calculate_sse(actual, predicted):
    squared_errors = (actual - predicted) ** 2
    return np.sum(squared_errors)

# Calculate R-squared (R²)
def calculate_r_squared(actual, predicted):
    sse = calculate_sse(actual, predicted)
    sst = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (sse / sst)

experiment_path = '' # Experiment data
chaboche_path = '' # Data after LM implementation


experiment_data = pd.read_csv(experiment_path)
experimental_stress_values = experiment_data['Stress'].values # Make sure your experiment data in csv file has a colume called 'Stress'

chaboche_data = pd.read_csv(chaboche_path)
chaboche_stress_values = chaboche_data['Stress'].values


# Perform the calculations
sse = calculate_sse(experimental_stress_values, chaboche_stress_values)
r_squared = calculate_r_squared(experimental_stress_values, chaboche_stress_values)


# Print the results
print(f"Sum of Squared Errors (SSE): {sse}")
print(f"R-squared (R^2): {r_squared}")
