import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def apply_bounds(params, bounds):
    """Apply parameter bounds by clipping values."""
    param_names = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
    bounded_params = params.copy()
    
    for i, name in enumerate(param_names):
        if name in bounds:
            lower, upper = bounds[name]
            bounded_params[i] = np.clip(bounded_params[i], lower, upper)
    
    return bounded_params


def chaboche_model(strain, params, E, sigmay):
    """Chaboche model implementation."""
    C1, r1, C2, r2, C3, r3, Q, b = params
    alphas = np.zeros((len(strain), 3))  
    sigma = np.zeros(len(strain))
    ep = np.zeros(len(strain))
    R = np.zeros(len(strain))

    for i in range(len(strain)-1):
        et_dot = strain[i+1] - strain[i]
        lamda_dot = 0
        T_sdot = E * et_dot  
        T_stress = sigma[i] + T_sdot  
        T_alpha = alphas[i]  
        f = abs(T_stress - sum(T_alpha)) - (sigmay + R[i])

        sign_sig = -1 if T_stress - sum(T_alpha) < 0 else 1

        if f <= 0:
            sigma[i+1] = sigma[i] + T_sdot
            alphas[i+1] = T_alpha
            ep[i+1] = ep[i]
            R[i+1] = R[i] + b * (Q - R[i]) * lamda_dot
        else:
            C_alpha = (C1 - sign_sig * r1 * T_alpha[0],
                       C2 - sign_sig * r2 * T_alpha[1],
                       C3 - sign_sig * r3 * T_alpha[2])
            denominator = E + sum(C_alpha) + b * (Q - R[i])
            lamda_dot = f / denominator
            ep_dot = sign_sig * lamda_dot

            ep[i+1] = ep[i] + ep_dot
            alphas[i+1] = T_alpha + np.multiply(C_alpha, ep_dot)
            sigma[i+1] = T_stress - E * lamda_dot * sign_sig
            R[i+1] = R[i] + b * (Q - R[i]) * lamda_dot

    return sigma



def cal_deriv(params, strain, param_index, E, sigmay, _lambda):
    """Calculate numerical derivative."""
    params1 = params.copy()
    params2 = params.copy()
    params1[param_index] += _lambda
    params2[param_index] -= _lambda
    sigma1 = chaboche_model(strain, params1, E, sigmay)
    sigma2 = chaboche_model(strain, params2, E, sigmay)
    return (sigma1 - sigma2) / (2 * _lambda)

def cal_Jacobian(params, input_data, E, sigmay, _lambda):
    """Calculate Jacobian matrix."""
    num_params = len(params)
    num_data = len(input_data)
    J = np.zeros((num_data, num_params))
    for i in range(num_params):
        J[:, i] = cal_deriv(params, input_data, i, E, sigmay, _lambda)
    return J

def cal_residual(params, input_data, output_data, E, sigmay):
    """Calculate residual."""
    data_est_output = chaboche_model(input_data, params, E, sigmay)
    residual = output_data - data_est_output
    return residual

def LM_bounded(num_iter, params, input_data, output_data, E, sigmay, _lambda = 1e-8, bounds=None):
    """Levenberg-Marquardt with bounds."""
    num_params = len(params)
    k = 0
    
    # Apply initial bounds
    if bounds is not None:
        params = apply_bounds(params, bounds)
    
    residual = cal_residual(params, input_data, output_data, E, sigmay)
    prev_error = np.linalg.norm(residual)

    while k < num_iter:
        Jacobian = cal_Jacobian(params, input_data, E, sigmay, _lambda)
        A = Jacobian.T.dot(Jacobian)
        g = Jacobian.T.dot(residual)
        u = 1e-3 * np.max(np.diag(A))
        
        H = A + u * np.eye(num_params)
        try:
            step = np.linalg.solve(H, g)
        except:
            step = np.linalg.pinv(H).dot(g)

        if np.linalg.norm(step) <= 1e-15:
            break

        new_params = params + step
        
        # Apply bounds
        if bounds is not None:
            new_params = apply_bounds(new_params, bounds)
        
        new_residual = cal_residual(new_params, input_data, output_data, E, sigmay)
        new_error = np.linalg.norm(new_residual)
        
        if new_error < prev_error:
            params = new_params
            residual = new_residual
            prev_error = new_error
            
        if k % 10 == 0:
            print(f"Iteration {k}: Error = {new_error:.6f}")

        k += 1
    
    print(f"Optimization completed after {k} iterations")
    return params



def load_data(filepath):
    """Load experimental data."""
    data = pd.read_csv(filepath)
    strain_data = data['Strain 1 (%)'].values / 100
    stress_data = data['Stress'].values
    return strain_data, stress_data

def main():
    # Configuration
    experiment_path = 'sample_data/sample_data_1-10c.csv' # Data file here
    
    # Material properties
    E = 176980 # Young's Modulus
    sigmay = 246.53 # Yield point
    
    # Parameter bounds
    bounds = {
        'C1': (10000, 100000),
        'r1': (100, 2000),
        'C2': (1000, 50000),
        'r2': (10, 500),
        'C3': (100, 10000),
        'r3': (1, 100),
        'Q': (10, 150),
        'b': (1, 10)
    }
    
    # Initial guess
    initial_guess = np.array([75014.9416, 1526.7128, 25034.4820, 95.7779, 1013.8127, 100.0000, 70.9643, 10])# Initial guess
    
    try:
        print("Loading data...")
        strain, stress = load_data(experiment_path)
        print(f"Loaded {len(strain)} data points")
        
        print("\nStarting optimization with bounds...")
        fitted_params = LM_bounded(20, initial_guess, strain, stress, E, sigmay, bounds)
        
        # Results
        predicted_stress = chaboche_model(strain, fitted_params, E, sigmay)
        
        # Error metrics
        sse = np.sum((stress - predicted_stress) ** 2)
        sst = np.sum((stress - np.mean(stress)) ** 2)
        r_squared = 1 - (sse / sst)
        rmse = np.sqrt(np.mean((stress - predicted_stress) ** 2))
        
        print(f"\nFitted Parameters:")
        param_names = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
        for name, value in zip(param_names, fitted_params):
            print(f"{name}: {value:.4f}")
        
        print(f"\nError Metrics:")
        print(f"SSE: {sse:.6f}")
        print(f"R^2: {r_squared:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        # Plot
        plt.figure(figsize=(10, 7))
        plt.plot(strain, stress, 'o', label='Experimental Data', alpha=0.6, markersize=3)
        plt.plot(strain, predicted_stress, '-', label='Fitted Model', linewidth=2, color='orange')
        plt.title("Chaboche Model Fit with Bounds")
        plt.xlabel("Strain")
        plt.ylabel("Stress (MPa)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Save results
        results_df = pd.DataFrame({
            'Strain': strain,
            'Experimental_Stress': stress,
            'Fitted_Stress': predicted_stress
        })
        results_df.to_csv('bounded_results.csv', index=False)
        print("\nResults saved to 'bounded_results.csv'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()