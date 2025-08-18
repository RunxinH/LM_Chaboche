import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load experimental data."""
    data = pd.read_csv(filepath)
    strain_data = data['Strain 1 (%)'].values / 100 # Notice percentage or non-percentage for strain
    stress_data = data['Stress'].values
    return strain_data, stress_data

def apply_bounds(params, bounds):
    param_names = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
    bounded_params = params.copy()
    
    for i, name in enumerate(param_names):
        if name in bounds:
            lower, upper = bounds[name]
            bounded_params[i] = np.clip(bounded_params[i], lower, upper)
    
    return bounded_params


def chaboche_model(strain, params, E, sigmay):
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
            # Elastic
            sigma[i+1] = sigma[i] + T_sdot
            alphas[i+1] = T_alpha
            ep[i+1] = ep[i]
            R[i+1] = R[i]  
        else:
            # Plastic
            C_alpha = np.array([ 
                C1 - sign_sig * r1 * T_alpha[0],
                C2 - sign_sig * r2 * T_alpha[1],
                C3 - sign_sig * r3 * T_alpha[2]
            ])
            
            denominator = E + sum(C_alpha) + b * (Q - R[i])
            if abs(denominator) < 1e-10:  
                denominator = 1e-10 if denominator >= 0 else -1e-10
                
            lamda_dot = f / denominator
            ep_dot = sign_sig * lamda_dot

            ep[i+1] = ep[i] + ep_dot
            alphas[i+1] = T_alpha + C_alpha * ep_dot  
            sigma[i+1] = T_stress - E * lamda_dot * sign_sig
            R[i+1] = R[i] + b * (Q - R[i]) * lamda_dot

    return sigma


def cal_deriv(params, strain, param_index, E, sigmay):
    """Calculate numerical derivative with improved step size."""
    h = 1e-6
    params1 = params.copy()
    params2 = params.copy()
    step = h * max(abs(params[param_index]), 1.0)
    params1[param_index] += step
    params2[param_index] -= step
    sigma1 = chaboche_model(strain, params1, E, sigmay)
    sigma2 = chaboche_model(strain, params2, E, sigmay)
    return (sigma1 - sigma2) / (2 * step)


def cal_Jacobian(params, input_data, E, sigmay):
    num_params = len(params)
    num_data = len(input_data)
    J = np.zeros((num_data, num_params))
    for i in range(num_params):
        J[:, i] = cal_deriv(params, input_data, i, E, sigmay)
    return J


def cal_residual(params, input_data, output_data, E, sigmay):
    data_est_output = chaboche_model(input_data, params, E, sigmay)
    residual = output_data - data_est_output
    return residual


def LM_bounded_improved(num_iter, params, input_data, output_data, E, sigmay, bounds=None):
    num_params = len(params)
    k = 0
    
    if bounds is not None:
        params = apply_bounds(params, bounds)
    
    residual = cal_residual(params, input_data, output_data, E, sigmay)
    prev_error = np.linalg.norm(residual)
    
    mu = 0.01
    best_params = params.copy()
    best_error = prev_error

    while k < num_iter:
        try:
            Jacobian = cal_Jacobian(params, input_data, E, sigmay)
            A = Jacobian.T.dot(Jacobian)
            g = Jacobian.T.dot(residual)
            
            H = A + mu * np.eye(num_params)
            step = np.linalg.solve(H, g)
            
            if np.linalg.norm(step) <= 1e-15:
                break

            new_params = params + step
            
            if bounds is not None:
                new_params = apply_bounds(new_params, bounds)
            
            new_residual = cal_residual(new_params, input_data, output_data, E, sigmay)
            new_error = np.linalg.norm(new_residual)
            
            if new_error < prev_error:
                params = new_params
                residual = new_residual
                prev_error = new_error
                mu = mu * 0.7  
                
                if new_error < best_error:
                    best_error = new_error
                    best_params = params.copy()
                    
                if k % 10 == 0:
                    print(f"Iteration {k}: Error = {new_error:.6f}")
            else:
                mu = mu * 2.0  
                if mu > 1e6:  
                    break
        
        except np.linalg.LinAlgError:
            print("Matrix inversion failed, stopping optimization")
            break
        except Exception as e:
            print(f"Error in iteration {k}: {e}")
            break

        k += 1
    
    print(f"Optimization completed after {k} iterations")
    return best_params


def main():
    experiment_path = 'C:/Users/huo17/OneDrive - The University of Manchester/LCF/LM_Chaboche/sample_data/sample_data_1-10c.csv'
    
    E = 183000  # Young's Modulus
    sigmay = 264  # Yield point
    
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
    initial_guess = np.array([50000, 500, 10000, 100, 5000, 50, 50, 5])
    
    try:
        print("Loading data")
        strain, stress = load_data(experiment_path)
        print(f"Loaded {len(strain)} data points")
        
        fitted_params = LM_bounded_improved(50, initial_guess, strain, stress, E, sigmay, bounds)
        
        
        predicted_stress = chaboche_model(strain, fitted_params, E, sigmay)
        
        
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
        plt.plot(strain, predicted_stress, '-', label='Fitted Model', linewidth=2, color='red')
        plt.title("Fixed Chaboche Model Fit")
        plt.xlabel("Strain")
        plt.ylabel("Stress (MPa)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        textstr = f"RMSE: {rmse:.4f} MPa\nR^2: {r_squared:.4f}\nSSE: {sse:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()