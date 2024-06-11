import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sigmay =   # yield stress
# E =   # Young's Modulus

def load_real_data(filepath): # load your experiment data here
    data = pd.read_csv(filepath)
    strain_data = data['Strain 1 (%)'].values / 100  # Careful with the unit of strain data, you need to convert percentages to decimals if your data is shown in percentages
    stress_data = data['Stress'].values
    return strain_data, stress_data

def chaboche_model(strain, params):
    C1, r1, C2, r2, C3, r3, Q, b = params # Notice that there is three backstresses in this case, you could choose 2 backstresses by deleting C3 and r3
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

        sign_sig = -1 if T_stress - sum(T_alpha) < 0 else 1 # Here is define if the stress state is under compression or tension

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

def cal_deriv(params, strain, param_index): # Calculate Numerical Derivative
    h = 0.000000000001 # You could change the factor here to balance the calculation time and accuracy
    params1 = params.copy()
    params2 = params.copy()
    params1[param_index] += h
    params2[param_index] -= h
    sigma1 = chaboche_model(strain, params1)
    sigma2 = chaboche_model(strain, params2)
    return (sigma1 - sigma2) / (2 * h)

def cal_Hessian_LM(Jacobian, u, num_params):
    # Include the damping factor in the Hessian calculation, u is the damping factor
    H = Jacobian.T.dot(Jacobian) + u * np.eye(num_params)
    return H

def cal_g(Jacobian, residual): # Calculate Gradient Vector
    # Calculate gradient g
    g = Jacobian.T.dot(residual)
    return g

def cal_step(Hessian_LM, g):
    # Calculate step using a robust method to handle potential singular matrices
    try:
        # Directly solving for s where H*s = g
        s = np.linalg.solve(Hessian_LM, g)
    except np.linalg.LinAlgError:
        # If the matrix is singular, use pseudo-inverse as a fallback
        s = np.linalg.pinv(Hessian_LM).dot(g)
        print("Used pseudo-inverse due to singular matrix.")
    return s

def cal_Jacobian(params, input_data): # Calculate Jacobian Matrix
    num_params = len(params)
    num_data = len(input_data)
    J = np.zeros((num_data, num_params))
    for i in range(num_params):
        J[:, i] = cal_deriv(params, input_data, i)
    return J

def cal_residual(params, input_data, output_data): 
    data_est_output = chaboche_model(input_data, params)
    residual = output_data - data_est_output
    return residual

def LM(num_iter, params, input_data, output_data): # Main LM Algorithm
    num_params = len(params)
    k = 0
    residual = cal_residual(params, input_data, output_data)
    Jacobian = cal_Jacobian(params, input_data)

    while k < num_iter:
        A = Jacobian.T.dot(Jacobian)
        g = cal_g(Jacobian, residual)
        u = 10**-1 * np.max(np.diag(A))  # Start with a small but non-zero damping factor
        Hessian_LM = cal_Hessian_LM(Jacobian, u, num_params)
        step = cal_step(Hessian_LM, g)

        if np.linalg.norm(step) <= 10**-15:
            break  # Stop if step size below threshold

        new_params = params + step
        new_residual = cal_residual(new_params, input_data, output_data)
        rou = (np.linalg.norm(residual)**2 - np.linalg.norm(new_residual)**2) / step.T.dot(u * step + g)

        if rou > 0:
            params = new_params
            residual = new_residual
            Jacobian = cal_Jacobian(params, input_data)
        else:
            u *= 10  # Increase damping factor if no improvement

        k += 1

    return params


def main(): # Fitting Process
    filepath = ''  # Your experiment data file path here
    strain, stress = load_real_data(filepath)
    # Here is the guessing for both kinematic parameters and isotropic parameters, following by C1,r1,C2,r2,C3,r3,Q,b)
    initial_guess = np.array([]) 
    
    
    fitted_params = LM(100, initial_guess, strain, stress)  # Adjust iterations as needed
    formatted_params = ", ".join(f"{x:.2f}" for x in fitted_params)
    print(f"Fitted Parameters: [{formatted_params}]")

    predicted_stress = chaboche_model(strain, fitted_params)
    plt.figure(figsize=(10, 5))
    plt.plot(strain, stress, 'o', label='Experimental Data')
    plt.plot(strain, predicted_stress, '-', label='Fitted')
    plt.title("Chaboche Model Fit to Experimental Data")
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.legend()
    plt.show()
    
    save_file_path = '' # The data after LM fitting
    fitted_data = np.column_stack((strain, predicted_stress))
    np.savetxt(save_file_path, fitted_data, delimiter=",", header="Strain,Stress", comments='')


if __name__ == '__main__': # Running Script
    main()
