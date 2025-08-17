"""
Accurate Chaboche Model - Closely following the original LM_SA.py logic with Parameter Bounds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AccurateChabocheFitter:
    """Chaboche model following the exact logic from LM_SA.py with parameter bounds"""
    
    def __init__(self, E, sigmay):
        self.E = E
        self.sigmay = sigmay
        
        # Parameter bounds for physical validity
        self.param_bounds = {
            'C1': (10000, 100000),
            'r1': (100, 2000),
            'C2': (1000, 50000),
            'r2': (10, 500),
            'C3': (100, 10000),
            'r3': (1, 100),
            'Q': (10, 150),
            'b': (1, 10)
        }
        
        # Convert bounds to arrays for easier processing
        self.param_names = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
        self.lower_bounds = np.array([self.param_bounds[name][0] for name in self.param_names])
        self.upper_bounds = np.array([self.param_bounds[name][1] for name in self.param_names])
    
    def apply_bounds(self, params):
        """Apply parameter bounds by clipping values"""
        bounded_params = np.clip(params, self.lower_bounds, self.upper_bounds)
        return bounded_params
    
    def check_bounds_violation(self, params):
        """Check if parameters violate bounds and report violations"""
        violations = []
        for i, name in enumerate(self.param_names):
            if params[i] < self.lower_bounds[i]:
                violations.append(f"{name} = {params[i]:.4f} < {self.lower_bounds[i]} (lower bound)")
            elif params[i] > self.upper_bounds[i]:
                violations.append(f"{name} = {params[i]:.4f} > {self.upper_bounds[i]} (upper bound)")
        
        if violations:
            print("Parameter bound violations detected:")
            for violation in violations:
                print(f"  - {violation}")
        
        return len(violations) > 0
    
    def chaboche_model(self, strain, params):
        """Exact implementation following LM_SA.py logic"""
        C1, r1, C2, r2, C3, r3, Q, b = params
        alphas = np.zeros((len(strain), 3))  
        sigma = np.zeros(len(strain))
        ep = np.zeros(len(strain))
        R = np.zeros(len(strain))

        for i in range(len(strain)-1):
            et_dot = strain[i+1] - strain[i]
            lamda_dot = 0
            T_sdot = self.E * et_dot  
            T_stress = sigma[i] + T_sdot  
            T_alpha = alphas[i]  
            f = abs(T_stress - sum(T_alpha)) - (self.sigmay + R[i])

            # Exact same sign function as LM_SA.py
            sign_sig = np.sign(T_stress - sum(T_alpha))  

            if f <= 0:
                # IMPORTANT: Keep the same logic as LM_SA.py - including R update in elastic
                sigma[i+1] = sigma[i] + T_sdot
                alphas[i+1] = T_alpha
                ep[i+1] = ep[i]
                R[i+1] = R[i] + b * (Q - R[i]) * lamda_dot  # lamda_dot = 0, so R[i+1] = R[i]
            else:
                # Exact same as LM_SA.py - using tuple for C_alpha
                C_alpha = (C1 - sign_sig * r1 * T_alpha[0],
                           C2 - sign_sig * r2 * T_alpha[1],
                           C3 - sign_sig * r3 * T_alpha[2])
                denominator = self.E + sum(C_alpha) + b * (Q - R[i])
                lamda_dot = f / denominator
                ep_dot = sign_sig * lamda_dot

                ep[i+1] = ep[i] + ep_dot
                # CRITICAL: Use np.multiply exactly as in LM_SA.py
                alphas[i+1] = T_alpha + np.multiply(C_alpha, ep_dot)
                sigma[i+1] = T_stress - self.E * lamda_dot * sign_sig
                R[i+1] = R[i] + b * (Q - R[i]) * lamda_dot

        return sigma
    
    def cal_deriv(self, params, strain, param_index):
        """Exact derivative calculation from LM_SA.py with bounds consideration"""
        h = 1e-8  # Same step size as original
        params1 = params.copy()
        params2 = params.copy()
        
        # Ensure derivative calculation respects bounds
        if params1[param_index] + h > self.upper_bounds[param_index]:
            h_plus = self.upper_bounds[param_index] - params1[param_index]
        else:
            h_plus = h
            
        if params2[param_index] - h < self.lower_bounds[param_index]:
            h_minus = params2[param_index] - self.lower_bounds[param_index]
        else:
            h_minus = h
        
        params1[param_index] += h_plus
        params2[param_index] -= h_minus
        
        sigma1 = self.chaboche_model(strain, params1)
        sigma2 = self.chaboche_model(strain, params2)
        
        return (sigma1 - sigma2) / (h_plus + h_minus)
    
    def cal_Jacobian(self, params, input_data):
        """Exact Jacobian calculation from LM_SA.py"""
        num_params = len(params)
        num_data = len(input_data)
        J = np.zeros((num_data, num_params))
        for i in range(num_params):
            J[:, i] = self.cal_deriv(params, input_data, i)
        return J
    
    def cal_residual(self, params, input_data, output_data):
        """Exact residual calculation from LM_SA.py"""
        data_est_output = self.chaboche_model(input_data, params)
        residual = output_data - data_est_output
        return residual
    
    def cal_Hessian_LM(self, Jacobian, u, num_params):
        """Exact Hessian calculation from LM_SA.py"""
        H = Jacobian.T.dot(Jacobian) + u * np.eye(num_params)
        return H
    
    def cal_g(self, Jacobian, residual):
        """Exact gradient calculation from LM_SA.py"""
        g = Jacobian.T.dot(residual)
        return g
    
    def cal_step(self, Hessian_LM, g):
        """Exact step calculation from LM_SA.py"""
        try:
            s = np.linalg.solve(Hessian_LM, g)
        except np.linalg.LinAlgError:
            s = np.linalg.pinv(Hessian_LM).dot(g)
            print("Used pseudo-inverse due to singular matrix.")
        return s
    
    def project_step_to_bounds(self, params, step):
        """Project the step to ensure new parameters stay within bounds"""
        new_params = params + step
        
        # Check which parameters would violate bounds
        lower_violations = new_params < self.lower_bounds
        upper_violations = new_params > self.upper_bounds
        
        # For parameters that would violate lower bounds
        for i in np.where(lower_violations)[0]:
            # Reduce step to reach exactly the lower bound
            if step[i] < 0:  # Only if step is moving towards violation
                step[i] = self.lower_bounds[i] - params[i]
        
        # For parameters that would violate upper bounds
        for i in np.where(upper_violations)[0]:
            # Reduce step to reach exactly the upper bound
            if step[i] > 0:  # Only if step is moving towards violation
                step[i] = self.upper_bounds[i] - params[i]
        
        return step
    
    def LM_with_bounds(self, num_iter, params, input_data, output_data):
        """Enhanced LM algorithm with parameter bounds"""
        num_params = len(params)
        k = 0
        
        # Apply initial bounds
        params = self.apply_bounds(params)
        print(f"Initial parameters (after bounds): {params}")
        
        residual = self.cal_residual(params, input_data, output_data)
        Jacobian = self.cal_Jacobian(params, input_data)

        print(f"Initial error: {np.linalg.norm(residual):.6f}")
        
        # Track best parameters
        best_params = params.copy()
        best_error = np.linalg.norm(residual)

        while k < num_iter:
            A = Jacobian.T.dot(Jacobian)
            g = self.cal_g(Jacobian, residual)
            u = 10**-1 * np.max(np.diag(A))  # Exact same damping initialization
            Hessian_LM = self.cal_Hessian_LM(Jacobian, u, num_params)
            step = self.cal_step(Hessian_LM, g)

            if np.linalg.norm(step) <= 10**-15:
                print(f"Converged at iteration {k}: step size too small")
                break

            # Project step to respect bounds
            projected_step = self.project_step_to_bounds(params, step)
            new_params = params + projected_step
            
            # Ensure bounds are respected (safety check)
            new_params = self.apply_bounds(new_params)
            
            new_residual = self.cal_residual(new_params, input_data, output_data)
            new_error = np.linalg.norm(new_residual)
            
            # Track best solution
            if new_error < best_error:
                best_error = new_error
                best_params = new_params.copy()
            
            # Exact same ρ (rho) calculation as LM_SA.py but with projected step
            if np.linalg.norm(projected_step) > 1e-15:
                rou = (np.linalg.norm(residual)**2 - np.linalg.norm(new_residual)**2) / projected_step.T.dot(u * projected_step + g)
            else:
                rou = -1  # Force rejection if step is too small

            if rou > 0:
                params = new_params
                residual = new_residual
                Jacobian = self.cal_Jacobian(params, input_data)
                u *= 0.1  # Exact same damping reduction
                
                if k % 20 == 0:
                    print(f"Iteration {k:3d}: Error = {np.linalg.norm(residual):.6f}, ρ = {rou:.4f}")
                    # Check and report bound violations
                    self.check_bounds_violation(params)
            else:
                u *= 10  # Exact same damping increase
                if u > 1e8:  # Prevent excessive damping
                    print("Damping factor too large, stopping optimization")
                    break

            k += 1

        print(f"Optimization completed after {k} iterations")
        print(f"Using best parameters with error: {best_error:.6f}")
        
        # Final bounds check
        print("\nFinal parameter bounds check:")
        self.check_bounds_violation(best_params)
        
        return best_params
    
    def calculate_errors(self, experimental_stress, predicted_stress):
        """Calculate error metrics"""
        mask = np.isfinite(experimental_stress) & np.isfinite(predicted_stress)
        exp_clean = experimental_stress[mask]
        pred_clean = predicted_stress[mask]
        
        if len(exp_clean) == 0:
            return float('inf'), float('inf'), 0.0
        
        sse = np.sum((exp_clean - pred_clean)**2)
        rmse = np.sqrt(np.mean((exp_clean - pred_clean)**2))
        ss_tot = np.sum((exp_clean - np.mean(exp_clean))**2)
        r_squared = 1 - (sse / ss_tot) if ss_tot > 0 else 0.0
        
        return sse, rmse, r_squared
    
    def fit_and_plot(self, strain_data, stress_data, initial_params, max_iterations=1000):
        """Complete fitting and plotting workflow with bounds"""
        print("Starting accurate Chaboche model fitting with parameter bounds...")
        print(f"Initial parameters: {initial_params}")
        
        # Display bounds information
        print("\nParameter bounds:")
        for i, name in enumerate(self.param_names):
            lower, upper = self.param_bounds[name]
            initial_val = initial_params[i]
            status = "OK"
            if initial_val < lower or initial_val > upper:
                status = "VIOLATION"
            print(f"  {name}: [{lower:8.0f}, {upper:8.0f}] - Initial: {initial_val:8.1f} ({status})")
        
        # Fit parameters using bounded LM algorithm
        fitted_params = self.LM_with_bounds(max_iterations, initial_params, strain_data, stress_data)
        
        # Calculate predictions and errors
        predicted_stress = self.chaboche_model(strain_data, fitted_params)
        sse, rmse, r_squared = self.calculate_errors(stress_data, predicted_stress)
        
        # Display results
        print(f"\nFitted Parameters:")
        for name, value in zip(self.param_names, fitted_params):
            lower, upper = self.param_bounds[name]
            pct_range = ((value - lower) / (upper - lower)) * 100
            print(f"  {name}: {value:8.4f} [{lower:6.0f}, {upper:6.0f}] ({pct_range:5.1f}% of range)")
        
        print(f"\nError Metrics:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R^2:  {r_squared:.6f}")
        print(f"  SSE:  {sse:.6f}")
        
        # Plot results with same style as LM_SA.py
        plt.figure(figsize=(12, 8))
        
        # Downsample experimental data for clarity (like in LM_SA.py)
        exp_indices = np.arange(0, len(strain_data), 10)
        plt.plot(strain_data, predicted_stress, '-', label='Fitted Curve', alpha=0.9, zorder=1, linewidth=2, color='red')
        plt.scatter(strain_data[exp_indices], stress_data[exp_indices], s=15, color='blue',
                   label='Experimental Data (Sparse Dots)', alpha=0.8, zorder=3)

        plt.xlabel('Strain', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Chaboche Model Fit with Parameter Bounds', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add error metrics and bounds info
        textstr = f"RMSE: {rmse:.4f} MPa\nR^2: {r_squared:.4f}\nSSE: {sse:.4f}\n\nBounded Parameters:\n"
        for name, value in zip(self.param_names[:4], fitted_params[:4]):  # Show first 4 parameters
            textstr += f"{name}: {value:.1f}\n"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'parameters': fitted_params,
            'parameter_names': self.param_names,
            'predicted_stress': predicted_stress,
            'sse': sse,
            'rmse': rmse,
            'r_squared': r_squared,
            'bounds': self.param_bounds
        }


def load_data_from_csv(filename):
    """Load strain and stress data from CSV file"""
    try:
        data = pd.read_csv(filename)
        
        if 'Strain 1 (%)' not in data.columns or 'Stress' not in data.columns:
            raise ValueError("CSV must contain 'Strain 1 (%)' and 'Stress' columns")
        
        strain_data = data['Strain 1 (%)'].values / 100
        stress_data = data['Stress'].values
        
        print(f"Loaded {len(strain_data)} data points from {filename}")
        print(f"Strain range: {strain_data.min():.4f} to {strain_data.max():.4f}")
        print(f"Stress range: {stress_data.min():.1f} to {stress_data.max():.1f} MPa")
        
        return strain_data, stress_data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


# Example usage
if __name__ == "__main__":
    # Load data
    filename = r"C:/Users/huo17/OneDrive - The University of Manchester/LCF/LM_Chaboche/sample_data/sample_data_1-10c.csv"
    strain_data, stress_data = load_data_from_csv(filename)
    
    if strain_data is None:
        exit()
    
    # Material properties - using same as your working version
    E = 183000  # MPa
    sigmay = 264  # MPa
    
    print(f"\nMaterial Properties:")
    print(f"  E = {E} MPa")
    print(f"  σy = {sigmay} MPa")
    
    # Create accurate fitter with bounds
    fitter = AccurateChabocheFitter(E, sigmay)
    
    # Initial parameters - same as working version
    initial_params = np.array([50000, 500, 10000, 100, 5000, 50, 50, 5])
    
    # Fit and display results
    results = fitter.fit_and_plot(strain_data, stress_data, initial_params, max_iterations=200)