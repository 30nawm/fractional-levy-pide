# pide_solver.py v2
# Mathematically corrected time-fractional PIDE solver
# Based on rigorous numerical methods from academic literature
# All comments and documentation in pure English

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import gamma
from config import S0, K, T, r, alpha, sigma, n_t, n_s

class FractionalPIDESolver:
    """
    Corrected implementation of time-fractional PIDE for option pricing
    
    Solves: D^α_t V = LV + (r-q)S∂V/∂S + (1/2)σ²S²∂²V/∂S² - rV
    where LV is the Lévy integral operator and D^α_t is Caputo fractional derivative
    """
    
    def __init__(self, S0, K, T, r, sigma, alpha, n_t=100, n_s=200):
        self.S0 = S0
        self.K = K  
        self.T = T
        self.r = r
        self.sigma = sigma
        self.alpha = alpha
        self.n_t = n_t
        self.n_s = n_s
        
        # Ensure minimum time to avoid numerical issues
        self.T = max(T, 0.05)  # Minimum 18 days
        
        # Spatial grid setup
        self.S_max = 2.0 * max(S0, K)
        self.S_min = 0.0
        self.S_grid = np.linspace(self.S_min, self.S_max, n_s)
        self.dS = self.S_grid[1] - self.S_grid[0]
        
        # Time grid
        self.dt = T / n_t
        
        # Find index of S0 for final interpolation
        self.idx_S0 = np.argmin(np.abs(self.S_grid - S0))
        
        print(f"PIDE Solver initialized:")
        print(f"  Time to expiration: {self.T:.4f} years")
        print(f"  Spatial domain: [${self.S_min:.0f}, ${self.S_max:.0f}]")
        print(f"  Grid size: {n_t} × {n_s}")
    
    def vg_levy_density_correct(self, y, theta, sigma_vg, nu):
        """
        Correct implementation of Variance Gamma Lévy density
        Based on Madan-Carr-Chang (1998) formulation
        """
        if abs(y) < 1e-10 or nu <= 0:
            return 0.0
        
        # VG Lévy measure: C * exp(G*y)/|y| for y<0, C * exp(-M*y)/|y| for y>0
        # Parameters: G = (-θ + √(θ² + 2σ²/ν))*ν/σ², M = (θ + √(θ² + 2σ²/ν))*ν/σ²
        
        discriminant = theta**2 + 2 * sigma_vg**2 / nu
        sqrt_disc = np.sqrt(discriminant)
        
        if y > 0:
            M = (theta + sqrt_disc) * nu / sigma_vg**2
            return (1.0 / nu) * np.exp(-M * y) / y
        else:
            G = (-theta + sqrt_disc) * nu / sigma_vg**2  
            return (1.0 / nu) * np.exp(G * y) / (-y)
    
    def compute_levy_integral(self, V, theta, sigma_vg, nu):
        """
        Compute Lévy integral operator using corrected numerical integration
        """
        result = np.zeros_like(V)
        
        # Integration domain for jumps (log-returns)
        y_max = 2.0
        n_y = 80
        y_grid = np.linspace(-y_max, y_max, n_y)
        dy = y_grid[1] - y_grid[0]
        
        for i, S_i in enumerate(self.S_grid):
            integral = 0.0
            
            for y_j in y_grid:
                if abs(y_j) < 1e-8:  # Skip singularity
                    continue
                
                # Price after jump
                S_new = S_i * np.exp(y_j)
                
                # Interpolate V at new price
                if S_new <= self.S_grid[0]:
                    V_new = 0.0
                elif S_new >= self.S_grid[-1]:
                    V_new = max(S_new - self.K, 0)
                else:
                    # Linear interpolation
                    idx = np.searchsorted(self.S_grid, S_new) - 1
                    idx = max(0, min(idx, len(self.S_grid) - 2))
                    w = (S_new - self.S_grid[idx]) / (self.S_grid[idx + 1] - self.S_grid[idx])
                    V_new = (1 - w) * V[idx] + w * V[idx + 1]
                
                # Lévy measure contribution
                levy_measure = self.vg_levy_density_correct(y_j, theta, sigma_vg, nu)
                integral += (V_new - V[i]) * levy_measure * dy
            
            result[i] = integral
        
        return result
    
    def caputo_derivative_corrected(self, V_history, n):
        """
        Corrected L1 approximation for Caputo fractional derivative
        Based on Sun-Wu (2006) and Podlubny (1999)
        """
        if n == 0 or len(V_history) < 2:
            return np.zeros_like(V_history[-1])
        
        # L1 coefficients: a_k = (k+1)^(1-α) - k^(1-α) for k = 0, 1, ..., n-1
        coeffs = np.array([(k + 1)**(1 - self.alpha) - k**(1 - self.alpha) 
                          for k in range(n)])
        
        # Compute fractional derivative
        frac_deriv = np.zeros_like(V_history[0])
        
        for k in range(n):
            if n - k - 1 < len(V_history) and n - k < len(V_history):
                diff = V_history[n - k] - V_history[n - k - 1]
                frac_deriv += coeffs[k] * diff
        
        # Scale by dt^(-α) / Γ(2-α)
        factor = self.dt**(-self.alpha) / gamma(2 - self.alpha)
        
        return frac_deriv * factor
    
    def build_finite_difference_operator(self):
        """
        Build finite difference operator for spatial derivatives
        """
        # Second derivative coefficients
        a = 0.5 * self.sigma**2 * self.S_grid**2 / self.dS**2
        
        # First derivative coefficients
        b = self.r * self.S_grid / (2 * self.dS)
        
        # Build tridiagonal matrix for spatial operator
        lower_diag = -a[:-1] + b[:-1]  # Lower diagonal
        main_diag = 2 * a + self.r      # Main diagonal (including discount)
        upper_diag = -a[1:] - b[1:]     # Upper diagonal
        
        return lower_diag, main_diag, upper_diag
    
    def solve_fractional_pide(self, theta, sigma_vg, nu):
        """
        Main solver for time-fractional PIDE
        """
        print("Starting fractional PIDE solution...")
        
        # Terminal condition (European call payoff)
        V = np.maximum(self.S_grid - self.K, 0.0)
        V_history = [V.copy()]
        
        # Build finite difference operator
        lower_diag, main_diag, upper_diag = self.build_finite_difference_operator()
        
        # Time stepping loop
        for n in range(1, self.n_t + 1):
            current_time = self.T - n * self.dt
            
            # Compute Caputo fractional derivative
            D_alpha_V = self.caputo_derivative_corrected(V_history, n)
            
            # Compute Lévy integral
            levy_term = self.compute_levy_integral(V_history[-1], theta, sigma_vg, nu)
            
            # Build system matrix (implicit scheme)
            A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
            
            # Right-hand side
            RHS = V_history[-1] + self.dt * (D_alpha_V + levy_term)
            
            # Apply boundary conditions
            # Lower boundary: V(0, t) = 0
            A[0, :] = 0
            A[0, 0] = 1
            RHS[0] = 0
            
            # Upper boundary: V(S_max, t) = S_max - K*exp(-r*τ)
            A[-1, :] = 0  
            A[-1, -1] = 1
            RHS[-1] = max(self.S_grid[-1] - self.K * np.exp(-self.r * current_time), 0)
            
            # Solve linear system
            V_new = spsolve(A, RHS)
            
            # Ensure non-negative option values
            V_new = np.maximum(V_new, 0)
            
            # Store solution
            V_history.append(V_new.copy())
            
            # Memory management
            if len(V_history) > self.n_t + 2:
                V_history.pop(0)
        
        # Final option price at S0
        final_V = V_history[-1]
        option_price = np.interp(self.S0, self.S_grid, final_V)
        
        print(f"PIDE solution completed. Price: ${option_price:.4f}")
        
        return max(option_price, 0), self.S_grid, final_V

def fractional_pide_solver(theta, sigma_vg, nu):
    """
    Main interface function for fractional PIDE solver
    """
    # Handle very short time to expiration
    T_adjusted = max(T, 0.05)  # Minimum 18 days
    
    if T < 0.05:
        print(f"WARNING: Very short time to expiration ({T:.4f} years) adjusted to {T_adjusted:.4f}")
    
    # Create and run solver
    solver = FractionalPIDESolver(S0, K, T_adjusted, r, sigma, alpha, n_t=80, n_s=150)
    
    return solver.solve_fractional_pide(theta, sigma_vg, nu)