# pide_solver.py
# Corrected Time-fractional PIDE solver for Lévy-driven option pricing
# Based on research papers and proven numerical methods
# Fixed: Proper boundary conditions, stability, and mathematical formulation

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import gamma
from scipy.integrate import quad
from config import S0, K, T, r, alpha, sigma, n_t, n_s

def vg_levy_measure(y, theta, sigma_vg, nu):
    """
    Variance Gamma Lévy measure density
    Based on Madan-Carr-Chang (1998) formulation
    """
    if abs(y) < 1e-10:
        return 0.0
    
    # VG Lévy measure: ν(dy) = C * exp(G*y) / |y| * dy for y < 0
    #                        = C * exp(-M*y) / |y| * dy for y > 0
    # where G, M are positive parameters related to θ, σ, ν
    
    # Convert VG parameters to G, M parametrization
    # From Cont-Tankov (2004) Financial Modelling with Jump Processes
    alpha_pos = (theta + np.sqrt(theta**2 + 2*sigma_vg**2/nu)) * nu / sigma_vg**2
    alpha_neg = (-theta + np.sqrt(theta**2 + 2*sigma_vg**2/nu)) * nu / sigma_vg**2
    
    C = 1.0 / nu  # Normalizing constant
    
    if y > 0:
        return C * np.exp(-alpha_pos * y) / y
    else:
        return C * np.exp(alpha_neg * y) / (-y)

def levy_integral_corrected(V, S_grid, theta, sigma_vg, nu):
    """
    Corrected Lévy integral operator using proper numerical integration
    Based on Cont-Voltchkova (2005) finite difference scheme
    """
    result = np.zeros_like(V)
    dS = S_grid[1] - S_grid[0]
    
    # Integration bounds for jump sizes (in log space)
    y_min, y_max = -3.0, 3.0  # Reasonable bounds for VG jumps
    n_y = 100  # Integration points
    y_grid = np.linspace(y_min, y_max, n_y)
    dy = (y_max - y_min) / (n_y - 1)
    
    for i in range(len(S_grid)):
        S_i = S_grid[i]
        integral = 0.0
        
        for y_j in y_grid:
            if abs(y_j) < 1e-8:  # Skip near zero to avoid singularity
                continue
                
            # Price after jump: S_new = S * exp(y)
            S_new = S_i * np.exp(y_j)
            
            # Find V(S_new) by interpolation
            if S_new <= S_grid[0]:
                V_new = 0.0  # Below grid: worthless
            elif S_new >= S_grid[-1]:
                # Above grid: linear extrapolation or intrinsic value
                V_new = max(S_new - K, 0)
            else:
                # Linear interpolation
                idx = np.searchsorted(S_grid, S_new) - 1
                idx = max(0, min(idx, len(S_grid)-2))
                
                w = (S_new - S_grid[idx]) / (S_grid[idx+1] - S_grid[idx])
                V_new = (1-w) * V[idx] + w * V[idx+1]
            
            # Lévy measure contribution
            levy_density = vg_levy_measure(y_j, theta, sigma_vg, nu)
            
            # Integral contribution: (V(S*exp(y)) - V(S)) * ν(dy)
            integral += (V_new - V[i]) * levy_density * dy
        
        result[i] = integral
    
    return result

def caputo_derivative_l1(V_history, dt, alpha):
    """
    L1 approximation of Caputo fractional derivative
    Based on Podlubny (1999) and Sun-Wu (2006)
    """
    n = len(V_history) - 1  # Current time index
    if n == 0:
        return np.zeros_like(V_history[0])
    
    # L1 weights: a_j = (j+1)^(1-α) - j^(1-α) for j = 0, 1, ..., n-1
    weights = np.array([(j+1)**(1-alpha) - j**(1-alpha) for j in range(n)])
    weights = weights / gamma(2-alpha)
    
    # Compute fractional derivative: sum_{j=0}^{n-1} a_j * (V^{n-j} - V^{n-j-1})
    frac_deriv = np.zeros_like(V_history[0])
    for j in range(n):
        frac_deriv += weights[j] * (V_history[n-j] - V_history[n-j-1])
    
    return frac_deriv * dt**(-alpha)

def fractional_pide_solver(theta, sigma_vg, nu):
    """
    Solve time-fractional PIDE for European call option
    Using implicit finite difference with L1 scheme for fractional derivative
    
    PDE: D^α_t V = (σ²S²/2)V_SS + rSV_S - rV + ∫(V(S*e^y) - V(S))ν(dy)
    """
    
    # Spatial grid (wider range for stability)
    S_min, S_max = 0.0, 3.0 * S0
    S_grid = np.linspace(S_min, S_max, n_s)
    dS = S_grid[1] - S_grid[0]
    
    # Find index closest to S0 for final interpolation
    idx_S0 = np.argmin(np.abs(S_grid - S0))
    
    # Terminal condition: European call payoff
    V = np.maximum(S_grid - K, 0.0)
    V_history = [V.copy()]
    
    # Time step
    dt = T / n_t
    
    # Build spatial operator matrix (time-independent part)
    # Second derivative coefficients
    a = sigma**2 * S_grid**2 / (2 * dS**2)
    # First derivative coefficients  
    b = r * S_grid / (2 * dS)
    
    # Interior points finite difference stencil
    main_diag = np.ones(n_s) + dt * r  # Identity + discount term
    upper_diag = -dt * (a[1:] + b[1:])  # -dt*(a + b/2)
    lower_diag = -dt * (a[:-1] - b[:-1])  # -dt*(a - b/2)
    
    # Add second derivative term to main diagonal
    main_diag += dt * 2 * a
    
    # Time stepping (backward Euler with L1 for fractional part)
    for n in range(1, n_t + 1):
        current_time = T - n * dt
        
        # Compute fractional derivative from history
        D_alpha_V = caputo_derivative_l1(V_history, dt, alpha)
        
        # Compute Lévy integral term
        levy_term = levy_integral_corrected(V_history[-1], S_grid, theta, sigma_vg, nu)
        
        # System matrix A*V^{n+1} = RHS
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
        
        # Right-hand side: V^n + dt*(fractional_derivative + levy_term)
        RHS = V_history[-1] + dt * D_alpha_V + dt * levy_term
        
        # Boundary conditions
        # Left boundary: V(0,t) = 0 (call option worthless at S=0)
        A[0, :] = 0
        A[0, 0] = 1
        RHS[0] = 0
        
        # Right boundary: V(S_max,t) = S_max - K*exp(-r*τ) (deep ITM)
        A[-1, :] = 0
        A[-1, -1] = 1
        RHS[-1] = max(S_grid[-1] - K * np.exp(-r * current_time), 0)
        
        # Solve system
        V_new = spsolve(A, RHS)
        
        # Ensure non-negative values (option prices must be non-negative)
        V_new = np.maximum(V_new, 0)
        
        # Update history
        V_history.append(V_new)
        if len(V_history) > n_t + 5:  # Keep reasonable history
            V_history.pop(0)
    
    # Final option price at S0 by interpolation
    final_V = V_history[-1]
    option_price = np.interp(S0, S_grid, final_V)
    
    return max(option_price, 0), S_grid, final_V