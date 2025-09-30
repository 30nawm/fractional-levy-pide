# levy_simulation.py
# Corrected Variance Gamma path simulation with proper martingale correction
# Based on Madan et al. (1998) and Cont-Tankov (2004)

import numpy as np

def simulate_vg_paths(theta, sigma_vg, nu, S0=100.0, T=1.0, r=0.05, 
                     n_paths=1000, n_steps=252):
    """
    Simulate Variance Gamma process paths with correct implementation
    
    VG process: X(t) = theta*G(t) + sigma*W(G(t))
    where G(t) ~ Gamma(t/nu, nu) is subordinator
    
    Parameters:
    -----------
    theta : float
        Drift parameter of Brownian motion
    sigma_vg : float  
        Volatility parameter of Brownian motion
    nu : float
        Variance rate parameter (controls jump activity)
    S0 : float
        Initial stock price
    T : float
        Time horizon in years
    r : float
        Risk-free rate
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
        
    Returns:
    --------
    paths : ndarray
        Array of shape (n_paths, n_steps+1) with simulated prices
    """
    
    # Ensure positive parameters
    sigma_vg = max(sigma_vg, 1e-6)
    nu = max(nu, 1e-6)
    
    # Time step
    dt = T / n_steps
    
    # Martingale correction (omega) from characteristic function
    # E[exp(X(t))] = exp(r*t) requires this correction
    # omega = -log(1 - theta*nu - 0.5*sigma_vg^2*nu) / nu
    
    correction_term = theta * nu + 0.5 * sigma_vg**2 * nu
    
    # Check for numerical stability
    if correction_term >= 0.99:
        # Parameter combination too extreme, apply damping
        correction_term = 0.99
        print(f"Warning: VG parameters near stability boundary, applying correction")
    
    omega = -np.log(1.0 - correction_term) / nu
    
    # Drift per time step (includes martingale correction)
    drift = (r - omega) * dt
    
    # Initialize path matrix
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate gamma subordinator increments
    # G(dt) ~ Gamma(shape=dt/nu, scale=nu)
    gamma_shape = dt / nu
    gamma_scale = nu
    
    # Pre-generate all random numbers for efficiency
    gamma_increments = np.random.gamma(gamma_shape, gamma_scale, 
                                       size=(n_paths, n_steps))
    
    # Generate Brownian motion components
    Z = np.random.randn(n_paths, n_steps)
    
    # Simulate paths
    for t in range(n_steps):
        # Gamma time increment
        g_t = gamma_increments[:, t]
        
        # VG increment: theta*g + sigma*sqrt(g)*Z
        X_increment = theta * g_t + sigma_vg * np.sqrt(g_t) * Z[:, t]
        
        # Update log-price with drift and VG increment
        log_S = np.log(paths[:, t]) + drift + X_increment
        
        # Convert back to price
        paths[:, t + 1] = np.exp(log_S)
        
        # Numerical stability check
        paths[:, t + 1] = np.clip(paths[:, t + 1], 1e-10, 1e10)
    
    return paths


def simulate_standard_gbm(S0, mu, sigma, T, n_paths=1000, n_steps=252):
    """
    Standard Geometric Brownian Motion for comparison/testing
    
    dS/S = mu*dt + sigma*dW
    """
    dt = T / n_steps
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate random increments
    Z = np.random.randn(n_paths, n_steps)
    
    for t in range(n_steps):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, t]
        
        log_S = np.log(paths[:, t]) + drift + diffusion
        paths[:, t + 1] = np.exp(log_S)
    
    return paths


def validate_vg_parameters(theta, sigma_vg, nu):
    """
    Validate VG parameters for mathematical consistency
    
    Returns: (is_valid, message)
    """
    issues = []
    
    # Check positivity
    if sigma_vg <= 0:
        issues.append("sigma_vg must be positive")
    if nu <= 0:
        issues.append("nu must be positive")
    
    # Check moment existence
    # Second moment exists if nu*theta^2 + nu*sigma_vg^2 < infinity (always true)
    # Fourth moment exists roughly if nu < 0.5 (rule of thumb)
    if nu > 1.0:
        issues.append("nu > 1.0 may cause slow convergence in simulations")
    
    # Check martingale condition
    correction_term = theta * nu + 0.5 * sigma_vg**2 * nu
    if correction_term >= 1.0:
        issues.append("Parameters violate martingale condition: theta*nu + 0.5*sigma_vg^2*nu >= 1")
    
    # Check practical bounds
    if abs(theta) > 1.0:
        issues.append("Large |theta| > 1 may indicate calibration issues")
    if sigma_vg > 1.0:
        issues.append("Large sigma_vg > 1 may indicate calibration issues")
    
    if issues:
        return False, "; ".join(issues)
    else:
        return True, "Parameters valid"


def compute_vg_moments(theta, sigma_vg, nu, t=1.0):
    """
    Compute theoretical moments of VG distribution
    Useful for validation and diagnostics
    
    Returns: dict with mean, variance, skewness, excess kurtosis
    """
    # Mean of X(t)
    mean = theta * t
    
    # Variance of X(t)
    variance = (sigma_vg**2 + theta**2 * nu) * t
    
    # Skewness
    if variance > 0:
        skewness = (theta * nu * (3*sigma_vg**2 + 2*theta**2*nu)) / (variance**(3/2))
    else:
        skewness = 0
    
    # Excess kurtosis
    if variance > 0:
        kurtosis = (3*nu*(sigma_vg**4 + 4*theta**2*sigma_vg**2*nu + 2*theta**4*nu**2)) / (variance**2)
    else:
        kurtosis = 0
    
    return {
        'mean': mean,
        'variance': variance,
        'std': np.sqrt(variance),
        'skewness': skewness,
        'excess_kurtosis': kurtosis
    }


# Testing and validation
if __name__ == "__main__":
    print("Testing VG simulation module...")
    
    # Test parameters
    theta = -0.1
    sigma_vg = 0.2
    nu = 0.3
    
    # Validate
    valid, msg = validate_vg_parameters(theta, sigma_vg, nu)
    print(f"\nParameter validation: {valid}")
    print(f"Message: {msg}")
    
    # Compute theoretical moments
    moments = compute_vg_moments(theta, sigma_vg, nu)
    print(f"\nTheoretical moments (t=1):")
    print(f"  Mean: {moments['mean']:.4f}")
    print(f"  Std: {moments['std']:.4f}")
    print(f"  Skewness: {moments['skewness']:.4f}")
    print(f"  Excess Kurtosis: {moments['excess_kurtosis']:.4f}")
    
    # Simulate paths
    print("\nSimulating paths...")
    paths = simulate_vg_paths(theta, sigma_vg, nu, n_paths=5000, n_steps=252)
    
    # Compute empirical moments from terminal values
    terminal_log_returns = np.log(paths[:, -1] / paths[:, 0])
    
    print(f"\nEmpirical moments from simulation:")
    print(f"  Mean: {np.mean(terminal_log_returns):.4f}")
    print(f"  Std: {np.std(terminal_log_returns):.4f}")
    print(f"  Skewness: {np.mean(((terminal_log_returns - np.mean(terminal_log_returns))/np.std(terminal_log_returns))**3):.4f}")
    
    print("\nTest completed successfully!")