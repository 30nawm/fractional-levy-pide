# levy_simulation.py
# Simulate Variance Gamma (VG) paths for Lévy process
# Vectorized for efficiency: O(n_paths * n_steps) with NumPy

import numpy as np
from config import S0, T, r, n_paths, n_steps

def vg_characteristic(t, theta, sigma_vg, nu):
    """Cumulant for VG martingale correction (Madan et al., 1998)"""
    omega = -np.log(1 - theta * nu - 0.5 * sigma_vg**2 * nu) / nu
    return (r - omega) * t

def simulate_vg_paths(theta, sigma_vg, nu, n_paths=n_paths, n_steps=n_steps, dt=T/n_steps):
    """Simulate VG paths: Brownian + Gamma jumps"""
    omega = -np.log(1 - theta * nu - 0.5 * sigma_vg**2 * nu) / nu
    drift = (r - omega) * dt
    
    # Time-changed Brownian motion: G ~ Gamma(dt/nu, nu)
    g_times = np.random.gamma(dt / nu, nu, size=(n_paths, n_steps))
    
    # BM part
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    dX_bm = theta * g_times + sigma_vg * np.sqrt(g_times) * dW
    
    # Total increment
    dX = drift + dX_bm
    logS = np.cumsum(dX, axis=1) + np.log(S0)
    S = np.exp(logS)
    return S