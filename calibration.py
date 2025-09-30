# calibration.py
# Core pricing and calibration - production quality

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def bs_call_price(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def vg_mc_price(theta, sigma_vg, nu, S, K, T, r, n_paths=5000):
    from levy_simulation import simulate_vg_paths
    
    paths = simulate_vg_paths(theta, sigma_vg, nu, S0=S, T=T, r=r, 
                             n_paths=n_paths, n_steps=max(int(T*252), 50))
    payoffs = np.maximum(paths[:, -1] - K, 0)
    return np.mean(payoffs) * np.exp(-r * T)

def vg_call_fft(S, K, T, r, theta, sigma_vg, nu):
    # FFT pricing using Carr-Madan
    N = 4096
    eta = 0.15
    alpha = 1.1
    
    lambda_val = 2 * np.pi / (N * eta)
    b = N * lambda_val / 2
    k_u = -b + lambda_val * np.arange(N)
    v_j = eta * np.arange(N)
    
    # Martingale correction
    omega = -np.log(1.0 - theta*nu - 0.5*sigma_vg**2*nu) / nu
    
    # Characteristic function
    drift = 1j * v_j * (r - omega) * T
    vg = -T/nu * np.log(1.0 - 1j*v_j*theta*nu + 0.5*v_j**2*sigma_vg**2*nu)
    phi = np.exp(drift + vg)
    
    # Modified CF
    psi = phi * np.exp(-r*T) / ((alpha + 1j*v_j) * (alpha + 1 + 1j*v_j))
    
    # FFT
    w = np.ones(N) * eta
    w[0] = 0.5 * eta
    x = np.exp(1j * b * v_j) * w * psi
    y = np.fft.fft(x)
    
    call_fft = np.real(np.exp(-alpha * k_u) * y / np.pi)
    log_k = np.log(K / S)
    
    return np.interp(log_k, k_u, call_fft) * S

def calibrate_vg_to_single_option(option_data):
    S = option_data['stock_price']
    K = option_data['strike']
    T = option_data['time_to_expiration']
    r = option_data.get('risk_free_rate', 0.05)
    market = option_data['market_price']
    
    def objective(params):
        theta, sigma_vg, nu = params
        
        if nu <= 0.01 or sigma_vg <= 0.01:
            return 1e10
        if theta*nu + 0.5*sigma_vg**2*nu >= 0.95:
            return 1e10
        
        try:
            model = vg_mc_price(theta, sigma_vg, nu, S, K, T, r, 3000)
            error = ((model - market) / market) ** 2
            penalty = 0.01 * (theta**2 + (sigma_vg-0.2)**2 + (nu-0.3)**2)
            return error + penalty
        except:
            return 1e10
    
    bounds = [(-0.4, 0.4), (0.05, 0.6), (0.05, 0.9)]
    
    best = None
    best_err = np.inf
    
    for x0 in [[0, 0.2, 0.3], [-0.1, 0.15, 0.25], [0.1, 0.25, 0.4]]:
        try:
            res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            if res.success and res.fun < best_err:
                best = res
                best_err = res.fun
        except:
            continue
    
    if best and best_err < 1.0:
        return best.x
    
    # Fallback
    iv = option_data.get('implied_vol', 0.2)
    return np.array([-0.1*iv, iv*0.8, 0.3])