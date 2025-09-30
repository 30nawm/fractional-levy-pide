# config.py
# Global configuration - minimal and functional

import numpy as np

# Defaults (updated dynamically)
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
alpha = 0.85
theta = -0.12
sigma_vg = 0.18
nu = 0.25

def update_config_with_option_data(option_data):
    global S0, K, T, sigma
    
    S0 = option_data['stock_price']
    K = option_data['strike']
    T = max(option_data['time_to_expiration'], 0.01)
    sigma = option_data.get('implied_vol', 0.2)

def validate_all_parameters():
    issues = []
    
    if S0 <= 0 or K <= 0 or T <= 0:
        issues.append("Market parameters invalid")
    
    if not (0.5 < alpha < 0.99):
        issues.append(f"Alpha {alpha} out of range")
    
    mg = theta * nu + 0.5 * sigma_vg**2 * nu
    if mg >= 1.0:
        issues.append(f"Martingale condition violated: {mg:.3f}")
    
    return len(issues) == 0, issues