# config.py
# Optimized configuration for fractional Lévy PIDE solver
# Based on literature and numerical stability requirements

import numpy as np

# Market parameters - Standard test case
S0 = 100.0  # Initial stock price
K = 100.0   # Strike price (at-the-money)
T = 1.0     # Time to maturity (years)
r = 0.05    # Risk-free rate (5%)
sigma = 0.2 # Black-Scholes volatility (20%)

# Fractional PDE parameters - Optimized for stability
alpha = 0.85    # Caputo fractional order (0 < alpha < 1)
                # α = 0.85 provides good balance between memory effects
                # and numerical stability (based on Huang et al. 2010)

# Grid parameters - Balanced for accuracy vs computational cost
n_t = 150       # Time grid points (reduced for stability)
n_s = 250       # Space grid points (sufficient for convergence)

# Variance Gamma parameters - Literature-based initial values
# Based on Madan-Carr-Chang (1998) and Cont-Tankov (2004)
theta = -0.12   # VG drift parameter 
                # Negative values capture market negative skewness
                # Typical range: [-0.5, 0.1]

sigma_vg = 0.18 # VG volatility parameter
                # Controls the intensity of jumps
                # Typical range: [0.1, 0.5]

nu = 0.25       # VG variance rate
                # Controls jump frequency vs size trade-off  
                # Smaller ν = less frequent, larger jumps
                # Typical range: [0.1, 1.0]

# Simulation parameters - Optimized for accuracy
n_paths = 2000  # Monte Carlo paths (increased for better convergence)
n_steps = 252   # Time steps (daily frequency)

# Derived parameters
dt = T / n_t                    # Time step for PDE
ds_factor = 3.0                 # Space domain factor (0 to ds_factor*S0)
S_max = ds_factor * S0          # Maximum stock price
dS = S_max / n_s               # Space step size

# Numerical stability parameters
cfl_condition = 0.5             # CFL condition for explicit schemes
max_iterations = n_t * 2        # Maximum iterations for solvers

# Validation and bounds checking
def validate_parameters():
    """Validate all parameters for mathematical consistency"""
    errors = []
    
    if not (0 < alpha < 1):
        errors.append(f"Fractional order α={alpha} must be in (0,1)")
    
    if nu <= 0:
        errors.append(f"VG parameter ν={nu} must be positive")
        
    if sigma_vg <= 0:
        errors.append(f"VG parameter σ_vg={sigma_vg} must be positive")
        
    if abs(theta) > 1.0:
        errors.append(f"VG parameter θ={theta} should be in [-1,1] for stability")
        
    if n_t <= 0 or n_s <= 0:
        errors.append("Grid sizes must be positive")
        
    # Check CFL-like condition for fractional PDE
    dt_stability = T / n_t
    dx_stability = S_max / n_s
    stability_ratio = dt_stability / (dx_stability**2)
    
    if stability_ratio > 1.0:
        errors.append(f"Potential stability issue: dt/dx² = {stability_ratio:.3f} > 1")
    
    # Check VG parameter consistency (moment existence)
    # Second moment exists if ν*theta² + ν*sigma_vg² < ∞ (always true)
    # Fourth moment exists if ν < 1/2 for practical purposes
    if nu > 0.8:
        errors.append(f"Large ν={nu} may cause slow convergence")
    
    return errors

# Parameter sets for different test scenarios
PARAMETER_SETS = {
    'conservative': {
        'alpha': 0.9, 'theta': -0.05, 'sigma_vg': 0.15, 'nu': 0.2
    },
    'moderate': {
        'alpha': 0.85, 'theta': -0.12, 'sigma_vg': 0.18, 'nu': 0.25  
    },
    'aggressive': {
        'alpha': 0.75, 'theta': -0.25, 'sigma_vg': 0.25, 'nu': 0.4
    }
}

def load_parameter_set(set_name='moderate'):
    """Load a predefined parameter set"""
    global alpha, theta, sigma_vg, nu
    
    if set_name in PARAMETER_SETS:
        params = PARAMETER_SETS[set_name]
        alpha = params['alpha']
        theta = params['theta'] 
        sigma_vg = params['sigma_vg']
        nu = params['nu']
        print(f"Loaded '{set_name}' parameter set")
    else:
        print(f"Unknown parameter set '{set_name}', using defaults")

def print_config():
    """Print current configuration with validation"""
    print("="*60)
    print("🔧 FRACTIONAL LÉVY PIDE CONFIGURATION")
    print("="*60)
    
    print("📈 Market Parameters:")
    print(f"   S₀ (Initial Price)     = ${S0}")
    print(f"   K (Strike Price)       = ${K}")
    print(f"   T (Time to Maturity)   = {T} years")
    print(f"   r (Risk-free Rate)     = {r:.1%}")
    print(f"   σ (BS Volatility)      = {sigma:.1%}")
    
    print("\n🧮 Fractional Parameters:")
    print(f"   α (Fractional Order)   = {alpha}")
    print(f"   Time Steps (nₜ)        = {n_t}")
    print(f"   Space Steps (nₛ)       = {n_s}")
    print(f"   Domain: S ∈ [0, {S_max:.0f}]")
    
    print("\n📊 Variance Gamma Parameters:")
    print(f"   θ (Drift Parameter)    = {theta:.4f}")
    print(f"   σ_vg (VG Volatility)   = {sigma_vg:.4f}")  
    print(f"   ν (Variance Rate)      = {nu:.4f}")
    
    print("\n🎲 Simulation Parameters:")
    print(f"   Monte Carlo Paths      = {n_paths:,}")
    print(f"   Time Steps per Path    = {n_steps}")
    
    print("\n✅ Validation:")
    errors = validate_parameters()
    if errors:
        print("   ❌ Parameter Issues:")
        for error in errors:
            print(f"      • {error}")
    else:
        print("   ✅ All parameters valid")
    
    print("="*60)

# Data source configuration for real market data
DATA_SOURCES = {
    'yahoo_finance': {
        'module': 'yfinance',
        'default_ticker': 'SPY',
        'description': 'Yahoo Finance (free, good coverage)'
    },
    'alpha_vantage': {
        'module': 'alpha_vantage',
        'default_ticker': 'SPY', 
        'api_required': True,
        'description': 'Alpha Vantage (requires API key, high quality)'
    },
    'quandl': {
        'module': 'quandl',
        'default_ticker': 'WIKI/AAPL',
        'api_required': True,
        'description': 'Quandl (requires API key, historical focus)'
    }
}

def print_data_sources():
    """Print available data sources for real market data"""
    print("\n📊 Available Data Sources for Real Market Data:")
    print("-" * 55)
    
    for name, info in DATA_SOURCES.items():
        status = "🔑 API Key Required" if info.get('api_required') else "🆓 Free"
        print(f"• {name.upper()}:")
        print(f"    {info['description']}")  
        print(f"    Default ticker: {info['default_ticker']}")
        print(f"    Status: {status}")
        print()
    
    print("📋 Setup Instructions:")
    print("1. Yahoo Finance: pip install yfinance")
    print("2. Alpha Vantage: pip install alpha-vantage, get API key")
    print("3. Quandl: pip install quandl, get API key")

# Performance monitoring
PERFORMANCE_TARGETS = {
    'pide_solve_time': 30.0,    # seconds
    'mc_simulation_time': 10.0,  # seconds  
    'memory_usage_mb': 500,      # MB
    'convergence_tolerance': 1e-4
}

# Initialize with validation
if __name__ == "__main__":
    print_config()
    print_data_sources()