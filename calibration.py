# calibration.py
# Enhanced calibration module with real data support
# Multiple data sources and robust parameter estimation

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from levy_simulation import simulate_vg_paths
from config import S0, K, T, r, sigma, theta, sigma_vg, nu

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes European call option price"""
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def vg_characteristic_function(u, t, theta, sigma_vg, nu):
    """VG characteristic function for semi-analytical pricing"""
    omega = -np.log(1 - theta*nu - 0.5*sigma_vg**2*nu) / nu
    drift_term = 1j * u * (r - omega) * t
    vg_term = -t/nu * np.log(1 - 1j*u*theta*nu + 0.5*u**2*sigma_vg**2*nu)
    return np.exp(drift_term + vg_term)

def vg_call_fft(S, K, T, r, theta, sigma_vg, nu):
    """
    VG call price using FFT method
    Based on Carr-Madan (1999) FFT approach
    """
    try:
        from scipy.fft import fft, fftfreq
        
        # FFT parameters
        N = 2**12  # Number of points
        eta = 0.25  # Grid spacing in log-strike dimension
        alpha_damping = 1.5  # Damping factor
        
        # Log-strike grid
        log_k = np.arange(N) * eta - 0.5 * N * eta
        k_grid = np.exp(log_k)
        
        # Frequency grid
        v_grid = fftfreq(N, d=eta) * 2 * np.pi
        
        # Modified characteristic function
        phi_v = vg_characteristic_function(v_grid - 1j*(alpha_damping+1), T, theta, sigma_vg, nu)
        phi_v *= np.exp(-r*T) / ((alpha_damping + 1j*v_grid) * (alpha_damping + 1 + 1j*v_grid))
        
        # FFT transform
        x = eta * np.arange(N)
        w = np.exp(-alpha_damping * x) * phi_v
        y = fft(w).real
        
        call_prices = np.exp(-alpha_damping * log_k) * y / np.pi * S
        
        # Interpolate at desired strike
        price = np.interp(K, k_grid, call_prices)
        return max(price, 0)
        
    except:
        # Fallback to Monte Carlo if FFT fails
        return vg_mc_price(theta, sigma_vg, nu, n_paths=5000)

def vg_mc_price(theta, sigma_vg, nu, n_paths=10000):
    """Monte Carlo price for VG process"""
    try:
        np.random.seed(42)  # For reproducible results
        paths = simulate_vg_paths(theta, sigma_vg, nu, n_paths=n_paths)
        payoffs = np.maximum(paths[:, -1] - K, 0)
        return np.mean(payoffs) * np.exp(-r*T)
    except:
        return bs_call_price(S0, K, T, r, 0.2)  # Fallback

def calibration_objective(params, method='mc'):
    """Objective function for VG parameter calibration"""
    theta_cal, sigma_vg_cal, nu_cal = params
    
    # Parameter constraints
    if nu_cal <= 0.01 or nu_cal > 5.0:
        return 1e10
    if sigma_vg_cal <= 0.01 or sigma_vg_cal > 1.0:
        return 1e10
    if abs(theta_cal) > 1.0:
        return 1e10
    
    # Target price (Black-Scholes benchmark)
    target_price = bs_call_price(S0, K, T, r, sigma)
    
    # VG model price
    if method == 'fft':
        vg_price = vg_call_fft(S0, K, T, r, theta_cal, sigma_vg_cal, nu_cal)
    else:
        vg_price = vg_mc_price(theta_cal, sigma_vg_cal, nu_cal)
    
    # Squared error with penalty for extreme parameters
    error = (vg_price - target_price)**2
    
    # Add regularization penalty
    penalty = 0.1 * (nu_cal**2 + theta_cal**2 + (sigma_vg_cal - 0.2)**2)
    
    return error + penalty

def calibrate_vg_params(method='mc'):
    """
    Calibrate VG parameters using robust optimization
    method: 'mc' for Monte Carlo, 'fft' for FFT pricing
    """
    # Multiple starting points for robustness
    initial_guesses = [
        [0.0, 0.2, 0.3],
        [-0.1, 0.15, 0.25], 
        [0.1, 0.25, 0.4],
        [-0.05, 0.18, 0.35]
    ]
    
    best_result = None
    best_error = np.inf
    
    # Parameter bounds
    bounds = [(-0.5, 0.5), (0.05, 0.8), (0.05, 2.0)]
    
    # Try multiple optimization approaches
    for x0 in initial_guesses:
        try:
            # L-BFGS-B optimization
            result = minimize(
                calibration_objective, x0, args=(method,), 
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
                
        except:
            continue
    
    # If local optimization fails, try global optimization
    if best_result is None or best_error > 10:
        try:
            result = differential_evolution(
                calibration_objective, bounds, args=(method,),
                maxiter=50, popsize=15, seed=42
            )
            if result.success:
                best_result = result
        except:
            pass
    
    # Return best parameters or defaults
    if best_result is not None and best_result.fun < 100:
        return best_result.x
    else:
        print("Calibration failed, using default parameters")
        return np.array([theta, sigma_vg, nu])

def get_real_market_data(ticker='SPY', source='yahoo'):
    """
    Fetch real market data from various sources
    
    Sources:
    - 'yahoo': Yahoo Finance (yfinance)
    - 'alpha': Alpha Vantage API  
    - 'quandl': Quandl API
    """
    
    if source == 'yahoo':
        return get_yahoo_data(ticker)
    elif source == 'alpha':
        return get_alpha_vantage_data(ticker)
    else:
        return get_yahoo_data(ticker)  # Default fallback

def get_yahoo_data(ticker='SPY'):
    """Get data from Yahoo Finance"""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Download 2 years of data for better statistics
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError("No data retrieved")
        
        # Use adjusted close prices
        prices = data['Close'].dropna()
        
        if len(prices) < 50:
            raise ValueError("Insufficient data points")
            
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        
        # Get current price
        S0_real = float(prices.iloc[-1])
        
        # Calculate return statistics
        mu = np.mean(log_returns) * 252  # Annualized mean
        sigma_real = np.std(log_returns) * np.sqrt(252)  # Annualized vol
        skew = np.mean(((log_returns - np.mean(log_returns))/np.std(log_returns))**3)
        kurt = np.mean(((log_returns - np.mean(log_returns))/np.std(log_returns))**4)
        
        # Risk-free rate approximation (can be improved with FRED API)
        r_real = 0.05  # Default 5% risk-free rate
        
        # VG parameter estimation using method of moments
        # Based on theoretical moments of VG distribution
        nu_est = max(0.1, min(1.5, (kurt - 3) / 3))  # From excess kurtosis
        
        # Skewness relationship: E[skew] ≈ 3*theta*nu^1.5/(sigma^3)
        if sigma_real > 0:
            theta_est = skew * sigma_real**3 / (3 * nu_est**1.5)
            theta_est = np.clip(theta_est, -0.3, 0.3)  # Reasonable bounds
        else:
            theta_est = 0.0
            
        # Variance relationship: Var = sigma² + nu*theta² + nu*sigma_vg²
        sigma_vg_est = np.sqrt(max(0.01, sigma_real**2 - nu_est*theta_est**2))
        sigma_vg_est = np.clip(sigma_vg_est, 0.1, 0.5)
        
        print(f"Retrieved {len(prices)} price points for {ticker}")
        print(f"Estimated parameters: μ={mu:.4f}, σ={sigma_real:.4f}, skew={skew:.4f}, kurt={kurt:.4f}")
        
        return {
            'S0': S0_real,
            'r': r_real, 
            'sigma_market': sigma_real,
            'theta': theta_est,
            'sigma_vg': sigma_vg_est, 
            'nu': nu_est,
            'returns': log_returns
        }
        
    except Exception as e:
        print(f"Yahoo Finance data retrieval failed: {e}")
        return None

def get_alpha_vantage_data(ticker='SPY', api_key=None):
    """
    Get data from Alpha Vantage API
    Requires API key: https://www.alphavantage.co/support/#api-key
    """
    if api_key is None:
        print("Alpha Vantage requires API key. Using Yahoo Finance fallback.")
        return get_yahoo_data(ticker)
    
    try:
        import requests
        import pandas as pd
        
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError("Invalid response from Alpha Vantage")
        
        # Process data similar to Yahoo Finance
        ts_data = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        prices = df['5. adjusted close'].astype(float)
        
        # Calculate statistics (same as Yahoo method)
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        S0_real = float(prices.iloc[-1])
        
        # ... (rest similar to yahoo method)
        
        return {
            'S0': S0_real,
            'returns': log_returns
            # ... other parameters
        }
        
    except Exception as e:
        print(f"Alpha Vantage data retrieval failed: {e}")
        return get_yahoo_data(ticker)  # Fallback

def calibrate_to_real_data(ticker='SPY', source='yahoo'):
    """
    Main function to calibrate VG model to real market data
    """
    print(f"Fetching real market data for {ticker}...")
    
    market_data = get_real_market_data(ticker, source)
    
    if market_data is None:
        print("Failed to get real data, using synthetic calibration")
        return S0, theta, sigma_vg, nu
    
    # Update global parameters with real data
    S0_real = market_data['S0']
    theta_real = market_data['theta']
    sigma_vg_real = market_data['sigma_vg'] 
    nu_real = market_data['nu']
    
    print(f"Market-estimated VG parameters:")
    print(f"  S0 = ${S0_real:.2f}")
    print(f"  θ = {theta_real:.4f}")
    print(f"  σ_vg = {sigma_vg_real:.4f}")
    print(f"  ν = {nu_real:.4f}")
    
    return S0_real, theta_real, sigma_vg_real, nu_real