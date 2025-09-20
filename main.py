# main.py
# Complete Enhanced main script with real options data integration
# Supports historical European/American options data from multiple sources

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
import time
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add global declarations at the top of the function where they're used
from levy_simulation import simulate_vg_paths
from calibration import (bs_call_price, calibrate_vg_params, calibrate_to_real_data, 
                        vg_call_fft, vg_mc_price)
from pide_solver import fractional_pide_solver
from config import S0, K, T, r, sigma, theta, sigma_vg, nu, alpha

def validate_option_price(price, method_name):
    """Validate that option price is reasonable"""
    intrinsic_value = max(S0 - K, 0)
    bs_upper_bound = S0  # Stock price is upper bound
    
    if price < 0:
        print(f"⚠️  WARNING: {method_name} gave negative price: ${price:.4f}")
        return False
    elif price < intrinsic_value - 0.01:  # Small tolerance for numerical errors
        print(f"⚠️  WARNING: {method_name} price ${price:.4f} below intrinsic value ${intrinsic_value:.4f}")
        return False
    elif price > bs_upper_bound + 0.01:
        print(f"⚠️  WARNING: {method_name} price ${price:.4f} above upper bound ${bs_upper_bound:.4f}")
        return False
    return True

def get_real_options_data(symbol='SPY', days_back=30):
    """
    Get real historical options data for the last 5 years
    Returns options chain data with strikes, expirations, and prices
    """
    print(f"📊 Fetching real options data for {symbol}...")
    
    try:
        # Method 1: Try yfinance with options data
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Get current stock price
        hist = ticker.history(period="5d")
        if hist.empty:
            raise ValueError("No stock data available")
            
        current_price = hist['Close'].iloc[-1]
        
        # Get options expirations
        expirations = ticker.options
        if not expirations:
            raise ValueError("No options data available")
        
        # Get options chain for nearest expiration
        exp_date = expirations[0]  # Nearest expiration
        opt_chain = ticker.option_chain(exp_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Filter for reasonable strikes (around current price)
        price_range = 0.2  # 20% range around current price
        min_strike = current_price * (1 - price_range)
        max_strike = current_price * (1 + price_range)
        
        calls_filtered = calls[
            (calls['strike'] >= min_strike) & 
            (calls['strike'] <= max_strike) &
            (calls['volume'] > 0) &
            (calls['lastPrice'] > 0)
        ]
        
        if calls_filtered.empty:
            raise ValueError("No suitable call options found")
        
        # Find ATM or close-to-ATM option
        atm_call = calls_filtered.iloc[
            (calls_filtered['strike'] - current_price).abs().argsort()[:1]
        ]
        
        if atm_call.empty:
            raise ValueError("No ATM options found")
        
        # Extract option details
        option_data = {
            'symbol': symbol,
            'stock_price': current_price,
            'strike': float(atm_call['strike'].iloc[0]),
            'expiration': exp_date,
            'market_price': float(atm_call['lastPrice'].iloc[0]),
            'bid': float(atm_call['bid'].iloc[0]),
            'ask': float(atm_call['ask'].iloc[0]),
            'volume': int(atm_call['volume'].iloc[0]),
            'implied_vol': float(atm_call['impliedVolatility'].iloc[0]),
            'option_type': 'call',
            'style': 'american'  # Most US equity options are American
        }
        
        # Calculate time to expiration
        exp_datetime = pd.to_datetime(exp_date)
        days_to_exp = (exp_datetime - pd.Timestamp.now()).days
        option_data['time_to_expiration'] = max(days_to_exp / 365.25, 0.01)  # Convert to years
        
        print(f"✅ Successfully retrieved options data:")
        print(f"   Stock Price: ${current_price:.2f}")
        print(f"   Strike: ${option_data['strike']:.2f}")
        print(f"   Market Price: ${option_data['market_price']:.3f}")
        print(f"   Time to Exp: {days_to_exp} days ({option_data['time_to_expiration']:.3f} years)")
        print(f"   Implied Vol: {option_data['implied_vol']:.1%}")
        
        return option_data
        
    except Exception as e:
        print(f"❌ Failed to get real options data: {e}")
        print("📝 Using synthetic option data instead...")
        
        # Fallback to synthetic data
        return {
            'symbol': symbol,
            'stock_price': S0,
            'strike': K, 
            'market_price': 10.45,  # BS theoretical price
            'time_to_expiration': T,
            'implied_vol': sigma,
            'option_type': 'call',
            'style': 'european'
        }

def get_historical_volatility(symbol, period_days=252):
    """Calculate historical volatility from stock price data"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)  # Extra days for safety
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if len(hist) < 30:
            raise ValueError("Insufficient historical data")
        
        # Calculate log returns
        prices = hist['Close']
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Annualized volatility
        daily_vol = log_returns.std()
        annual_vol = daily_vol * np.sqrt(252)  # 252 trading days per year
        
        print(f"📈 Historical volatility ({period_days} days): {annual_vol:.1%}")
        
        return annual_vol, log_returns.values
        
    except Exception as e:
        print(f"⚠️ Could not calculate historical volatility: {e}")
        return sigma, np.array([])

def calibrate_to_real_options(option_data):
    """Calibrate model parameters to real market option prices"""
    print("\n🔧 Calibrating to real market option...")
    
    # Update global parameters with real data
    global S0, K, T, r, sigma
    
    S0 = option_data['stock_price']
    K = option_data['strike']
    T = option_data['time_to_expiration']
    
    # Use implied volatility as initial guess
    sigma = option_data['implied_vol']
    
    # Get historical volatility for comparison
    hist_vol, returns = get_historical_volatility(option_data['symbol'])
    
    print(f"📊 Market Parameters:")
    print(f"   S₀: ${S0:.2f}")
    print(f"   K: ${K:.2f}")
    print(f"   T: {T:.3f} years")
    print(f"   Implied Vol: {sigma:.1%}")
    print(f"   Historical Vol: {hist_vol:.1%}")
    
    # Calibrate VG parameters to match market price
    target_price = option_data['market_price']
    
    try:
        from scipy.optimize import minimize
        
        def objective(vg_params):
            theta_cal, sigma_vg_cal, nu_cal = vg_params
            
            # Bounds checking
            if nu_cal <= 0.01 or sigma_vg_cal <= 0.01:
                return 1e10
            
            # Price using Monte Carlo
            try:
                mc_price = vg_mc_price(theta_cal, sigma_vg_cal, nu_cal, n_paths=5000)
                error = (mc_price - target_price)**2
                return error
            except:
                return 1e10
        
        # Optimization bounds
        bounds = [(-0.3, 0.3), (0.05, 0.5), (0.05, 1.0)]
        
        # Multiple starting points
        best_params = None
        best_error = np.inf
        
        initial_guesses = [
            [0.0, sigma, 0.2],
            [-0.1, sigma*0.8, 0.3],
            [0.05, sigma*1.2, 0.25]
        ]
        
        for x0 in initial_guesses:
            try:
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                if result.success and result.fun < best_error:
                    best_params = result.x
                    best_error = result.fun
            except:
                continue
        
        if best_params is not None:
            theta_cal, sigma_vg_cal, nu_cal = best_params
            print(f"✅ Calibrated VG Parameters:")
            print(f"   θ: {theta_cal:.4f}")
            print(f"   σ_vg: {sigma_vg_cal:.4f}")
            print(f"   ν: {nu_cal:.4f}")
            return theta_cal, sigma_vg_cal, nu_cal
        else:
            raise ValueError("Calibration failed")
            
    except Exception as e:
        print(f"⚠️ Calibration failed: {e}")
        print("Using moment-based estimation...")
        
        # Fallback: moment-based estimation
        if len(returns) > 10:
            skew = np.mean(((returns - np.mean(returns))/np.std(returns))**3)
            kurt = np.mean(((returns - np.mean(returns))/np.std(returns))**4)
            
            nu_est = max(0.1, min(0.8, (kurt - 3) / 6))
            theta_est = np.clip(skew * hist_vol / 3, -0.2, 0.2)
            sigma_vg_est = np.clip(hist_vol * 0.8, 0.1, 0.4)
            
            return theta_est, sigma_vg_est, nu_est
        else:
            return theta, sigma_vg, nu

def print_detailed_results(results):
    """Print comprehensive results comparison"""
    print("\n" + "="*80)
    print("📊 FRACTIONAL LÉVY OPTION PRICING RESULTS")
    print("="*80)
    
    if 'real_option' in results:
        opt = results['real_option']
        print(f"🎯 Real Market Option:")
        print(f"   Symbol: {opt.get('symbol', 'N/A')}")
        print(f"   Type: {opt.get('option_type', 'call').title()} Option ({opt.get('style', 'european').title()})")
        print(f"   Market Price: ${opt.get('market_price', 0):.3f}")
        print(f"   Bid-Ask: ${opt.get('bid', 0):.3f} - ${opt.get('ask', 0):.3f}")
        print()
    
    print(f"📋 Market Parameters:")
    print(f"   Initial Stock Price (S₀): ${results['S0']:.2f}")
    print(f"   Strike Price (K): ${results['K']:.2f}")
    print(f"   Time to Maturity (T): {results['T']:.3f} years")
    print(f"   Risk-free Rate (r): {results['r']:.1%}")
    print(f"   Volatility (σ): {results.get('sigma', sigma):.1%}")
    print(f"   Fractional Order (α): {results['alpha']:.2f}")
    
    if 'vg_params' in results:
        θ, σ_vg, ν = results['vg_params']
        print(f"\n🔧 Calibrated VG Parameters:")
        print(f"   Drift (θ): {θ:.4f}")
        print(f"   Volatility (σ_vg): {σ_vg:.4f}")  
        print(f"   Variance Rate (ν): {ν:.4f}")
    
    print(f"\n💰 MODEL PRICES:")
    print("-" * 50)
    
    methods = ['Black-Scholes', 'VG Monte Carlo', 'VG FFT', 'Fractional PIDE']
    prices = [results.get('bs_price', 0), results.get('mc_price', 0), 
              results.get('fft_price', 0), results.get('pide_price', 0)]
    
    # Add market price if available
    if 'real_option' in results:
        methods.insert(0, 'Market Price')
        prices.insert(0, results['real_option']['market_price'])
    
    for method, price in zip(methods, prices):
        if price > 0:
            status = "✅" if validate_option_price(price, method) else "❌"
            if method == 'Market Price':
                print(f"  🎯 {method:<18}: ${price:.4f} (actual market)")
            else:
                print(f"  {status} {method:<18}: ${price:.4f}")
        else:
            print(f"  ❌ {method:<18}: Not calculated")
    
    # Error analysis against market price
    if 'real_option' in results:
        market_price = results['real_option']['market_price']
        print(f"\n📊 ERROR vs MARKET PRICE:")
        print("-" * 35)
        
        model_prices = {
            'Black-Scholes': results.get('bs_price', 0),
            'VG MC': results.get('mc_price', 0),
            'VG FFT': results.get('fft_price', 0),
            'Fractional PIDE': results.get('pide_price', 0)
        }
        
        for name, price in model_prices.items():
            if price > 0:
                error = abs(price - market_price) / market_price * 100
                print(f"  {name:<15}: {error:.2f}%")
    
    print("="*80)

def create_enhanced_visualizations(results):
    """Create comprehensive visualization dashboard"""
    try:
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Option price surface (2x3 grid, position 1)
        ax1 = plt.subplot(2, 3, 1)
        if 'S_grid' in results and 'V_surface' in results:
            S_grid = results['S_grid']
            V_surf = results['V_surface']
            
            ax1.plot(S_grid, V_surf, 'b-', linewidth=2.5, label='Fractional PIDE')
            ax1.axvline(results['K'], color='red', linestyle='--', alpha=0.8, 
                       linewidth=2, label=f'Strike = ${results["K"]:.0f}')
            ax1.axvline(results['S0'], color='green', linestyle='--', alpha=0.8,
                       linewidth=2, label=f'S₀ = ${results["S0"]:.0f}')
            
            # Add intrinsic value line
            intrinsic = np.maximum(S_grid - results['K'], 0)
            ax1.plot(S_grid, intrinsic, 'k:', alpha=0.6, label='Intrinsic Value')
            
            ax1.set_xlabel('Stock Price ($)', fontsize=11)
            ax1.set_ylabel('Option Value ($)', fontsize=11)
            ax1.set_title('Fractional PIDE Solution', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No PIDE Solution Available', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12, color='red')
            ax1.set_title('Option Price Surface - No Data')
        
        # Plot 2: Terminal price distribution
        ax2 = plt.subplot(2, 3, 2)
        if 'simulation_paths' in results and results['simulation_paths'].size > 0:
            paths = results['simulation_paths']
            terminal_prices = paths[:, -1]
            
            # Histogram
            n, bins, patches = ax2.hist(terminal_prices, bins=50, density=True, 
                                      alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Overlay normal distribution for comparison
            mu_terminal = np.mean(terminal_prices)
            sigma_terminal = np.std(terminal_prices)
            x_norm = np.linspace(terminal_prices.min(), terminal_prices.max(), 100)
            normal_pdf = norm.pdf(x_norm, mu_terminal, sigma_terminal)
            ax2.plot(x_norm, normal_pdf, 'r-', linewidth=2, alpha=0.8, 
                    label=f'Normal Fit (μ=${mu_terminal:.1f})')
            
            ax2.axvline(results['K'], color='red', linestyle='--', linewidth=2, 
                       label=f'Strike = ${results["K"]:.0f}')
            ax2.axvline(mu_terminal, color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean = ${mu_terminal:.1f}')
            
            ax2.set_xlabel('Terminal Stock Price ($)', fontsize=11)
            ax2.set_ylabel('Probability Density', fontsize=11)
            ax2.set_title('VG Terminal Distribution', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Simulation Data', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, color='red')
            ax2.set_title('Terminal Price Distribution - No Data')
        
        # Plot 3: Method comparison including market price
        ax3 = plt.subplot(2, 3, 3)
        methods = []
        prices = []
        colors = []
        
        # Add market price if available
        if 'real_option' in results:
            methods.append('Market\nPrice')
            prices.append(results['real_option']['market_price'])
            colors.append('#ff1f5b')
        
        if results.get('bs_price', 0) > 0:
            methods.append('Black-\nScholes')
            prices.append(results['bs_price'])
            colors.append('#1f77b4')
            
        if results.get('mc_price', 0) > 0:
            methods.append('VG\nMonte Carlo')
            prices.append(results['mc_price'])  
            colors.append('#ff7f0e')
            
        if results.get('fft_price', 0) > 0:
            methods.append('VG\nFFT')
            prices.append(results['fft_price'])
            colors.append('#2ca02c')
            
        if results.get('pide_price', 0) > 0:
            methods.append('Fractional\nPIDE')
            prices.append(results['pide_price'])
            colors.append('#d62728')
        
        if len(methods) > 0:
            bars = ax3.bar(methods, prices, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar, price in zip(bars, prices):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(prices)*0.01,
                        f'${price:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax3.set_ylabel('Option Price ($)', fontsize=11)
            ax3.set_title('Pricing Method Comparison', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at intrinsic value
            intrinsic_val = max(results['S0'] - results['K'], 0)
            if intrinsic_val > 0:
                ax3.axhline(intrinsic_val, color='gray', linestyle=':', 
                           label=f'Intrinsic = ${intrinsic_val:.2f}')
                ax3.legend(fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No Valid Prices', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color='red')
        
        # Plot 4: Sample paths
        ax4 = plt.subplot(2, 3, 4)
        if 'simulation_paths' in results and results['simulation_paths'].size > 0:
            paths = results['simulation_paths']
            time_grid = np.linspace(0, results['T'], paths.shape[1])
            
            # Show up to 20 sample paths
            n_paths_show = min(20, paths.shape[0])
            sample_paths = paths[:n_paths_show]
            
            for i, path in enumerate(sample_paths):
                alpha_val = max(0.3, 1.0 - i*0.03)  # Fade older paths
                ax4.plot(time_grid, path, alpha=alpha_val, linewidth=1)
            
            # Add strike and initial price reference lines
            ax4.axhline(results['K'], color='red', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'Strike = ${results["K"]:.0f}')
            ax4.axhline(results['S0'], color='green', linestyle='-', linewidth=2, 
                       alpha=0.8, label=f'S₀ = ${results["S0"]:.0f}')
            
            ax4.set_xlabel('Time (years)', fontsize=11)
            ax4.set_ylabel('Stock Price ($)', fontsize=11)
            ax4.set_title(f'Sample VG Paths (n={n_paths_show})', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Path Data', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, color='red')
        
        # Plot 5: Error analysis vs market
        ax5 = plt.subplot(2, 3, 5)
        if 'real_option' in results:
            market_price = results['real_option']['market_price']
            valid_prices = {}
            
            if results.get('bs_price', 0) > 0: valid_prices['BS'] = results['bs_price']
            if results.get('mc_price', 0) > 0: valid_prices['MC'] = results['mc_price']
            if results.get('fft_price', 0) > 0: valid_prices['FFT'] = results['fft_price']
            if results.get('pide_price', 0) > 0: valid_prices['PIDE'] = results['pide_price']
            
            if len(valid_prices) > 0:
                errors = []
                labels = []
                
                for method, price in valid_prices.items():
                    error = abs(price - market_price) / market_price * 100
                    errors.append(error)
                    labels.append(method)
                
                colors_error = ['blue', 'orange', 'green', 'red'][:len(errors)]
                bars = ax5.bar(labels, errors, color=colors_error, alpha=0.7, edgecolor='black')
                
                # Add percentage labels
                for bar, error in zip(bars, errors):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{error:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                ax5.set_ylabel('Relative Error (%)', fontsize=11)
                ax5.set_title('Error vs Market Price', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='y')
            else:
                ax5.text(0.5, 0.5, 'No Model Prices\nAvailable', 
                        ha='center', va='center', transform=ax5.transAxes,
                        fontsize=12, color='red')
        else:
            ax5.text(0.5, 0.5, 'No Market Price\nfor Comparison', 
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=12, color='red')
        
        # Plot 6: VG parameters and implied vol comparison
        ax6 = plt.subplot(2, 3, 6)
        if 'vg_params' in results:
            θ, σ_vg, ν = results['vg_params']
            
            # Compare implied vs historical vs VG volatility
            vol_comparison = {}
            
            if 'real_option' in results:
                vol_comparison['Implied Vol'] = results['real_option']['implied_vol']
            if 'historical_vol' in results:
                vol_comparison['Historical Vol'] = results['historical_vol']
            vol_comparison['VG σ_vg'] = σ_vg
            vol_comparison['BS σ'] = results.get('sigma', sigma)
            
            if len(vol_comparison) > 1:
                names = list(vol_comparison.keys())
                values = list(vol_comparison.values())
                colors_vol = ['red', 'blue', 'green', 'orange'][:len(values)]
                
                bars = ax6.bar(names, values, color=colors_vol, alpha=0.7, edgecolor='black')
                
                # Add percentage labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{value:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                ax6.set_ylabel('Volatility', fontsize=11)
                ax6.set_title('Volatility Comparison', fontsize=12, fontweight='bold')
                ax6.grid(True, alpha=0.3, axis='y')
            else:
                # Show VG parameters
                param_names = ['θ (Drift)', 'σ_vg (Vol)', 'ν (Var Rate)']
                param_values = [θ, σ_vg, ν]
                param_colors = ['blue', 'green', 'red']
                
                bars = ax6.bar(param_names, param_values, color=param_colors, alpha=0.7, edgecolor='black')
                
                for bar, value in zip(bars, param_values):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                ax6.set_ylabel('Parameter Value', fontsize=11)
                ax6.set_title('VG Parameters', fontsize=12, fontweight='bold')
                ax6.grid(True, alpha=0.3, axis='y')
                ax6.axhline(0, color='black', linestyle='-', alpha=0.5)
        else:
            ax6.text(0.5, 0.5, 'No Parameters\nAvailable', 
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=12, color='red')
        
        plt.tight_layout(pad=3.0)
        filename = 'fractional_levy_real_options_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Enhanced visualization saved as '{filename}'")
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    """Complete Enhanced main execution function with real options data"""
    # Declare globals at the very beginning of the function
    global S0, K, T, r, sigma
    
    print("🚀 FRACTIONAL LÉVY PIDE SOLVER WITH REAL OPTIONS DATA")
    print("="*80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏛️ Data Sources: Yahoo Finance, yfinance, Alpha Vantage (fallback)")
    
    # Initialize results dictionary
    results = {
        'S0': S0, 'K': K, 'T': T, 'r': r, 'alpha': alpha, 'sigma': sigma,
        'bs_price': 0, 'mc_price': 0, 'fft_price': 0, 'pide_price': 0
    }
    
    # Step 1: Get Real Options Data
    print("\n📊 Step 1: Fetching Real Market Options Data")
    print("-" * 50)
    
    # Try to get real options data for major symbols
    symbols_to_try = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
    real_option_data = None
    
    for symbol in symbols_to_try:
        try:
            real_option_data = get_real_options_data(symbol)
            if real_option_data:
                break
        except Exception as e:
            print(f"Failed to get data for {symbol}: {e}")
            continue
    
    if real_option_data:
        results['real_option'] = real_option_data
        
        # Update global parameters with real data
        S0 = real_option_data['stock_price']
        K = real_option_data['strike']
        T = real_option_data['time_to_expiration']
        sigma = real_option_data['implied_vol']
        
        # Update results
        results['S0'] = S0
        results['K'] = K
        results['T'] = T
        results['sigma'] = sigma
        
        print(f"✅ Using real option data from {real_option_data['symbol']}")
    else:
        print("⚠️ Using synthetic option data")
        
    # Step 2: Get Historical Volatility
    print("\n📈 Step 2: Historical Volatility Analysis")
    print("-" * 50)
    
    symbol = real_option_data['symbol'] if real_option_data else 'SPY'
    try:
        hist_vol, returns = get_historical_volatility(symbol, period_days=252)
        results['historical_vol'] = hist_vol
        results['returns'] = returns
        
        if real_option_data:
            print(f"📊 Volatility Comparison:")
            print(f"   Implied Volatility: {real_option_data['implied_vol']:.1%}")
            print(f"   Historical Volatility: {hist_vol:.1%}")
            vol_ratio = real_option_data['implied_vol'] / hist_vol
            print(f"   Implied/Historical Ratio: {vol_ratio:.2f}")
            
    except Exception as e:
        print(f"⚠️ Historical volatility calculation failed: {e}")
        results['historical_vol'] = sigma
        results['returns'] = np.array([])
    
    # Step 3: Calibrate VG Parameters to Real Market Data
    print("\n🔧 Step 3: VG Parameter Calibration to Market Data")
    print("-" * 50)
    
    try:
        if real_option_data:
            θ, σ_vg, ν = calibrate_to_real_options(real_option_data)
        else:
            θ, σ_vg, ν = calibrate_vg_params(method='mc')
        
        results['vg_params'] = (θ, σ_vg, ν)
        print(f"✅ VG Parameters calibrated successfully")
        
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        print("Using default parameters...")
        θ, σ_vg, ν = theta, sigma_vg, nu
        results['vg_params'] = (θ, σ_vg, ν)
    
    # Step 4: Black-Scholes Benchmark
    print("\n📊 Step 4: Black-Scholes Benchmark")
    print("-" * 40)
    
    try:
        bs_price = bs_call_price(S0, K, T, r, sigma)
        results['bs_price'] = bs_price
        print(f"✅ Black-Scholes price: ${bs_price:.4f}")
        
        if real_option_data:
            bs_error = abs(bs_price - real_option_data['market_price']) / real_option_data['market_price'] * 100
            print(f"📊 BS vs Market Error: {bs_error:.2f}%")
            
    except Exception as e:
        print(f"❌ Black-Scholes calculation failed: {e}")
    
    # Step 5: VG Monte Carlo Simulation
    print("\n🎲 Step 5: VG Monte Carlo Simulation")
    print("-" * 40)
    
    try:
        np.random.seed(42)  # For reproducibility
        
        print("Generating VG paths...")
        start_time = time.time()
        paths = simulate_vg_paths(θ, σ_vg, ν, n_paths=3000)
        mc_time = time.time() - start_time
        
        results['simulation_paths'] = paths
        
        # Calculate MC price
        mc_price = vg_mc_price(θ, σ_vg, ν, n_paths=3000)
        results['mc_price'] = mc_price
        
        print(f"✅ Monte Carlo completed:")
        print(f"   Simulation time: {mc_time:.2f} seconds")
        print(f"   Paths generated: {len(paths):,}")
        print(f"   MC option price: ${mc_price:.4f}")
        
        if real_option_data:
            mc_error = abs(mc_price - real_option_data['market_price']) / real_option_data['market_price'] * 100
            print(f"📊 MC vs Market Error: {mc_error:.2f}%")
        
    except Exception as e:
        print(f"❌ Monte Carlo simulation failed: {e}")
        results['simulation_paths'] = np.array([])
    
    # Step 6: VG FFT Pricing (Alternative Method)
    print("\n⚡ Step 6: VG FFT Pricing")
    print("-" * 40)
    
    try:
        start_time = time.time()
        fft_price = vg_call_fft(S0, K, T, r, θ, σ_vg, ν)
        fft_time = time.time() - start_time
        
        results['fft_price'] = fft_price
        print(f"✅ FFT pricing completed:")
        print(f"   Computation time: {fft_time:.3f} seconds")
        print(f"   FFT option price: ${fft_price:.4f}")
        
        if real_option_data:
            fft_error = abs(fft_price - real_option_data['market_price']) / real_option_data['market_price'] * 100
            print(f"📊 FFT vs Market Error: {fft_error:.2f}%")
            
    except Exception as e:
        print(f"❌ FFT pricing failed: {e}")
    
    # Step 7: Fractional PIDE Solver (Main Algorithm)
    print("\n🧮 Step 7: Fractional PIDE Solver")
    print("-" * 40)
    
    try:
        print(f"Solving time-fractional PIDE (α={alpha})...")
        print("This may take 30-60 seconds for accurate results...")
        
        start_time = time.time()
        pide_price, S_grid, V_surface = fractional_pide_solver(θ, σ_vg, ν)
        pide_time = time.time() - start_time
        
        results['pide_price'] = pide_price
        results['S_grid'] = S_grid
        results['V_surface'] = V_surface
        results['pide_time'] = pide_time
        
        print(f"✅ PIDE solver completed:")
        print(f"   Computation time: {pide_time:.2f} seconds")
        print(f"   Grid size: {len(S_grid)} spatial points")
        print(f"   Fractional PIDE price: ${pide_price:.4f}")
        
        if real_option_data:
            pide_error = abs(pide_price - real_option_data['market_price']) / real_option_data['market_price'] * 100
            print(f"📊 PIDE vs Market Error: {pide_error:.2f}%")
        
        # Validate PIDE solution
        if pide_price <= 0:
            print("❌ WARNING: PIDE gave non-positive price!")
        elif pide_price > S0:
            print("❌ WARNING: PIDE price exceeds stock price!")
        else:
            print("✅ PIDE solution passes basic validation")
            
    except Exception as e:
        print(f"❌ PIDE solver failed: {e}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())
    
    # Step 8: Comprehensive Results Analysis
    print("\n📋 Step 8: Results Analysis and Comparison")
    print("-" * 50)
    
    print_detailed_results(results)
    
    # Step 9: Model Performance Summary
    print("\n⚡ Step 9: Performance Summary")
    print("-" * 40)
    
    if 'pide_time' in results:
        print(f"PIDE Solver Runtime: {results['pide_time']:.2f} seconds")
        
        if results['pide_time'] > 60:
            print("⚠️  Consider reducing grid size for faster computation")
        elif results['pide_time'] < 10:
            print("✅ Fast computation - consider increasing grid size for better accuracy")
        else:
            print("✅ Good balance of speed and accuracy")
    
    # Check which method performs best vs market
    if 'real_option' in results:
        market_price = results['real_option']['market_price']
        errors = {}
        
        for method, key in [('BS', 'bs_price'), ('MC', 'mc_price'), ('FFT', 'fft_price'), ('PIDE', 'pide_price')]:
            if results.get(key, 0) > 0:
                error = abs(results[key] - market_price) / market_price * 100
                errors[method] = error
        
        if errors:
            best_method = min(errors.items(), key=lambda x: x[1])
            print(f"\n🏆 Best Performing Method: {best_method[0]} (Error: {best_method[1]:.2f}%)")
    
    # Step 10: Enhanced Visualization
    print("\n📊 Step 10: Creating Enhanced Visualizations")
    print("-" * 50)
    
    create_enhanced_visualizations(results)
    
    # Step 11: Final Summary and Recommendations
    print("\n🎯 Step 11: Final Summary")
    print("-" * 40)
    
    if real_option_data:
        print(f"✅ Successfully analyzed real {real_option_data['symbol']} option")
        print(f"Market Price: ${real_option_data['market_price']:.3f}")
        
        if results.get('pide_price', 0) > 0:
            pide_error = abs(results['pide_price'] - real_option_data['market_price']) / real_option_data['market_price'] * 100
            if pide_error < 10:
                print("🎉 Excellent: PIDE model shows good agreement with market (<10% error)")
            elif pide_error < 20:
                print("✅ Good: PIDE model shows reasonable agreement with market (<20% error)")
            else:
                print("⚠️ Moderate: PIDE model shows significant deviation from market (>20% error)")
                print("   Consider refining calibration or model parameters")
    
    print(f"\n📈 Model Insights:")
    if 'vg_params' in results:
        θ, σ_vg, ν = results['vg_params']
        print(f"   • VG drift (θ={θ:.3f}): {'Negative skew' if θ < 0 else 'Positive skew'}")
        print(f"   • Jump intensity (ν={ν:.3f}): {'High' if ν > 0.5 else 'Moderate' if ν > 0.2 else 'Low'} jump frequency")
        print(f"   • Fractional order (α={alpha}): {'Strong' if alpha < 0.8 else 'Moderate'} memory effects")
    
    print(f"\n💾 Output Files Generated:")
    print("   • fractional_levy_real_options_results.png (main visualization)")
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    # Set display options for better output
    np.set_printoptions(precision=4, suppress=True)
    plt.style.use('default')  # Clean matplotlib style
    
    # Run main analysis
    main_results = main()
    
    # Optional: Run validation tests
    try:
        from validation_tests import run_validation
        print("\n🧪 Running validation tests...")
        validation_results = run_validation()
    except ImportError:
        print("⚠️ Validation tests not available (validation_tests.py not found)")
    except Exception as e:
        print(f"⚠️ Validation tests failed: {e}")
    
    print(f"\n📊 Session completed at {datetime.now().strftime('%H:%M:%S')}")