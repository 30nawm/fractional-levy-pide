# main.py
# Complete production-grade fractional Lévy options pricing system

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
import json
from options_data_fetcher import get_option_data, get_historical_volatility
from portfolio_constructor import PORTFOLIO

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import modules
from options_data_fetcher import get_option_data
from calibration import bs_call_price, vg_mc_price, vg_call_fft, calibrate_vg_to_single_option
from pide_solver import fractional_pide_solver
from config import update_config_with_option_data, r
from levy_simulation import validate_vg_parameters


class AnalysisEngine:
    
    def __init__(self):
        self.results = {}
        self.summary_stats = {}
        self.start_time = time.time()
        
    def print_header(self):
        print("\n" + "="*80)
        print("🚀 FRACTIONAL LÉVY PIDE OPTIONS PRICING SYSTEM")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Portfolio Size: {sum(len(v) for v in PORTFOLIO.values())} symbols")
        print(f"🎯 Methods: Black-Scholes | VG Monte Carlo | VG FFT | Fractional PIDE")
        print("="*80 + "\n")
        
    def fetch_and_filter(self, symbol):
        """Step 1: Data acquisition with quality filtering"""
        print(f"\n{'─'*80}")
        print(f"📈 {symbol}")
        print(f"{'─'*80}")
        
        option = get_option_data(symbol)
        
        if not option:
            logger.warning(f"❌ {symbol}: No suitable options found")
            return None
            
        print(f"✅ Option Data Retrieved:")
        print(f"   Stock: ${option['stock_price']:.2f} | Strike: ${option['strike']:.2f} | "
              f"Moneyness: {option['strike']/option['stock_price']:.3f}")
        print(f"   Market Price: ${option['market_price']:.3f} | "
              f"Bid-Ask: ${option['bid']:.3f}-${option['ask']:.3f}")
        print(f"   Volume: {option['volume']:,} | Open Interest: {option['open_interest']:,}")
        print(f"   Days to Expiry: {option['days_to_expiration']} | IV: {option['implied_vol']:.1%}")
        
        return option
    
    def calibrate_parameters(self, option):
        """Step 2: VG parameter calibration"""
        print(f"\n🔧 Calibrating VG Parameters...")
        
        update_config_with_option_data(option)
        
        try:
            theta, sigma_vg, nu = calibrate_vg_to_single_option(option)
            
            # Validate
            is_valid, msg = validate_vg_parameters(theta, sigma_vg, nu)
            
            if is_valid:
                print(f"   ✓ VG Calibrated: θ={theta:.4f}, σ_vg={sigma_vg:.4f}, ν={nu:.4f}")
            else:
                print(f"   ⚠ VG Warning: {msg}")
                print(f"   → Using: θ={theta:.4f}, σ_vg={sigma_vg:.4f}, ν={nu:.4f}")
            
            return (theta, sigma_vg, nu)
            
        except Exception as e:
            logger.error(f"   ❌ Calibration failed: {e}")
            return None
    
    def price_all_methods(self, option, vg_params):
        """Step 3: Price with all methods"""
        print(f"\n💰 Pricing with Multiple Methods...")
        
        S = option['stock_price']
        K = option['strike']
        T = option['time_to_expiration']
        iv = option['implied_vol']
        market = option['market_price']
        theta, sigma_vg, nu = vg_params
        
        results = {}
        
        # Black-Scholes
        print(f"   → Black-Scholes...", end=" ")
        try:
            t0 = time.time()
            bs = bs_call_price(S, K, T, r, iv)
            results['BS'] = {
                'price': bs,
                'error_pct': abs(bs - market) / market * 100,
                'time': time.time() - t0,
                'valid': 0 < bs < S
            }
            print(f"${bs:.4f} ({results['BS']['error_pct']:.1f}% error)")
        except Exception as e:
            print(f"Failed: {e}")
            results['BS'] = None
        
        # Monte Carlo
        print(f"   → VG Monte Carlo...", end=" ")
        try:
            t0 = time.time()
            mc = vg_mc_price(theta, sigma_vg, nu, S, K, T, r, 5000)
            results['MC'] = {
                'price': mc,
                'error_pct': abs(mc - market) / market * 100,
                'time': time.time() - t0,
                'valid': 0 < mc < S
            }
            print(f"${mc:.4f} ({results['MC']['error_pct']:.1f}% error, {results['MC']['time']:.2f}s)")
        except Exception as e:
            print(f"Failed: {e}")
            results['MC'] = None
        
        # FFT
        print(f"   → VG FFT...", end=" ")
        try:
            t0 = time.time()
            fft = vg_call_fft(S, K, T, r, theta, sigma_vg, nu)
            results['FFT'] = {
                'price': fft,
                'error_pct': abs(fft - market) / market * 100,
                'time': time.time() - t0,
                'valid': 0 < fft < S
            }
            print(f"${fft:.4f} ({results['FFT']['error_pct']:.1f}% error, {results['FFT']['time']:.3f}s)")
        except Exception as e:
            print(f"Failed: {e}")
            results['FFT'] = None
        
        # Fractional PIDE
        print(f"   → Fractional PIDE...", end=" ")
        try:
            t0 = time.time()
            pide, S_grid, V_surface = fractional_pide_solver(theta, sigma_vg, nu)
            
            if 0 < pide < S * 1.5:
                results['PIDE'] = {
                    'price': pide,
                    'error_pct': abs(pide - market) / market * 100,
                    'time': time.time() - t0,
                    'valid': True,
                    'grid': (S_grid, V_surface)
                }
                print(f"${pide:.4f} ({results['PIDE']['error_pct']:.1f}% error, {results['PIDE']['time']:.2f}s)")
            else:
                print(f"${pide:.4f} - Out of bounds, rejected")
                results['PIDE'] = None
        except Exception as e:
            print(f"Failed: {e}")
            results['PIDE'] = None
        
        return results
    
    def analyze_symbol(self, symbol):
        """Complete analysis pipeline for one symbol"""
        
        # Step 1: Data
        option = self.fetch_and_filter(symbol)
        if not option:
            return None
        
        # Step 2: Calibration
        vg_params = self.calibrate_parameters(option)
        if not vg_params:
            return None
        
        # Step 3: Pricing
        prices = self.price_all_methods(option, vg_params)
        
        # Compile results
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'option': option,
            'vg_params': {'theta': vg_params[0], 'sigma_vg': vg_params[1], 'nu': vg_params[2]},
            'prices': prices
        }
        
        self.results[symbol] = result
        return result
    
    def run_portfolio_analysis(self):
        """Analyze entire portfolio"""
        self.print_header()
        
        successful = 0
        failed = 0
        
        for category, symbols in PORTFOLIO.items():
            print(f"\n{'#'*80}")
            print(f"📂 {category}")
            print(f"{'#'*80}")
            
            for symbol in symbols:
                try:
                    result = self.analyze_symbol(symbol)
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Fatal error on {symbol}: {e}")
                    failed += 1
        
        self.generate_summary(successful, failed)
        self.create_visualizations()
        self.export_results()
    
    def generate_summary(self, successful, failed):
        """Generate comprehensive summary statistics"""
        print(f"\n{'='*80}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful: {successful} | ❌ Failed: {failed}")
        print(f"⏱️  Total Runtime: {time.time() - self.start_time:.1f}s")
        
        if not self.results:
            return
        
        # Aggregate errors by method
        errors = {'BS': [], 'MC': [], 'FFT': [], 'PIDE': []}
        times = {'BS': [], 'MC': [], 'FFT': [], 'PIDE': []}
        
        for result in self.results.values():
            for method in ['BS', 'MC', 'FFT', 'PIDE']:
                data = result['prices'].get(method)
                if data and data.get('valid', False):
                    errors[method].append(data['error_pct'])
                    times[method].append(data['time'])
        
        print(f"\n{'Method':<12} {'Count':<8} {'Mean Err%':<12} {'Median Err%':<12} {'Avg Time(s)':<12}")
        print("─"*80)
        
        for method in ['BS', 'MC', 'FFT', 'PIDE']:
            if errors[method]:
                print(f"{method:<12} {len(errors[method]):<8} "
                      f"{np.mean(errors[method]):<12.2f} "
                      f"{np.median(errors[method]):<12.2f} "
                      f"{np.mean(times[method]):<12.3f}")
        
        # Best performing method
        valid_methods = {m: np.mean(e) for m, e in errors.items() if e}
        if valid_methods:
            best = min(valid_methods, key=valid_methods.get)
            print(f"\n🏆 Best Performing: {best} (Mean Error: {valid_methods[best]:.2f}%)")
        
        self.summary_stats = {'errors': errors, 'times': times}
    
    def create_visualizations(self):
        """Create professional visualizations"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fractional Lévy Options Pricing Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Error comparison
        ax1 = axes[0, 0]
        methods = ['BS', 'MC', 'FFT', 'PIDE']
        symbols = list(self.results.keys())[:10]  # First 10
        
        x = np.arange(len(symbols))
        width = 0.2
        
        for i, method in enumerate(methods):
            errors = []
            for sym in symbols:
                data = self.results[sym]['prices'].get(method)
                if data and data.get('valid'):
                    errors.append(min(data['error_pct'], 100))  # Cap at 100%
                else:
                    errors.append(0)
            ax1.bar(x + i*width, errors, width, label=method)
        
        ax1.set_xlabel('Symbol')
        ax1.set_ylabel('Pricing Error (%)')
        ax1.set_title('Pricing Error by Method')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(symbols, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model vs Market prices
        ax2 = axes[0, 1]
        market_prices = []
        mc_prices = []
        
        for result in self.results.values():
            market_prices.append(result['option']['market_price'])
            mc_data = result['prices'].get('MC')
            if mc_data and mc_data.get('valid'):
                mc_prices.append(mc_data['price'])
            else:
                mc_prices.append(np.nan)
        
        ax2.scatter(market_prices, mc_prices, alpha=0.6, s=100)
        max_price = max(market_prices) * 1.1
        ax2.plot([0, max_price], [0, max_price], 'r--', alpha=0.5, label='Perfect Prediction')
        ax2.set_xlabel('Market Price ($)')
        ax2.set_ylabel('Model Price ($)')
        ax2.set_title('Model vs Market Prices (MC)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[1, 0]
        if self.summary_stats.get('errors'):
            for method in ['BS', 'MC']:
                errors = self.summary_stats['errors'][method]
                if errors:
                    ax3.hist(errors, bins=15, alpha=0.6, label=method)
        ax3.set_xlabel('Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Execution time
        ax4 = axes[1, 1]
        if self.summary_stats.get('times'):
            avg_times = {m: np.mean(t) if t else 0 for m, t in self.summary_stats['times'].items()}
            methods = list(avg_times.keys())
            times = list(avg_times.values())
            ax4.bar(methods, times, color=['blue', 'green', 'orange', 'red'])
            ax4.set_ylabel('Average Time (seconds)')
            ax4.set_title('Computational Performance')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: portfolio_analysis_results.png")
        plt.close()
    
    def export_results(self):
        """Export results to CSV and JSON"""
        if not self.results:
            return
        
        # Helper function to convert numpy types to Python native types
        def convert_value(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, np.bool_):
                return bool(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            else:
                return v
        
        # DataFrame for CSV
        rows = []
        for result in self.results.values():
            row = {
                'Symbol': result['symbol'],
                'Stock_Price': result['option']['stock_price'],
                'Strike': result['option']['strike'],
                'Market_Price': result['option']['market_price'],
                'Days_to_Exp': result['option']['days_to_expiration'],
                'IV': result['option']['implied_vol'],
                'VG_theta': result['vg_params']['theta'],
                'VG_sigma': result['vg_params']['sigma_vg'],
                'VG_nu': result['vg_params']['nu']
            }
            
            for method in ['BS', 'MC', 'FFT', 'PIDE']:
                data = result['prices'].get(method)
                if data and data.get('valid'):
                    row[f'{method}_Price'] = data['price']
                    row[f'{method}_Error'] = data['error_pct']
                else:
                    row[f'{method}_Price'] = np.nan
                    row[f'{method}_Error'] = np.nan
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv('portfolio_results.csv', index=False)
        print(f"📄 Results exported: portfolio_results.csv")
        
        # JSON for detailed results
        json_results = {}
        for sym, result in self.results.items():
            json_results[sym] = {
                'symbol': result['symbol'],
                'timestamp': result['timestamp'],
                'option': {k: convert_value(v) for k, v in result['option'].items()},
                'vg_params': result['vg_params'],
                'prices': {
                    method: {k: convert_value(v) for k, v in data.items() if k != 'grid'}
                    for method, data in result['prices'].items() 
                    if data and isinstance(data, dict)
                }
            }
        
        with open('portfolio_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"📄 Detailed results: portfolio_results.json")

def main():
    engine = AnalysisEngine()
    engine.run_portfolio_analysis()
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis Complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()