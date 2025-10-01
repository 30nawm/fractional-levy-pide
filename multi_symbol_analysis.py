# multi_symbol_analysis.py
# Comprehensive analysis across multiple symbols and markets
# US Equities, European indices, Energy/Oil options

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from pide_solver import fractional_pide_solver
from calibration import bs_call_price, vg_mc_price, vg_call_fft, calibrate_to_real_options
from levy_simulation import simulate_vg_paths

# Symbol configurations for comprehensive analysis
SYMBOL_CONFIGS = {
    # Top 3 US Equities by options volume
    'US_EQUITIES': [
        {'symbol': 'SPY', 'name': 'S&P 500 ETF', 'type': 'ETF', 'market': 'US'},
        {'symbol': 'QQQ', 'name': 'Nasdaq-100 ETF', 'type': 'ETF', 'market': 'US'},
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'Stock', 'market': 'US'}
    ],
    
    # Top 3 European indices/ETFs
    'EUROPEAN': [
        {'symbol': 'EWG', 'name': 'iShares MSCI Germany ETF', 'type': 'ETF', 'market': 'EU'},
        {'symbol': 'EWU', 'name': 'iShares MSCI United Kingdom ETF', 'type': 'ETF', 'market': 'EU'},
        {'symbol': 'EZU', 'name': 'iShares MSCI Eurozone ETF', 'type': 'ETF', 'market': 'EU'}
    ],
    
    # Energy/Oil related options
    'ENERGY': [
        {'symbol': 'USO', 'name': 'United States Oil Fund', 'type': 'ETF', 'market': 'Energy'},
        {'symbol': 'XLE', 'name': 'Energy Select Sector SPDR', 'type': 'ETF', 'market': 'Energy'},
        {'symbol': 'XOP', 'name': 'SPDR S&P Oil & Gas Exploration ETF', 'type': 'ETF', 'market': 'Energy'}
    ]
}

class MultiSymbolAnalyzer:
    """
    Comprehensive multi-symbol options pricing analysis
    Compares different models across various markets and asset classes
    """
    
    def __init__(self, symbols_dict=SYMBOL_CONFIGS):
        self.symbols = symbols_dict
        self.results = {}
        self.summary_stats = {}
    
    def analyze_single_symbol(self, symbol_info):
        """
        Analyze a single symbol with all pricing models
        """
        symbol = symbol_info['symbol']
        print(f"\n{'='*60}")
        print(f"Analyzing: {symbol_info['name']} ({symbol})")
        print(f"Market: {symbol_info['market']} | Type: {symbol_info['type']}")
        print(f"{'='*60}")
        
        try:
            # Get real options data
            from main_pipeline import get_real_options_data, get_historical_volatility
            
            option_data = get_real_options_data(symbol)
            if not option_data:
                print(f"Skipping {symbol}: No options data available")
                return None
            
            # Update global parameters
            global S0, K, T, r, sigma
            from configuration import r
            S0 = option_data['stock_price']
            K = option_data['strike']
            T = option_data['time_to_expiration']
            sigma = option_data['implied_vol']
            
            # Get historical volatility
            hist_vol, returns = get_historical_volatility(symbol, period_days=252)
            
            # Calibrate VG parameters
            print("\nCalibrating VG parameters...")
            theta_cal, sigma_vg_cal, nu_cal = calibrate_to_real_options(option_data)
            
            # Price using all methods
            print("\nPricing with all methods...")
            
            results = {
                'symbol': symbol,
                'name': symbol_info['name'],
                'market': symbol_info['market'],
                'type': symbol_info['type'],
                'market_data': {
                    'stock_price': S0,
                    'strike': K,
                    'time_to_exp': T,
                    'market_price': option_data['market_price'],
                    'implied_vol': option_data['implied_vol'],
                    'historical_vol': hist_vol,
                    'volume': option_data.get('volume', 0)
                },
                'vg_params': {
                    'theta': theta_cal,
                    'sigma_vg': sigma_vg_cal,
                    'nu': nu_cal
                },
                'prices': {},
                'errors': {},
                'timing': {}
            }
            
            market_price = option_data['market_price']
            
            # 1. Black-Scholes
            import time
            start = time.time()
            bs_price = bs_call_price(S0, K, T, r, sigma)
            results['timing']['BS'] = time.time() - start
            results['prices']['BS'] = bs_price
            results['errors']['BS'] = abs(bs_price - market_price) / market_price * 100
            print(f"  BS: ${bs_price:.4f} (Error: {results['errors']['BS']:.2f}%)")
            
            # 2. VG Monte Carlo
            start = time.time()
            mc_price = vg_mc_price(theta_cal, sigma_vg_cal, nu_cal, n_paths=2000)
            results['timing']['MC'] = time.time() - start
            results['prices']['MC'] = mc_price
            results['errors']['MC'] = abs(mc_price - market_price) / market_price * 100
            print(f"  MC: ${mc_price:.4f} (Error: {results['errors']['MC']:.2f}%)")
            
            # 3. VG FFT
            try:
                start = time.time()
                fft_price = vg_call_fft(S0, K, T, r, theta_cal, sigma_vg_cal, nu_cal)
                results['timing']['FFT'] = time.time() - start
                results['prices']['FFT'] = fft_price
                results['errors']['FFT'] = abs(fft_price - market_price) / market_price * 100
                print(f"  FFT: ${fft_price:.4f} (Error: {results['errors']['FFT']:.2f}%)")
            except:
                results['prices']['FFT'] = np.nan
                results['errors']['FFT'] = np.nan
                print(f"  FFT: Failed")
            
            # 4. Fractional PIDE
            try:
                start = time.time()
                pide_price, _, _ = fractional_pide_solver(theta_cal, sigma_vg_cal, nu_cal)
                results['timing']['PIDE'] = time.time() - start
                
                # Validate PIDE result
                if pide_price > 0 and pide_price < S0 * 2:  # Reasonable bounds
                    results['prices']['PIDE'] = pide_price
                    results['errors']['PIDE'] = abs(pide_price - market_price) / market_price * 100
                    print(f"  PIDE: ${pide_price:.4f} (Error: {results['errors']['PIDE']:.2f}%)")
                else:
                    results['prices']['PIDE'] = np.nan
                    results['errors']['PIDE'] = np.nan
                    print(f"  PIDE: ${pide_price:.4f} (Out of bounds - rejected)")
            except Exception as e:
                results['prices']['PIDE'] = np.nan
                results['errors']['PIDE'] = np.nan
                print(f"  PIDE: Failed ({str(e)[:50]})")
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_comprehensive_analysis(self):
        """
        Run analysis across all configured symbols
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MULTI-SYMBOL OPTIONS PRICING ANALYSIS")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Symbols: {sum(len(v) for v in self.symbols.values())}")
        
        all_results = []
        
        for category, symbol_list in self.symbols.items():
            print(f"\n{'#'*80}")
            print(f"# {category}")
            print(f"{'#'*80}")
            
            for symbol_info in symbol_list:
                result = self.analyze_single_symbol(symbol_info)
                if result:
                    all_results.append(result)
                    self.results[symbol_info['symbol']] = result
        
        # Generate summary statistics
        self.generate_summary_statistics(all_results)
        
        # Create comparative visualizations
        self.create_comparative_visualizations(all_results)
        
        # Export results
        self.export_results(all_results)
        
        return all_results
    
    def generate_summary_statistics(self, results):
        """
        Generate summary statistics across all analyzed symbols
        """
        if not results:
            print("\nNo results to summarize")
            return
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Aggregate errors by method
        methods = ['BS', 'MC', 'FFT', 'PIDE']
        
        summary = {}
        for method in methods:
            errors = [r['errors'].get(method, np.nan) for r in results if not np.isnan(r['errors'].get(method, np.nan))]
            if errors:
                summary[method] = {
                    'count': len(errors),
                    'mean_error': np.mean(errors),
                    'median_error': np.median(errors),
                    'std_error': np.std(errors),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors)
                }
        
        # Print summary table
        print(f"\n{'Method':<15} {'Count':<8} {'Mean %':<10} {'Median %':<10} {'Std %':<10} {'Min %':<10} {'Max %':<10}")
        print("-" * 80)
        
        for method, stats in summary.items():
            print(f"{method:<15} {stats['count']:<8} "
                  f"{stats['mean_error']:<10.2f} {stats['median_error']:<10.2f} "
                  f"{stats['std_error']:<10.2f} {stats['min_error']:<10.2f} {stats['max_error']:<10.2f}")
        
        # Best performing method
        best_method = min(summary.items(), key=lambda x: x[1]['mean_error'])
        print(f"\nBest Performing Method: {best_method[0]} (Mean Error: {best_method[1]['mean_error']:.2f}%)")
        
        self.summary_stats = summary
    
    def create_comparative_visualizations(self, results):
        """
        Create comparative visualizations across all symbols
        """
        if not results:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Error comparison by method
            ax1 = axes[0, 0]
            methods = ['BS', 'MC', 'FFT', 'PIDE']
            symbols = [r['symbol'] for r in results]
            
            x = np.arange(len(symbols))
            width = 0.2
            
            for i, method in enumerate(methods):
                errors = [r['errors'].get(method, np.nan) for r in results]
                errors = [e if not np.isnan(e) and e < 200 else 200 for e in errors]  # Cap at 200%
                ax1.bar(x + i*width, errors, width, label=method)
            
            ax1.set_xlabel('Symbol')
            ax1.set_ylabel('Absolute Error (%)')
            ax1.set_title('Pricing Error by Method and Symbol')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(symbols, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Market price vs model prices
            ax2 = axes[0, 1]
            market_prices = [r['market_data']['market_price'] for r in results]
            bs_prices = [r['prices'].get('BS', np.nan) for r in results]
            mc_prices = [r['prices'].get('MC', np.nan) for r in results]
            
            ax2.scatter(market_prices, bs_prices, label='BS', alpha=0.7, s=100)
            ax2.scatter(market_prices, mc_prices, label='MC', alpha=0.7, s=100)
            
            # Perfect prediction line
            max_price = max(market_prices) * 1.1
            ax2.plot([0, max_price], [0, max_price], 'k--', alpha=0.5, label='Perfect Prediction')
            
            ax2.set_xlabel('Market Price ($)')
            ax2.set_ylabel('Model Price ($)')
            ax2.set_title('Market Price vs Model Prices')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Execution time comparison
            ax3 = axes[1, 0]
            timing_data = {method: [] for method in methods}
            
            for r in results:
                for method in methods:
                    if method in r['timing']:
                        timing_data[method].append(r['timing'][method])
            
            avg_times = {m: np.mean(times) if times else 0 for m, times in timing_data.items()}
            ax3.bar(avg_times.keys(), avg_times.values())
            ax3.set_ylabel('Average Time (seconds)')
            ax3.set_title('Average Execution Time by Method')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Error distribution
            ax4 = axes[1, 1]
            for method in ['BS', 'MC']:
                errors = [r['errors'].get(method, np.nan) for r in results if not np.isnan(r['errors'].get(method, np.nan))]
                if errors:
                    ax4.hist(errors, bins=15, alpha=0.6, label=method)
            
            ax4.set_xlabel('Error (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Error Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('multi_symbol_comparative_analysis.png', dpi=300, bbox_inches='tight')
            print("\nComparative visualization saved: multi_symbol_comparative_analysis.png")
            plt.close()
            
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    def export_results(self, results):
        """
        Export results to CSV for further analysis
        """
        if not results:
            return
        
        try:
            # Flatten results for DataFrame
            rows = []
            for r in results:
                row = {
                    'Symbol': r['symbol'],
                    'Name': r['name'],
                    'Market': r['market'],
                    'Type': r['type'],
                    'Stock_Price': r['market_data']['stock_price'],
                    'Strike': r['market_data']['strike'],
                    'Time_to_Exp': r['market_data']['time_to_exp'],
                    'Market_Price': r['market_data']['market_price'],
                    'Implied_Vol': r['market_data']['implied_vol'],
                    'Historical_Vol': r['market_data']['historical_vol']
                }
                
                # Add prices and errors
                for method in ['BS', 'MC', 'FFT', 'PIDE']:
                    row[f'{method}_Price'] = r['prices'].get(method, np.nan)
                    row[f'{method}_Error'] = r['errors'].get(method, np.nan)
                    row[f'{method}_Time'] = r['timing'].get(method, np.nan)
                
                # Add VG parameters
                row['VG_theta'] = r['vg_params']['theta']
                row['VG_sigma'] = r['vg_params']['sigma_vg']
                row['VG_nu'] = r['vg_params']['nu']
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv('multi_symbol_results.csv', index=False)
            print(f"\nResults exported to: multi_symbol_results.csv")
            print(f"Total symbols analyzed: {len(df)}")
            
        except Exception as e:
            print(f"Export failed: {e}")

def run_multi_symbol_analysis():
    """Main entry point for multi-symbol analysis"""
    analyzer = MultiSymbolAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = run_multi_symbol_analysis()