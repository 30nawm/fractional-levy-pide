"""
Production Pipeline for Fractional Lévy Options Pricing
Integrates all components into unified analysis framework
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

from pide_solver import (
    MarketParameters, LevyParameters, ProfessionalFractionalPIDESolver
)
from calibration import (
    MultiStrikeCalibrator, OptionQuote, BlackScholesAnalytic,
    VarianceGammaMonteCarlo, VarianceGammaFFT
)
from data_infrastructure import (
    DataProvider, FuturesDataProvider, OptionData, FuturesUniverse
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PricingEngine:
    """Unified pricing engine supporting multiple models"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.results = {}
    
    def price_all_methods(
        self,
        option: OptionData,
        vg_params: Dict[str, float],
        alpha: float = 0.85
    ) -> Dict:
        """Price option using all available methods"""
        
        S0 = option.underlying_price
        K = option.strike
        T = option.time_to_expiry
        market_price = option.market_price
        
        theta = vg_params['theta']
        sigma_vg = vg_params['sigma_vg']
        nu = vg_params['nu']
        
        results = {}
        
        logger.info(f"Pricing {option.symbol} K={K:.2f} with all methods")
        
        try:
            t0 = time.time()
            bs_price = BlackScholesAnalytic.call_price(
                S0, K, T, self.r, option.implied_vol
            )
            results['BS'] = {
                'price': bs_price,
                'error_pct': abs(bs_price - market_price) / market_price * 100,
                'time': time.time() - t0,
                'valid': 0 < bs_price < S0 * 1.5
            }
            logger.info(f"  BS: ${bs_price:.4f} ({results['BS']['error_pct']:.2f}% error)")
        except Exception as e:
            logger.error(f"  BS failed: {e}")
            results['BS'] = None
        
        try:
            t0 = time.time()
            mc_price = VarianceGammaMonteCarlo.call_price(
                S0, K, T, self.r, theta, sigma_vg, nu, n_paths=5000
            )
            results['VG_MC'] = {
                'price': mc_price,
                'error_pct': abs(mc_price - market_price) / market_price * 100,
                'time': time.time() - t0,
                'valid': 0 < mc_price < S0 * 1.5
            }
            logger.info(f"  VG_MC: ${mc_price:.4f} ({results['VG_MC']['error_pct']:.2f}% error)")
        except Exception as e:
            logger.error(f"  VG_MC failed: {e}")
            results['VG_MC'] = None
        
        try:
            t0 = time.time()
            fft_price = VarianceGammaFFT.call_price(
                S0, K, T, self.r, theta, sigma_vg, nu
            )
            results['VG_FFT'] = {
                'price': fft_price,
                'error_pct': abs(fft_price - market_price) / market_price * 100,
                'time': time.time() - t0,
                'valid': 0 < fft_price < S0 * 1.5
            }
            logger.info(f"  VG_FFT: ${fft_price:.4f} ({results['VG_FFT']['error_pct']:.2f}% error)")
        except Exception as e:
            logger.error(f"  VG_FFT failed: {e}")
            results['VG_FFT'] = None
        
        try:
            t0 = time.time()
            market = MarketParameters(S0, K, T, self.r)
            levy = LevyParameters(theta, sigma_vg, nu)
            
            solver = ProfessionalFractionalPIDESolver(market, levy, alpha)
            pide_price, _, _ = solver.solve()
            
            results['FRACTIONAL_PIDE'] = {
                'price': pide_price,
                'error_pct': abs(pide_price - market_price) / market_price * 100,
                'time': time.time() - t0,
                'valid': 0 < pide_price < S0 * 1.5
            }
            logger.info(f"  PIDE: ${pide_price:.4f} ({results['FRACTIONAL_PIDE']['error_pct']:.2f}% error)")
        except Exception as e:
            logger.error(f"  PIDE failed: {e}")
            results['FRACTIONAL_PIDE'] = None
        
        return results


class PortfolioAnalyzer:
    """Complete portfolio analysis pipeline"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_provider = DataProvider()
        self.futures_provider = FuturesDataProvider()
        self.pricing_engine = PricingEngine()
        
        self.results = []
        self.start_time = None
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Complete analysis for single symbol"""
        
        logger.info(f"Analyzing {symbol}")
        
        option = self.data_provider.select_atm_option(symbol)
        
        if option is None:
            logger.warning(f"No suitable option found for {symbol}")
            return None
        
        logger.info(f"Selected: K=${option.strike:.2f}, "
                   f"T={option.days_to_expiry}d, "
                   f"Vol={option.volume}, "
                   f"IV={option.implied_vol:.1%}")
        
        quote = OptionQuote(
            strike=option.strike,
            market_price=option.market_price,
            implied_vol=option.implied_vol,
            time_to_expiry=option.time_to_expiry,
            volume=option.volume,
            open_interest=option.open_interest,
            bid=option.bid,
            ask=option.ask
        )
        
        try:
            calibrator = MultiStrikeCalibrator(
                option.underlying_price,
                self.pricing_engine.r,
                [quote],
                pricing_method='mc'
            )
            
            calib_result = calibrator.calibrate(method='local')
            
            logger.info(f"Calibrated: θ={calib_result['theta']:.4f}, "
                       f"σ={calib_result['sigma_vg']:.4f}, "
                       f"ν={calib_result['nu']:.4f}")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return None
        
        pricing_results = self.pricing_engine.price_all_methods(
            option,
            {
                'theta': calib_result['theta'],
                'sigma_vg': calib_result['sigma_vg'],
                'nu': calib_result['nu']
            }
        )
        
        hist_vol, _ = self.data_provider.calculate_historical_volatility(symbol)
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'option': {
                'strike': option.strike,
                'expiration': option.expiration,
                'market_price': option.market_price,
                'underlying_price': option.underlying_price,
                'implied_vol': option.implied_vol,
                'historical_vol': hist_vol,
                'volume': option.volume,
                'days_to_expiry': option.days_to_expiry
            },
            'vg_parameters': {
                'theta': calib_result['theta'],
                'sigma_vg': calib_result['sigma_vg'],
                'nu': calib_result['nu'],
                'calibration_rmse': calib_result['rmse']
            },
            'prices': pricing_results
        }
        
        return result
    
    def run_portfolio(self, symbols: List[str]) -> pd.DataFrame:
        """Execute analysis on portfolio"""
        
        self.start_time = time.time()
        self.results = []
        
        logger.info(f"Starting portfolio analysis: {len(symbols)} symbols")
        print("\n" + "="*80)
        print("FRACTIONAL LÉVY OPTIONS PRICING - PORTFOLIO ANALYSIS")
        print("="*80)
        print(f"Symbols: {len(symbols)}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol)
                if result:
                    self.results.append(result)
                    logger.info(f"✓ {symbol} completed")
                else:
                    logger.warning(f"✗ {symbol} failed")
            except Exception as e:
                logger.error(f"Fatal error on {symbol}: {e}")
        
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"Analysis Complete")
        print(f"{'='*80}")
        print(f"Successful: {len(self.results)}/{len(symbols)}")
        print(f"Runtime: {elapsed:.1f}s")
        print(f"{'='*80}\n")
        
        return self._generate_summary()
    
    def _generate_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame"""
        
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        
        for result in self.results:
            row = {
                'Symbol': result['symbol'],
                'Underlying': result['option']['underlying_price'],
                'Strike': result['option']['strike'],
                'Market_Price': result['option']['market_price'],
                'IV': result['option']['implied_vol'],
                'Hist_Vol': result['option']['historical_vol'],
                'VG_theta': result['vg_parameters']['theta'],
                'VG_sigma': result['vg_parameters']['sigma_vg'],
                'VG_nu': result['vg_parameters']['nu']
            }
            
            for method in ['BS', 'VG_MC', 'VG_FFT', 'FRACTIONAL_PIDE']:
                data = result['prices'].get(method)
                if data and data.get('valid'):
                    row[f'{method}_Price'] = data['price']
                    row[f'{method}_Error%'] = data['error_pct']
                    row[f'{method}_Time'] = data['time']
                else:
                    row[f'{method}_Price'] = np.nan
                    row[f'{method}_Error%'] = np.nan
                    row[f'{method}_Time'] = np.nan
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        csv_path = self.output_dir / f"portfolio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")
        
        json_path = self.output_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Detailed results: {json_path}")
        
        self._print_summary_statistics(df)
        
        return df
    
    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics"""
        
        print("\nSUMMARY STATISTICS")
        print("-" * 80)
        
        methods = ['BS', 'VG_MC', 'VG_FFT', 'FRACTIONAL_PIDE']
        
        print(f"{'Method':<20} {'Count':<8} {'Mean Err%':<12} {'Median Err%':<12} {'Avg Time(s)':<12}")
        print("-" * 80)
        
        for method in methods:
            error_col = f'{method}_Error%'
            time_col = f'{method}_Time'
            
            if error_col in df.columns:
                valid_errors = df[error_col].dropna()
                valid_times = df[time_col].dropna()
                
                if len(valid_errors) > 0:
                    print(f"{method:<20} {len(valid_errors):<8} "
                          f"{valid_errors.mean():<12.2f} "
                          f"{valid_errors.median():<12.2f} "
                          f"{valid_times.mean():<12.3f}")
        
        print("-" * 80)


def main():
    """Main execution entry point"""
    
    analyzer = PortfolioAnalyzer()
    
    test_symbols = [
        'SPY', 'QQQ', 'IWM',
        'AAPL', 'MSFT', 'GOOGL',
        'GLD', 'SLV', 'USO'
    ]
    
    df = analyzer.run_portfolio(test_symbols)
    
    print(f"\n✓ Analysis complete. Results in: {analyzer.output_dir}")
    
    return analyzer, df


if __name__ == "__main__":
    analyzer, results = main()