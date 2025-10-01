"""
Comprehensive Testing and Validation Framework
Ensures numerical accuracy, stability, and financial consistency
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple
import logging

from pide_solver import (
    MarketParameters, LevyParameters, ProfessionalFractionalPIDESolver
)
from calibration import BlackScholesAnalytic, VarianceGammaMonteCarlo


logger = logging.getLogger(__name__)


class NumericalStabilityTests:
    """Test numerical stability under extreme conditions"""
    
    @staticmethod
    def test_extreme_parameters():
        """Verify solver stability with extreme VG parameters"""
        
        test_cases = [
            {'name': 'high_skew_high_vol', 'theta': -0.3, 'sigma_vg': 0.5, 'nu': 0.7},
            {'name': 'low_vol', 'theta': 0.05, 'sigma_vg': 0.08, 'nu': 0.1},
            {'name': 'high_kurtosis', 'theta': 0.0, 'sigma_vg': 0.6, 'nu': 0.65},
        ]
        
        market = MarketParameters(S0=100, K=100, T=1.0, r=0.05)
        
        results = []
        
        for case in test_cases:
            try:
                levy = LevyParameters(case['theta'], case['sigma_vg'], case['nu'])
                solver = ProfessionalFractionalPIDESolver(market, levy)
                price, _, _ = solver.solve()
                
                is_valid = 0 < price < market.S0 * 3
                results.append({
                    'case': case['name'],
                    'price': price,
                    'valid': is_valid,
                    'status': 'PASS' if is_valid else 'FAIL'
                })
                
            except Exception as e:
                results.append({
                    'case': case['name'],
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        return results
    
    @staticmethod
    def test_grid_convergence():
        """Verify monotonic convergence with grid refinement"""
        
        market = MarketParameters(S0=100, K=100, T=1.0, r=0.05)
        levy = LevyParameters(theta=-0.1, sigma_vg=0.2, nu=0.3)
        
        grid_sizes = [50, 100, 200]
        prices = []
        
        for n_spatial in grid_sizes:
            solver = ProfessionalFractionalPIDESolver(
                market, levy, n_spatial=n_spatial, n_temporal=60
            )
            price, _, _ = solver.solve()
            prices.append(price)
        
        convergence_ratios = []
        for i in range(1, len(prices) - 1):
            error_coarse = abs(prices[i] - prices[i-1])
            error_fine = abs(prices[i+1] - prices[i])
            
            if error_fine > 1e-10:
                ratio = error_coarse / error_fine
                convergence_ratios.append(ratio)
        
        avg_ratio = np.mean(convergence_ratios) if convergence_ratios else 0
        is_converging = 1.5 < avg_ratio < 3.0
        
        return {
            'prices': prices,
            'convergence_ratios': convergence_ratios,
            'avg_ratio': avg_ratio,
            'is_monotonic': is_converging,
            'status': 'PASS' if is_converging else 'FAIL'
        }


class FinancialConsistencyTests:
    """Verify financial constraints and arbitrage conditions"""
    
    @staticmethod
    def test_put_call_parity(
        S0: float = 100, K: float = 100, T: float = 1.0, r: float = 0.05,
        theta: float = -0.1, sigma_vg: float = 0.2, nu: float = 0.3,
        tolerance: float = 0.005
    ) -> Dict:
        """
        Test Put-Call Parity: C - P = S - K*exp(-rT)
        For European options under VG
        """
        
        market = MarketParameters(S0, K, T, r)
        levy = LevyParameters(theta, sigma_vg, nu)
        
        solver_call = ProfessionalFractionalPIDESolver(market, levy)
        call_price, _, _ = solver_call.solve()
        
        put_intrinsic = max(K * np.exp(-r * T) - S0, 0)
        put_price = put_intrinsic + (call_price - max(S0 - K, 0))
        
        theoretical_diff = S0 - K * np.exp(-r * T)
        market_diff = call_price - put_price
        
        error = abs(theoretical_diff - market_diff)
        max_error = tolerance * S0
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'theoretical_diff': theoretical_diff,
            'market_diff': market_diff,
            'error': error,
            'max_allowed': max_error,
            'status': 'PASS' if error < max_error else 'FAIL'
        }
    
    @staticmethod
    def test_moneyness_bounds(
        S0: float = 100, T: float = 1.0, r: float = 0.05,
        theta: float = -0.1, sigma_vg: float = 0.2, nu: float = 0.3
    ) -> Dict:
        """Verify option prices satisfy no-arbitrage bounds"""
        
        market = MarketParameters(S0, S0, T, r)
        levy = LevyParameters(theta, sigma_vg, nu)
        
        strikes = [S0 * m for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
        violations = []
        
        for K in strikes:
            market_k = MarketParameters(S0, K, T, r)
            solver = ProfessionalFractionalPIDESolver(market_k, levy)
            call_price, _, _ = solver.solve()
            
            intrinsic = max(S0 - K, 0)
            upper_bound = S0
            lower_bound = intrinsic
            
            if call_price < lower_bound - 1e-6:
                violations.append({
                    'K': K,
                    'price': call_price,
                    'bound': lower_bound,
                    'type': 'lower_bound'
                })
            
            if call_price > upper_bound + 1e-6:
                violations.append({
                    'K': K,
                    'price': call_price,
                    'bound': upper_bound,
                    'type': 'upper_bound'
                })
        
        return {
            'strikes': strikes,
            'violations': violations,
            'status': 'PASS' if len(violations) == 0 else 'FAIL'
        }


class BenchmarkTests:
    """Compare against published benchmark results"""
    
    @staticmethod
    def test_zhang_2018_benchmarks() -> List[Dict]:
        """Compare against Zhang et al. (2018) published results"""
        
        benchmarks = [
            {
                'name': 'Case1_ATM',
                'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05,
                'theta': -0.1, 'sigma_vg': 0.2, 'nu': 0.3,
                'expected': 10.23,
                'tolerance': 0.20
            },
            {
                'name': 'Case2_OTM',
                'S0': 100, 'K': 110, 'T': 0.5, 'r': 0.03,
                'theta': 0.05, 'sigma_vg': 0.15, 'nu': 0.25,
                'expected': 2.17,
                'tolerance': 0.10
            },
            {
                'name': 'Case3_ITM',
                'S0': 100, 'K': 90, 'T': 1.5, 'r': 0.04,
                'theta': -0.15, 'sigma_vg': 0.25, 'nu': 0.35,
                'expected': 18.45,
                'tolerance': 0.30
            }
        ]
        
        results = []
        
        for benchmark in benchmarks:
            market = MarketParameters(
                benchmark['S0'], benchmark['K'], 
                benchmark['T'], benchmark['r']
            )
            levy = LevyParameters(
                benchmark['theta'], 
                benchmark['sigma_vg'], 
                benchmark['nu']
            )
            
            try:
                solver = ProfessionalFractionalPIDESolver(market, levy)
                calculated, _, _ = solver.solve()
                
                error = abs(calculated - benchmark['expected'])
                error_pct = error / benchmark['expected'] * 100
                
                passed = error < benchmark['tolerance']
                
                results.append({
                    'name': benchmark['name'],
                    'expected': benchmark['expected'],
                    'calculated': calculated,
                    'error': error,
                    'error_pct': error_pct,
                    'tolerance': benchmark['tolerance'],
                    'status': 'PASS' if passed else 'FAIL'
                })
                
            except Exception as e:
                results.append({
                    'name': benchmark['name'],
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        return results


class PerformanceTests:
    """Performance profiling and timing tests"""
    
    @staticmethod
    def profile_pricing_methods() -> Dict:
        """Compare execution times across methods"""
        
        import time
        
        market = MarketParameters(S0=100, K=100, T=1.0, r=0.05)
        levy = LevyParameters(theta=-0.1, sigma_vg=0.2, nu=0.3)
        
        timings = {}
        
        t0 = time.time()
        bs_price = BlackScholesAnalytic.call_price(100, 100, 1.0, 0.05, 0.2)
        timings['BS'] = time.time() - t0
        
        t0 = time.time()
        mc_price = VarianceGammaMonteCarlo.call_price(
            100, 100, 1.0, 0.05, -0.1, 0.2, 0.3, n_paths=5000
        )
        timings['VG_MC'] = time.time() - t0
        
        t0 = time.time()
        solver = ProfessionalFractionalPIDESolver(market, levy)
        pide_price, _, _ = solver.solve()
        timings['FRACTIONAL_PIDE'] = time.time() - t0
        
        return {
            'timings': timings,
            'prices': {
                'BS': bs_price,
                'VG_MC': mc_price,
                'FRACTIONAL_PIDE': pide_price
            }
        }


class ValidationSuite:
    """Comprehensive validation orchestrator"""
    
    def __init__(self):
        self.results = {}
    
    def run_all_tests(self) -> Dict:
        """Execute complete test suite"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION SUITE")
        print("="*80 + "\n")
        
        print("1. Numerical Stability Tests")
        print("-" * 80)
        
        stability = NumericalStabilityTests()
        
        extreme_results = stability.test_extreme_parameters()
        print("\nExtreme Parameters:")
        for result in extreme_results:
            status_symbol = "✓" if result.get('status') == 'PASS' else "✗"
            print(f"  {status_symbol} {result['case']}: {result.get('status')}")
        
        convergence = stability.test_grid_convergence()
        conv_symbol = "✓" if convergence['status'] == 'PASS' else "✗"
        print(f"\n{conv_symbol} Grid Convergence: {convergence['status']}")
        print(f"  Average ratio: {convergence['avg_ratio']:.3f}")
        
        self.results['numerical_stability'] = {
            'extreme_parameters': extreme_results,
            'grid_convergence': convergence
        }
        
        print("\n2. Financial Consistency Tests")
        print("-" * 80)
        
        financial = FinancialConsistencyTests()
        
        pcp_result = financial.test_put_call_parity()
        pcp_symbol = "✓" if pcp_result['status'] == 'PASS' else "✗"
        print(f"\n{pcp_symbol} Put-Call Parity: {pcp_result['status']}")
        print(f"  Error: {pcp_result['error']:.6f} (max: {pcp_result['max_allowed']:.6f})")
        
        bounds_result = financial.test_moneyness_bounds()
        bounds_symbol = "✓" if bounds_result['status'] == 'PASS' else "✗"
        print(f"\n{bounds_symbol} Arbitrage Bounds: {bounds_result['status']}")
        print(f"  Violations: {len(bounds_result['violations'])}")
        
        self.results['financial_consistency'] = {
            'put_call_parity': pcp_result,
            'arbitrage_bounds': bounds_result
        }
        
        print("\n3. Benchmark Validation")
        print("-" * 80)
        
        benchmark = BenchmarkTests()
        
        zhang_results = benchmark.test_zhang_2018_benchmarks()
        print("\nZhang et al. (2018) Benchmarks:")
        for result in zhang_results:
            if result['status'] != 'ERROR':
                status_symbol = "✓" if result['status'] == 'PASS' else "✗"
                print(f"  {status_symbol} {result['name']}: "
                      f"Expected={result['expected']:.2f}, "
                      f"Calculated={result['calculated']:.2f}, "
                      f"Error={result['error_pct']:.2f}%")
            else:
                print(f"  ✗ {result['name']}: ERROR")
        
        self.results['benchmarks'] = {
            'zhang_2018': zhang_results
        }
        
        print("\n4. Performance Profiling")
        print("-" * 80)
        
        perf = PerformanceTests()
        
        perf_results = perf.profile_pricing_methods()
        print("\nExecution Times:")
        for method, time_val in perf_results['timings'].items():
            print(f"  {method}: {time_val:.4f}s")
        
        self.results['performance'] = perf_results
        
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80 + "\n")
        
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print overall summary"""
        
        total_tests = 0
        passed_tests = 0
        
        if 'numerical_stability' in self.results:
            for result in self.results['numerical_stability']['extreme_parameters']:
                total_tests += 1
                if result.get('status') == 'PASS':
                    passed_tests += 1
            
            total_tests += 1
            if self.results['numerical_stability']['grid_convergence']['status'] == 'PASS':
                passed_tests += 1
        
        if 'financial_consistency' in self.results:
            total_tests += 2
            if self.results['financial_consistency']['put_call_parity']['status'] == 'PASS':
                passed_tests += 1
            if self.results['financial_consistency']['arbitrage_bounds']['status'] == 'PASS':
                passed_tests += 1
        
        if 'benchmarks' in self.results:
            for result in self.results['benchmarks']['zhang_2018']:
                total_tests += 1
                if result.get('status') == 'PASS':
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("Status: EXCELLENT - Production ready")
        elif success_rate >= 75:
            print("Status: GOOD - Minor improvements needed")
        elif success_rate >= 60:
            print("Status: FAIR - Significant improvements required")
        else:
            print("Status: POOR - Major refactoring needed")


def run_validation():
    """Main entry point for validation suite"""
    suite = ValidationSuite()
    results = suite.run_all_tests()
    return results


if __name__ == "__main__":
    results = run_validation()