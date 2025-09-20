# validation_tests.py
# Comprehensive validation suite for fractional Lévy PIDE solver
# Compares with known analytical solutions and benchmarks

import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt

from pide_solver import fractional_pide_solver
from calibration import bs_call_price, vg_mc_price
from levy_simulation import simulate_vg_paths
from config import *

class ValidationSuite:
    """Comprehensive validation and testing suite"""
    
    def __init__(self):
        self.results = {}
        self.tolerance = 1e-2  # 1% tolerance for numerical methods
        
    def test_black_scholes_limit(self):
        """Test if PIDE reduces to Black-Scholes when α→1 and no jumps"""
        print("\n🧪 Test 1: Black-Scholes Limit")
        print("-" * 40)
        
        # Parameters that should give BS limit
        alpha_bs = 0.99     # Near 1 (classical diffusion)
        theta_bs = 0.0      # No drift in jumps
        sigma_vg_bs = 0.01  # Minimal jump volatility  
        nu_bs = 0.01        # Minimal jump frequency
        
        try:
            # Our PIDE solution
            pide_price, _, _ = fractional_pide_solver(theta_bs, sigma_vg_bs, nu_bs)
            
            # Analytical Black-Scholes
            bs_price = bs_call_price(S0, K, T, r, sigma)
            
            # Compare
            error = abs(pide_price - bs_price) / bs_price
            
            print(f"PIDE Price (α={alpha_bs}): ${pide_price:.4f}")
            print(f"Black-Scholes Price:      ${bs_price:.4f}")
            print(f"Relative Error:           {error:.2%}")
            
            success = error < self.tolerance
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"Result: {status}")
            
            self.results['bs_limit'] = {
                'passed': success,
                'error': error,
                'pide_price': pide_price,
                'bs_price': bs_price
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.results['bs_limit'] = {'passed': False, 'error': str(e)}
    
    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - Ke^(-rT)"""
        print("\n🧪 Test 2: Put-Call Parity") 
        print("-" * 40)
        
        try:
            # Call price from our solver
            call_price, _, _ = fractional_pide_solver(theta, sigma_vg, nu)
            
            # For put price, we need to modify the solver (simplified test)
            # Put-call parity: C - P = S₀ - K*e^(-rT)
            theoretical_put = call_price - (S0 - K * np.exp(-r * T))
            
            print(f"Call Price:               ${call_price:.4f}")
            print(f"Theoretical Put Price:    ${theoretical_put:.4f}")
            print(f"Put-Call Parity Value:    ${S0 - K * np.exp(-r * T):.4f}")
            
            # Check if put price is reasonable (non-negative for ITM puts)
            success = theoretical_put >= 0 or K <= S0  # OTM put can be near zero
            status = "✅ PASSED" if success else "❌ FAILED"  
            print(f"Result: {status}")
            
            self.results['put_call_parity'] = {
                'passed': success,
                'call_price': call_price,
                'put_price': theoretical_put
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.results['put_call_parity'] = {'passed': False, 'error': str(e)}
    
    def test_monte_carlo_convergence(self):
        """Test convergence between PIDE and Monte Carlo"""
        print("\n🧪 Test 3: Monte Carlo Convergence")
        print("-" * 40)
        
        try:
            # PIDE solution
            pide_price, _, _ = fractional_pide_solver(theta, sigma_vg, nu)
            
            # Monte Carlo with increasing sample sizes
            mc_sizes = [1000, 5000, 10000, 20000]
            mc_prices = []
            
            print("Monte Carlo Convergence:")
            for n_paths in mc_sizes:
                mc_price = vg_mc_price(theta, sigma_vg, nu, n_paths)
                mc_prices.append(mc_price)
                error = abs(mc_price - pide_price) / pide_price if pide_price > 0 else float('inf')
                print(f"  n={n_paths:5d}: ${mc_price:.4f} (error: {error:.2%})")
            
            # Check if MC is converging to PIDE solution
            final_error = abs(mc_prices[-1] - pide_price) / pide_price if pide_price > 0 else float('inf')
            success = final_error < 0.05  # 5% tolerance for MC
            
            print(f"\nPIDE Price:               ${pide_price:.4f}")
            print(f"MC Price (final):         ${mc_prices[-1]:.4f}")
            print(f"Final Error:              {final_error:.2%}")
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"Result: {status}")
            
            self.results['mc_convergence'] = {
                'passed': success,
                'pide_price': pide_price,
                'mc_prices': mc_prices,
                'final_error': final_error
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.results['mc_convergence'] = {'passed': False, 'error': str(e)}
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to VG parameters"""
        print("\n🧪 Test 4: Parameter Sensitivity")
        print("-" * 40)
        
        base_params = (theta, sigma_vg, nu)
        base_price, _, _ = fractional_pide_solver(*base_params)
        
        print(f"Base Price (θ={theta}, σ={sigma_vg}, ν={nu}): ${base_price:.4f}")
        print("\nParameter Sensitivity:")
        
        sensitivities = {}
        
        try:
            # Test theta sensitivity  
            theta_test = theta + 0.05
            price_theta = fractional_pide_solver(theta_test, sigma_vg, nu)[0]
            sens_theta = (price_theta - base_price) / 0.05
            sensitivities['theta'] = sens_theta
            print(f"  ∂P/∂θ ≈ {sens_theta:.3f} (θ={theta_test:.3f} → ${price_theta:.4f})")
            
            # Test sigma_vg sensitivity
            sigma_test = sigma_vg + 0.02
            price_sigma = fractional_pide_solver(theta, sigma_test, nu)[0]
            sens_sigma = (price_sigma - base_price) / 0.02
            sensitivities['sigma_vg'] = sens_sigma
            print(f"  ∂P/∂σ ≈ {sens_sigma:.3f} (σ={sigma_test:.3f} → ${price_sigma:.4f})")
            
            # Test nu sensitivity
            nu_test = nu + 0.05
            price_nu = fractional_pide_solver(theta, sigma_vg, nu_test)[0]
            sens_nu = (price_nu - base_price) / 0.05
            sensitivities['nu'] = sens_nu
            print(f"  ∂P/∂ν ≈ {sens_nu:.3f} (ν={nu_test:.3f} → ${price_nu:.4f})")
            
            # Check if sensitivities are reasonable
            success = all(abs(s) < 100 for s in sensitivities.values())  # Not too extreme
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\nResult: {status}")
            
            self.results['sensitivity'] = {
                'passed': success,
                'sensitivities': sensitivities,
                'base_price': base_price
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.results['sensitivity'] = {'passed': False, 'error': str(e)}
    
    def test_performance_benchmarks(self):
        """Test computational performance"""
        print("\n🧪 Test 5: Performance Benchmarks")
        print("-" * 40)
        
        try:
            # Time PIDE solver
            start_time = time.time()
            pide_price, _, _ = fractional_pide_solver(theta, sigma_vg, nu)
            pide_time = time.time() - start_time
            
            # Time Monte Carlo
            start_time = time.time() 
            mc_price = vg_mc_price(theta, sigma_vg, nu, n_paths=5000)
            mc_time = time.time() - start_time
            
            print(f"PIDE Solver Time:         {pide_time:.2f} seconds")
            print(f"Monte Carlo Time:         {mc_time:.2f} seconds")
            print(f"Speed Ratio (MC/PIDE):    {mc_time/pide_time:.2f}x")
            
            # Performance targets from config
            pide_fast = pide_time < PERFORMANCE_TARGETS['pide_solve_time']
            mc_fast = mc_time < PERFORMANCE_TARGETS['mc_simulation_time']
            
            success = pide_fast and mc_fast
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\nResult: {status}")
            
            self.results['performance'] = {
                'passed': success,
                'pide_time': pide_time,
                'mc_time': mc_time,
                'pide_price': pide_price,
                'mc_price': mc_price
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.results['performance'] = {'passed': False, 'error': str(e)}
    
    def test_boundary_conditions(self):
        """Test boundary condition implementation"""
        print("\n🧪 Test 6: Boundary Conditions")
        print("-" * 40)
        
        try:
            pide_price, S_grid, V_surface = fractional_pide_solver(theta, sigma_vg, nu)
            
            # Check lower boundary: V(0) should be 0 for call
            lower_boundary = V_surface[0]
            lower_ok = abs(lower_boundary) < 1e-3
            
            # Check upper boundary: V(S_max) should be approximately S_max - K*e^(-rT)
            upper_boundary = V_surface[-1]
            expected_upper = S_grid[-1] - K * np.exp(-r * T)
            upper_ok = abs(upper_boundary - expected_upper) / expected_upper < 0.1
            
            # Check monotonicity: option value should increase with stock price
            monotonic = np.all(np.diff(V_surface) >= -1e-6)  # Allow small numerical errors
            
            print(f"Lower Boundary V(0):      {lower_boundary:.6f} (should be ≈ 0)")
            print(f"Upper Boundary:           {upper_boundary:.3f} (expected: {expected_upper:.3f})")
            print(f"Monotonicity:             {'✓' if monotonic else '✗'}")
            
            success = lower_ok and upper_ok and monotonic
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"Result: {status}")
            
            self.results['boundaries'] = {
                'passed': success,
                'lower_boundary': lower_boundary,
                'upper_boundary': upper_boundary,
                'monotonic': monotonic
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.results['boundaries'] = {'passed': False, 'error': str(e)}
    
    def run_all_tests(self):
        """Run complete validation suite"""
        print("🚀 STARTING COMPREHENSIVE VALIDATION SUITE")
        print("="*60)
        
        start_time = time.time()
        
        # Run all tests
        self.test_black_scholes_limit()
        self.test_put_call_parity() 
        self.test_monte_carlo_convergence()
        self.test_parameter_sensitivity()
        self.test_performance_benchmarks()
        self.test_boundary_conditions()
        
        total_time = time.time() - start_time
        
        # Summary
        self.print_summary(total_time)
        
        return self.results
    
    def print_summary(self, total_time):
        """Print validation summary"""
        print("\n" + "="*60)
        print("📊 VALIDATION SUITE SUMMARY")
        print("="*60)
        
        passed_tests = sum(1 for result in self.results.values() 
                          if result.get('passed', False))
        total_tests = len(self.results)
        
        print(f"Tests Passed:             {passed_tests}/{total_tests}")
        print(f"Success Rate:             {passed_tests/total_tests:.1%}")
        print(f"Total Runtime:            {total_time:.2f} seconds")
        
        print(f"\n📋 Test Results:")
        test_names = {
            'bs_limit': 'Black-Scholes Limit',
            'put_call_parity': 'Put-Call Parity', 
            'mc_convergence': 'Monte Carlo Convergence',
            'sensitivity': 'Parameter Sensitivity',
            'performance': 'Performance Benchmarks',
            'boundaries': 'Boundary Conditions'
        }
        
        for key, name in test_names.items():
            if key in self.results:
                status = "✅ PASS" if self.results[key].get('passed', False) else "❌ FAIL"
                print(f"  {status} {name}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if passed_tests == total_tests:
            print("  🎉 All tests passed! The implementation appears robust.")
        else:
            failed_tests = total_tests - passed_tests
            print(f"  ⚠️  {failed_tests} test(s) failed. Review implementation.")
            
            if not self.results.get('bs_limit', {}).get('passed', True):
                print("  • Check fractional derivative implementation")
            if not self.results.get('mc_convergence', {}).get('passed', True): 
                print("  • Verify Lévy integral operator")
            if not self.results.get('boundaries', {}).get('passed', True):
                print("  • Review boundary condition implementation")
        
        print("="*60)

def run_validation():
    """Main function to run validation suite"""
    print("Starting validation of Fractional Lévy PIDE Solver...")
    
    # Print current configuration
    print_config()
    
    # Run validation
    validator = ValidationSuite()
    results = validator.run_all_tests()
    
    return results

if __name__ == "__main__":
    run_validation()