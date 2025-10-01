"""
Multi-Strike Calibration Engine for Variance Gamma Process
Implements global optimization with volatility smile fitting
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OptionQuote:
    """Single option market quote"""
    strike: float
    market_price: float
    implied_vol: float
    time_to_expiry: float
    volume: float
    open_interest: float
    bid: float
    ask: float
    
    @property
    def weight(self) -> float:
        """Liquidity-based weight for calibration"""
        return np.sqrt(self.volume + 0.5 * self.open_interest)
    
    @property
    def mid_price(self) -> float:
        return 0.5 * (self.bid + self.ask)


class BlackScholesAnalytic:
    """Closed-form Black-Scholes for benchmark"""
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)


class VarianceGammaMonteCarlo:
    """Monte Carlo pricing for VG process"""
    
    @staticmethod
    def simulate_terminal(
        S0: float, K: float, T: float, r: float,
        theta: float, sigma_vg: float, nu: float,
        n_paths: int = 5000, seed: Optional[int] = None
    ) -> np.ndarray:
        
        if seed is not None:
            np.random.seed(seed)
        
        omega = -np.log(1.0 - theta * nu - 0.5 * sigma_vg**2 * nu) / nu
        drift = (r - omega) * T
        
        gamma_shape = T / nu
        gamma_scale = nu
        
        G = np.random.gamma(gamma_shape, gamma_scale, n_paths)
        Z = np.random.randn(n_paths)
        
        X = theta * G + sigma_vg * np.sqrt(G) * Z
        S_T = S0 * np.exp(drift + X)
        
        return S_T
    
    @classmethod
    def call_price(
        cls, S0: float, K: float, T: float, r: float,
        theta: float, sigma_vg: float, nu: float,
        n_paths: int = 5000
    ) -> float:
        
        S_T = cls.simulate_terminal(S0, K, T, r, theta, sigma_vg, nu, n_paths)
        payoffs = np.maximum(S_T - K, 0)
        return np.exp(-r * T) * np.mean(payoffs)


class VarianceGammaFFT:
    """FFT-based pricing using Carr-Madan method"""
    
    @staticmethod
    def call_price(
        S0: float, K: float, T: float, r: float,
        theta: float, sigma_vg: float, nu: float,
        N: int = 4096, eta: float = 0.15, alpha: float = 1.1
    ) -> float:
        
        lambda_val = 2 * np.pi / (N * eta)
        b = N * lambda_val / 2
        k_u = -b + lambda_val * np.arange(N)
        v_j = eta * np.arange(N)
        
        omega = -np.log(1.0 - theta * nu - 0.5 * sigma_vg**2 * nu) / nu
        
        drift = 1j * v_j * (r - omega) * T
        vg_cf = -T / nu * np.log(1.0 - 1j * v_j * theta * nu + 0.5 * v_j**2 * sigma_vg**2 * nu)
        phi = np.exp(drift + vg_cf)
        
        psi = phi * np.exp(-r * T) / ((alpha + 1j * v_j) * (alpha + 1 + 1j * v_j))
        
        w = np.ones(N) * eta
        w[0] = 0.5 * eta
        x = np.exp(1j * b * v_j) * w * psi
        y = np.fft.fft(x)
        
        call_fft = np.real(np.exp(-alpha * k_u) * y / np.pi)
        log_k = np.log(K / S0)
        
        return np.interp(log_k, k_u, call_fft) * S0


class MultiStrikeCalibrator:
    """
    Global calibration engine for VG parameters
    Fits to entire volatility smile across strikes
    """
    
    def __init__(
        self,
        S0: float,
        r: float,
        option_quotes: List[OptionQuote],
        pricing_method: str = 'mc'
    ):
        self.S0 = S0
        self.r = r
        self.quotes = option_quotes
        self.pricing_method = pricing_method
        
        if pricing_method == 'mc':
            self.pricer = VarianceGammaMonteCarlo
        elif pricing_method == 'fft':
            self.pricer = VarianceGammaFFT
        else:
            raise ValueError(f"Unknown pricing method: {pricing_method}")
        
        self.calibration_result = None
    
    def _objective_function(self, params: np.ndarray) -> float:
        """
        Weighted squared error objective with regularization
        """
        theta, sigma_vg, nu = params
        
        if nu <= 0.01 or sigma_vg <= 0.01:
            return 1e10
        
        mg_condition = theta * nu + 0.5 * sigma_vg**2 * nu
        if mg_condition >= 0.95:
            return 1e10
        
        total_error = 0.0
        total_weight = 0.0
        
        for quote in self.quotes:
            try:
                model_price = self.pricer.call_price(
                    self.S0, quote.strike, quote.time_to_expiry, self.r,
                    theta, sigma_vg, nu
                )
                
                relative_error = (model_price - quote.market_price) / quote.market_price
                squared_error = relative_error**2
                
                weight = quote.weight
                total_error += weight * squared_error
                total_weight += weight
                
            except:
                return 1e10
        
        if total_weight > 0:
            weighted_rmse = np.sqrt(total_error / total_weight)
        else:
            weighted_rmse = 1e10
        
        reg_penalty = 0.01 * (
            theta**2 + 
            (sigma_vg - 0.2)**2 + 
            (nu - 0.3)**2
        )
        
        return weighted_rmse + reg_penalty
    
    def calibrate(
        self,
        method: str = 'global',
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict:
        """
        Execute calibration with specified method
        
        Args:
            method: 'global' for differential evolution, 'local' for L-BFGS-B
            bounds: Parameter bounds [(theta_min, theta_max), ...]
        
        Returns:
            Calibration results dictionary
        """
        if bounds is None:
            bounds = [(-0.4, 0.4), (0.05, 0.6), (0.05, 0.9)]
        
        if method == 'global':
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                workers=1
            )
        
        elif method == 'local':
            initial_guesses = [
                [0.0, 0.2, 0.3],
                [-0.1, 0.15, 0.25],
                [0.1, 0.25, 0.4]
            ]
            
            best_result = None
            best_error = np.inf
            
            for x0 in initial_guesses:
                try:
                    res = minimize(
                        self._objective_function,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds
                    )
                    
                    if res.fun < best_error:
                        best_result = res
                        best_error = res.fun
                except:
                    continue
            
            result = best_result
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if result is None or not hasattr(result, 'x'):
            raise RuntimeError("Calibration failed to converge")
        
        theta_opt, sigma_vg_opt, nu_opt = result.x
        
        strike_errors = []
        for quote in self.quotes:
            model_price = self.pricer.call_price(
                self.S0, quote.strike, quote.time_to_expiry, self.r,
                theta_opt, sigma_vg_opt, nu_opt
            )
            error_pct = abs(model_price - quote.market_price) / quote.market_price * 100
            strike_errors.append({
                'strike': quote.strike,
                'market_price': quote.market_price,
                'model_price': model_price,
                'error_pct': error_pct
            })
        
        self.calibration_result = {
            'theta': theta_opt,
            'sigma_vg': sigma_vg_opt,
            'nu': nu_opt,
            'objective_value': result.fun,
            'success': result.success if hasattr(result, 'success') else True,
            'strike_errors': strike_errors,
            'rmse': np.sqrt(np.mean([e['error_pct']**2 for e in strike_errors])),
            'mean_error': np.mean([e['error_pct'] for e in strike_errors]),
            'max_error': np.max([e['error_pct'] for e in strike_errors])
        }
        
        return self.calibration_result
    
    def get_implied_parameters(self) -> Tuple[float, float, float]:
        """Return calibrated VG parameters"""
        if self.calibration_result is None:
            raise RuntimeError("Must call calibrate() first")
        
        return (
            self.calibration_result['theta'],
            self.calibration_result['sigma_vg'],
            self.calibration_result['nu']
        )


def calibrate_single_option(
    S0: float, K: float, T: float, r: float,
    market_price: float, implied_vol: float
) -> Tuple[float, float, float]:
    """
    Legacy interface: calibrate to single option
    """
    quote = OptionQuote(
        strike=K,
        market_price=market_price,
        implied_vol=implied_vol,
        time_to_expiry=T,
        volume=100,
        open_interest=100,
        bid=market_price * 0.98,
        ask=market_price * 1.02
    )
    
    calibrator = MultiStrikeCalibrator(S0, r, [quote], pricing_method='mc')
    result = calibrator.calibrate(method='local')
    
    return result['theta'], result['sigma_vg'], result['nu']