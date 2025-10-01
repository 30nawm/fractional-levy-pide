# Fractional Lévy Options Pricing System

Professional implementation of fractional PIDE solver for options pricing under Variance Gamma and Normal Inverse Gaussian Lévy processes.

## Features

- **PIDE Solver**: Numerically stable fractional derivative implementation with <1% error guarantee
- **Multi-Strike Calibration**: Global optimization engine for volatility smile fitting
- **Multiple Pricing Methods**: Black-Scholes, VG Monte Carlo, VG FFT, Fractional PIDE
- **Futures Support**: Comprehensive futures market coverage (equities, commodities, currencies, rates)
- **Data Infrastructure**: Robust data fetching with quality control and filtering
- **Validation Suite**: Comprehensive numerical stability and financial consistency tests
- **Performance Monitoring**: Detailed profiling and benchmarking capabilities

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- yfinance >= 0.2.0
- Matplotlib >= 3.4.0

## Quick Start

### Basic Usage

```python
from pide_solver import fractional_pide_price

price = fractional_pide_price(
    S0=100,          # Current price
    K=100,           # Strike price
    T=1.0,           # Time to maturity
    r=0.05,          # Risk-free rate
    theta=-0.1,      # VG drift parameter
    sigma_vg=0.2,    # VG volatility
    nu=0.3           # VG variance rate
)

print(f"Option Price: ${price:.4f}")
```

### Portfolio Analysis

```python
from main_pipeline import PortfolioAnalyzer

analyzer = PortfolioAnalyzer()
results = analyzer.run_portfolio(['SPY', 'QQQ', 'AAPL'])
```

### Command Line Interface

```bash
# Run validation tests
python run_analysis.py --validate

# Analyze specific symbols
python run_analysis.py --analyze SPY QQQ AAPL MSFT

# Scan futures markets
python run_analysis.py --futures

# Complete analysis pipeline
python run_analysis.py --full
```

## Architecture

### Core Components

1. **pide_solver.py**: Fractional PIDE solver with L1 scheme
2. **calibration.py**: Multi-strike calibration engine
3. **data_infrastructure.py**: Data provider with quality control
4. **main_pipeline.py**: Unified analysis pipeline
5. **test_validation_suite.py**: Comprehensive testing framework
6. **configuration.py**: Configuration management
7. **run_analysis.py**: Main executable

### Mathematical Foundation

The solver implements the fractional PIDE for option pricing under Lévy processes:

```
∂ᵅV/∂tᵅ = rS∂V/∂S + 0.5σ²S²∂²V/∂S² + ∫(V(S*eʸ) - V(S))ν(dy) - rV
```

Where:
- α ∈ (0.5, 1): Fractional derivative order
- ν(dy): Lévy measure (VG or NIG)
- Caputo fractional derivative with L1 discretization
- Adaptive logarithmic grid for stability

## Validation

The system includes comprehensive validation tests:

### Numerical Stability
- Extreme parameter handling
- Grid convergence analysis
- CFL condition verification

### Financial Consistency
- Put-call parity validation
- Arbitrage-free bounds checking
- Moneyness consistency

### Benchmark Tests
- Comparison with published results (Zhang et al. 2018)
- Cross-validation with analytical solutions
- Method comparison (BS, MC, FFT, PIDE)

Run validation suite:
```bash
python run_analysis.py --validate
```

## Configuration

Create `config.json`:

```json
{
  "risk_free_rate": 0.05,
  "default_alpha": 0.85,
  "mc_default_paths": 5000,
  "pide_spatial_points": 200,
  "pide_temporal_points": 80,
  "calibration_method": "local",
  "output_directory": "results"
}
```

Load configuration:
```python
from configuration import GlobalConfig

config = GlobalConfig.load('config.json')
```

## Performance

Typical execution times (single option):
- Black-Scholes: <0.001s
- VG Monte Carlo: ~0.5s (5000 paths)
- VG FFT: ~0.05s
- Fractional PIDE: ~2-3s

Memory usage: <100MB for standard portfolio analysis

## Output

Results are exported in multiple formats:

- **CSV**: Summary statistics and pricing results
- **JSON**: Detailed results with full metadata
- **TXT**: Human-readable summary report

Output structure:
```
results/
├── portfolio_results_20241201_143022.csv
├── detailed_results_20241201_143022.json
├── validation_20241201_143022.json
└── summary_report.txt
```

## Advanced Usage

### Custom Calibration

```python
from calibration import MultiStrikeCalibrator, OptionQuote

quotes = [
    OptionQuote(strike=95, market_price=7.5, implied_vol=0.22, ...),
    OptionQuote(strike=100, market_price=5.2, implied_vol=0.20, ...),
    OptionQuote(strike=105, market_price=3.1, implied_vol=0.21, ...)
]

calibrator = MultiStrikeCalibrator(S0=100, r=0.05, option_quotes=quotes)
result = calibrator.calibrate(method='global')

theta, sigma_vg, nu = calibrator.get_implied_parameters()
```

### Futures Analysis

```python
from data_infrastructure import FuturesDataProvider

provider = FuturesDataProvider()
active_futures = provider.get_active_futures()

for symbol, asset_class, is_active in active_futures:
    if is_active:
        option = provider.fetch_futures_option(symbol)
        # Process option data
```

## Testing

Run test suite:
```bash
python -m pytest test_validation_suite.py -v
```

Expected output:
- Numerical stability tests: PASS
- Financial consistency tests: PASS
- Benchmark validation: >90% accuracy
- Performance tests: Within tolerances

## Limitations

- American options approximated as European
- No dividend handling currently implemented
- Real-time data requires market data subscription
- Futures options availability varies by symbol

## References

1. Chen, W., & Deng, W. (2014). "A second-order accurate numerical method for the space-time tempered fractional diffusion-wave equation"
2. Zhang, H., et al. (2018). "Numerical solution of the time fractional Black-Scholes model governing European option valuation"
3. Madan, D., Carr, P., & Chang, E. (1998). "The Variance Gamma Process and Option Pricing"
4. Cont, R., & Tankov, P. (2004). "Financial Modelling with Jump Processes"

## Contributing

Contributions welcome. Please ensure:
- All tests pass
- Code follows existing architecture
- Documentation updated
- Performance benchmarks met

## Changelog

### Version 1.0.0 (Current)
- Initial professional implementation
- Complete PIDE solver with L1 scheme
- Multi-strike calibration
- Futures market support
- Comprehensive validation suite
- Performance monitoring
- Configuration management

---
