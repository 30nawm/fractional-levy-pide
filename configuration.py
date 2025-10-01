"""
Professional Configuration Management System
Centralized parameter validation and environment setup
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging


@dataclass
class GlobalConfig:
    """Global system configuration"""
    
    risk_free_rate: float = 0.05
    default_alpha: float = 0.85
    
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 180
    
    min_option_volume: int = 10
    max_bid_ask_spread_pct: float = 20.0
    
    mc_default_paths: int = 5000
    fft_grid_size: int = 4096
    pide_spatial_points: int = 200
    pide_temporal_points: int = 80
    
    calibration_method: str = 'local'
    pricing_methods: list = None
    
    output_directory: str = 'results'
    enable_caching: bool = True
    log_level: str = 'INFO'
    
    def __post_init__(self):
        if self.pricing_methods is None:
            self.pricing_methods = ['BS', 'VG_MC', 'VG_FFT', 'FRACTIONAL_PIDE']
    
    def validate(self) -> tuple[bool, list]:
        """Validate configuration parameters"""
        issues = []
        
        if not (0.0 <= self.risk_free_rate <= 0.2):
            issues.append(f"Risk-free rate {self.risk_free_rate} out of range [0, 0.2]")
        
        if not (0.5 < self.default_alpha < 1.0):
            issues.append(f"Alpha {self.default_alpha} out of range (0.5, 1.0)")
        
        if self.min_days_to_expiry >= self.max_days_to_expiry:
            issues.append("min_days_to_expiry must be < max_days_to_expiry")
        
        if self.mc_default_paths < 1000:
            issues.append("MC paths should be >= 1000 for convergence")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GlobalConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath: str = 'config.json'):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str = 'config.json') -> 'GlobalConfig':
        """Load configuration from file"""
        if not Path(filepath).exists():
            return cls()
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class ConfigurationManager:
    """Centralized configuration management"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = GlobalConfig()
            self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on config"""
        logging.basicConfig(
            level=getattr(logging, self._config.log_level),
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_config(self) -> GlobalConfig:
        """Get current configuration"""
        return self._config
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        is_valid, issues = self._config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(issues)}")
    
    def reset_to_defaults(self):
        """Reset to default configuration"""
        self._config = GlobalConfig()
        self._setup_logging()


def get_config() -> GlobalConfig:
    """Convenience function to get global config"""
    manager = ConfigurationManager()
    return manager.get_config()


class PerformanceMonitor:
    """Track and report system performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'calibration_times': [],
            'pricing_times': {},
            'memory_usage': [],
            'error_rates': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def record_calibration(self, duration: float, success: bool):
        """Record calibration performance"""
        self.metrics['calibration_times'].append(duration)
        if success:
            self.metrics.setdefault('successful_calibrations', 0)
            self.metrics['successful_calibrations'] += 1
    
    def record_pricing(self, method: str, duration: float, error: Optional[float] = None):
        """Record pricing performance"""
        if method not in self.metrics['pricing_times']:
            self.metrics['pricing_times'][method] = []
        
        self.metrics['pricing_times'][method].append(duration)
        
        if error is not None:
            if method not in self.metrics['error_rates']:
                self.metrics['error_rates'][method] = []
            self.metrics['error_rates'][method].append(error)
    
    def record_cache_access(self, hit: bool):
        """Record cache performance"""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
    
    def get_summary(self) -> Dict:
        """Generate performance summary"""
        import numpy as np
        
        summary = {}
        
        if self.metrics['calibration_times']:
            summary['calibration'] = {
                'mean_time': np.mean(self.metrics['calibration_times']),
                'median_time': np.median(self.metrics['calibration_times']),
                'total_runs': len(self.metrics['calibration_times']),
                'success_rate': self.metrics.get('successful_calibrations', 0) / len(self.metrics['calibration_times'])
            }
        
        summary['pricing'] = {}
        for method, times in self.metrics['pricing_times'].items():
            summary['pricing'][method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'total_runs': len(times)
            }
            
            if method in self.metrics['error_rates']:
                errors = self.metrics['error_rates'][method]
                summary['pricing'][method]['mean_error'] = np.mean(errors)
                summary['pricing'][method]['median_error'] = np.median(errors)
        
        total_cache = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache > 0:
            summary['cache'] = {
                'hit_rate': self.metrics['cache_hits'] / total_cache,
                'hits': self.metrics['cache_hits'],
                'misses': self.metrics['cache_misses']
            }
        
        return summary
    
    def print_report(self):
        """Print formatted performance report"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("PERFORMANCE REPORT")
        print("="*80 + "\n")
        
        if 'calibration' in summary:
            print("Calibration Performance:")
            print(f"  Mean Time: {summary['calibration']['mean_time']:.3f}s")
            print(f"  Success Rate: {summary['calibration']['success_rate']:.1%}")
            print(f"  Total Runs: {summary['calibration']['total_runs']}")
        
        if 'pricing' in summary:
            print("\nPricing Performance:")
            for method, stats in summary['pricing'].items():
                print(f"\n  {method}:")
                print(f"    Mean Time: {stats['mean_time']:.4f}s")
                if 'mean_error' in stats:
                    print(f"    Mean Error: {stats['mean_error']:.2f}%")
                print(f"    Runs: {stats['total_runs']}")
        
        if 'cache' in summary:
            print(f"\nCache Performance:")
            print(f"  Hit Rate: {summary['cache']['hit_rate']:.1%}")
            print(f"  Hits/Misses: {summary['cache']['hits']}/{summary['cache']['misses']}")
        
        print("\n" + "="*80 + "\n")


class ResultsExporter:
    """Handle export of results in multiple formats"""
    
    def __init__(self, output_dir: str = None):
        config = get_config()
        self.output_dir = Path(output_dir or config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_json(self, data: Dict, filename: str):
        """Export to JSON format"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def export_csv(self, data: Any, filename: str):
        """Export to CSV format"""
        import pandas as pd
        
        filepath = self.output_dir / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_summary_report(self, results: Dict, filename: str = 'summary_report.txt'):
        """Export text summary report"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FRACTIONAL LÉVY OPTIONS PRICING - ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {results.get('timestamp', 'N/A')}\n")
            f.write(f"Total Symbols: {len(results.get('symbols', []))}\n")
            f.write(f"Successful: {results.get('successful', 0)}\n")
            f.write(f"Failed: {results.get('failed', 0)}\n\n")
            
            f.write("="*80 + "\n")
        
        return filepath


if __name__ == "__main__":
    config = get_config()
    is_valid, issues = config.validate()
    
    if is_valid:
        print("Configuration valid")
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")