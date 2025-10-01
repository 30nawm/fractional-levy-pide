"""
Main Executable: Production-Grade Fractional Lévy Options Pricing System
Entry point for complete portfolio analysis with all features integrated
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

from configuration import get_config, ConfigurationManager, PerformanceMonitor, ResultsExporter
from main_pipeline import PortfolioAnalyzer
from test_validation_suite import ValidationSuite
from data_infrastructure import FuturesUniverse


logger = logging.getLogger(__name__)


class Application:
    """Main application controller"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_config()
        self.performance = PerformanceMonitor()
        self.exporter = ResultsExporter()
    
    def run_validation(self) -> bool:
        """Execute validation suite"""
        print("\nRunning comprehensive validation suite...")
        
        suite = ValidationSuite()
        results = suite.run_all_tests()
        
        validation_file = self.exporter.export_json(
            results, 
            f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        logger.info(f"Validation results saved: {validation_file}")
        
        return True
    
    def run_portfolio_analysis(self, symbols: list) -> bool:
        """Execute portfolio analysis"""
        print(f"\nRunning portfolio analysis on {len(symbols)} symbols...")
        
        analyzer = PortfolioAnalyzer(output_dir=str(self.exporter.output_dir))
        
        try:
            df = analyzer.run_portfolio(symbols)
            
            if df.empty:
                logger.error("No results generated")
                return False
            
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'successful': len(df),
                'failed': len(symbols) - len(df),
                'results_summary': df.describe().to_dict()
            }
            
            self.exporter.export_summary_report(summary_data)
            
            logger.info(f"Analysis complete: {len(df)} successful")
            return True
        
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return False
    
    def run_futures_scan(self) -> bool:
        """Scan futures markets for active contracts"""
        print("\nScanning futures markets...")
        
        from data_infrastructure import FuturesDataProvider
        
        provider = FuturesDataProvider()
        active_futures = provider.get_active_futures()
        
        results = []
        for symbol, asset_class, is_active in active_futures:
            results.append({
                'symbol': symbol,
                'asset_class': asset_class.value,
                'active': is_active
            })
        
        df_futures = __import__('pandas').DataFrame(results)
        
        filepath = self.exporter.export_csv(
            df_futures,
            f'futures_scan_{datetime.now().strftime("%Y%m%d")}.csv'
        )
        
        print(f"\nActive Futures: {df_futures['active'].sum()}/{len(df_futures)}")
        print(f"Results saved: {filepath}")
        
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fractional Lévy Options Pricing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --validate                    Run validation tests
  %(prog)s --analyze SPY QQQ AAPL       Analyze specific symbols
  %(prog)s --futures                     Scan futures markets
  %(prog)s --full                        Complete analysis pipeline
        """
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation test suite'
    )
    
    parser.add_argument(
        '--analyze',
        nargs='+',
        metavar='SYMBOL',
        help='Analyze specified symbols'
    )
    
    parser.add_argument(
        '--futures',
        action='store_true',
        help='Scan and analyze futures markets'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run complete pipeline (validation + analysis)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    app = Application()
    
    if args.config:
        app.config = app.config_manager.get_config().load(args.config)
    
    app.config_manager.update_config(
        output_directory=args.output_dir,
        log_level=args.log_level
    )
    
    print("="*80)
    print("FRACTIONAL LÉVY OPTIONS PRICING SYSTEM")
    print("Professional Implementation - Production Grade")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80 + "\n")
    
    success = True
    
    if args.validate or args.full:
        success &= app.run_validation()
    
    if args.futures:
        success &= app.run_futures_scan()
    
    if args.analyze:
        success &= app.run_portfolio_analysis(args.analyze)
    
    elif args.full:
        default_symbols = [
            'SPY', 'QQQ', 'IWM', 'DIA',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN',
            'GLD', 'SLV', 'USO'
        ]
        success &= app.run_portfolio_analysis(default_symbols)
    
    if not any([args.validate, args.analyze, args.futures, args.full]):
        print("No action specified. Use --help for options.")
        return 1
    
    app.performance.print_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Status: {'SUCCESS' if success else 'COMPLETED WITH ERRORS'}")
    print(f"Results Directory: {args.output_dir}")
    print("="*80 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
