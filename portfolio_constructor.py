# portfolio_constructor.py
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
import time

class PortfolioConstructor:
    def __init__(self):
        self.categories = {
            'indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F', 'ZS=F'],
            'etfs': ['GLD', 'SLV', 'USO', 'TLT', 'HYG', 'LQD'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
        }
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol has tradable options"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if hist.empty or len(hist) < 10:
                return False
            
            # Check options availability
            expirations = ticker.options
            if not expirations:
                return False
            
            # Basic liquidity check
            current_price = hist['Close'].iloc[-1]
            volume_avg = hist['Volume'].mean()
            
            return current_price > 0 and volume_avg > 10000
            
        except Exception:
            return False
    
    def build_portfolio(self) -> Dict:
        """Build portfolio with validated symbols"""
        portfolio = {}
        
        for category, symbols in self.categories.items():
            validated_symbols = []
            
            for symbol in symbols:
                if self.validate_symbol(symbol):
                    validated_symbols.append(symbol)
                    time.sleep(0.2)  # Rate limiting
                else:
                    print(f"⚠️  {symbol} failed validation")
            
            portfolio[category] = validated_symbols
            print(f"✅ {category}: {validated_symbols}")
        
        return portfolio

# Global portfolio instance
portfolio_builder = PortfolioConstructor()
PORTFOLIO = portfolio_builder.build_portfolio()