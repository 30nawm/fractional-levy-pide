"""
Professional Data Infrastructure for Options and Futures Markets
Implements robust data fetching, validation, and quality control
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AssetClass(Enum):
    EQUITY_INDEX = "equity_index"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    RATE = "rate"
    VOLATILITY = "volatility"


@dataclass
class FuturesUniverse:
    """Comprehensive futures portfolio across asset classes"""
    
    equity_indices: List[str] = field(default_factory=lambda: [
        'ES=F',   # S&P 500 E-mini
        'NQ=F',   # Nasdaq-100 E-mini
        'YM=F',   # Dow Jones E-mini
        'RTY=F'   # Russell 2000 E-mini
    ])
    
    commodities: List[str] = field(default_factory=lambda: [
        'GC=F',   # Gold
        'SI=F',   # Silver
        'CL=F',   # Crude Oil WTI
        'NG=F',   # Natural Gas
        'ZC=F',   # Corn
        'ZS=F'    # Soybeans
    ])
    
    currencies: List[str] = field(default_factory=lambda: [
        '6E=F',   # Euro FX
        '6J=F',   # Japanese Yen
        '6B=F',   # British Pound
        '6A=F'    # Australian Dollar
    ])
    
    rates: List[str] = field(default_factory=lambda: [
        'ZT=F',   # 2-Year Treasury Note
        'ZF=F',   # 5-Year Treasury Note
        'ZN=F',   # 10-Year Treasury Note
        'ZB=F'    # 30-Year Treasury Bond
    ])
    
    volatility: List[str] = field(default_factory=lambda: [
        'VIX'     # VIX Index (using ETF derivatives)
    ])
    
    def get_all_symbols(self) -> List[Tuple[str, AssetClass]]:
        """Return all symbols with their asset class"""
        symbols = []
        symbols.extend([(s, AssetClass.EQUITY_INDEX) for s in self.equity_indices])
        symbols.extend([(s, AssetClass.COMMODITY) for s in self.commodities])
        symbols.extend([(s, AssetClass.CURRENCY) for s in self.currencies])
        symbols.extend([(s, AssetClass.RATE) for s in self.rates])
        symbols.extend([(s, AssetClass.VOLATILITY) for s in self.volatility])
        return symbols


@dataclass
class QualityMetrics:
    """Data quality assessment metrics"""
    
    min_volume: int = 10
    max_bid_ask_spread_pct: float = 20.0
    min_implied_vol: float = 0.05
    max_implied_vol: float = 2.0
    moneyness_range: Tuple[float, float] = (0.8, 1.2)
    min_price: float = 0.10
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 180
    
    def validate_option(self, option_data: Dict) -> Tuple[bool, List[str]]:
        """Validate option meets quality standards"""
        issues = []
        
        if option_data.get('volume', 0) < self.min_volume:
            issues.append(f"Low volume: {option_data.get('volume', 0)}")
        
        if option_data.get('lastPrice', 0) < self.min_price:
            issues.append(f"Price too low: {option_data.get('lastPrice', 0)}")
        
        iv = option_data.get('impliedVolatility', 0)
        if not (self.min_implied_vol <= iv <= self.max_implied_vol):
            issues.append(f"IV out of range: {iv:.3f}")
        
        return len(issues) == 0, issues


@dataclass
class OptionData:
    """Standardized option data structure"""
    symbol: str
    underlying_price: float
    strike: float
    expiration: str
    market_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_vol: float
    time_to_expiry: float
    days_to_expiry: int
    option_type: str = 'call'
    
    @property
    def moneyness(self) -> float:
        return self.strike / self.underlying_price
    
    @property
    def mid_price(self) -> float:
        return 0.5 * (self.bid + self.ask)
    
    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return (self.ask - self.bid) / self.mid_price * 100
        return 999.0


class DataProvider:
    """Professional data provider with quality control"""
    
    def __init__(self, quality_metrics: Optional[QualityMetrics] = None):
        self.quality = quality_metrics or QualityMetrics()
        self.cache = {}
    
    def fetch_underlying_data(self, symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """Fetch underlying asset price history"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty or len(hist) < 3:
                return None
            
            return hist
        except:
            return None
    
    def fetch_options_chain(self, symbol: str) -> Optional[Dict]:
        """Fetch complete options chain with quality filtering"""
        try:
            ticker = yf.Ticker(symbol)
            hist = self.fetch_underlying_data(symbol)
            
            if hist is None:
                return None
            
            S0 = float(hist['Close'].iloc[-1])
            expirations = ticker.options
            
            if not expirations:
                return None
            
            chains = {}
            
            for exp_date in expirations[:8]:
                try:
                    chain = ticker.option_chain(exp_date)
                    calls = chain.calls.copy()
                    
                    if calls.empty:
                        continue
                    
                    calls['moneyness'] = calls['strike'] / S0
                    calls['spread_pct'] = 999.0
                    
                    mask = (calls['bid'] > 0) & (calls['ask'] > calls['bid'])
                    calls.loc[mask, 'spread_pct'] = (
                        (calls.loc[mask, 'ask'] - calls.loc[mask, 'bid']) / 
                        calls.loc[mask, 'lastPrice'] * 100
                    )
                    
                    exp_dt = pd.to_datetime(exp_date)
                    days_to_exp = (exp_dt - pd.Timestamp.now()).days
                    
                    if not (self.quality.min_days_to_expiry <= days_to_exp <= self.quality.max_days_to_expiry):
                        continue
                    
                    filtered = calls[
                        (calls['volume'] >= self.quality.min_volume) &
                        (calls['lastPrice'] >= self.quality.min_price) &
                        (calls['impliedVolatility'] >= self.quality.min_implied_vol) &
                        (calls['impliedVolatility'] <= self.quality.max_implied_vol) &
                        (calls['moneyness'] >= self.quality.moneyness_range[0]) &
                        (calls['moneyness'] <= self.quality.moneyness_range[1]) &
                        (calls['spread_pct'] <= self.quality.max_bid_ask_spread_pct)
                    ]
                    
                    if not filtered.empty:
                        chains[exp_date] = {
                            'calls': filtered,
                            'days_to_expiry': days_to_exp,
                            'underlying_price': S0
                        }
                
                except:
                    continue
            
            if not chains:
                return None
            
            return {
                'symbol': symbol,
                'underlying_price': S0,
                'chains': chains
            }
        
        except:
            return None
    
    def select_atm_option(self, symbol: str) -> Optional[OptionData]:
        """Select best ATM option for analysis"""
        chain_data = self.fetch_options_chain(symbol)
        
        if not chain_data:
            return None
        
        best_option = None
        best_score = -1
        
        for exp_date, chain_info in chain_data['chains'].items():
            calls = chain_info['calls']
            days = chain_info['days_to_expiry']
            S0 = chain_info['underlying_price']
            
            calls['atm_distance'] = abs(calls['moneyness'] - 1.0)
            calls['score'] = (
                50 * (1 - calls['atm_distance']) +
                np.minimum(20, calls['volume'] / 10) +
                np.maximum(0, 20 - calls['spread_pct'])
            )
            
            if 20 <= days <= 120:
                calls['score'] += 30
            
            best_in_chain = calls.nsmallest(1, 'atm_distance')
            
            if not best_in_chain.empty:
                row = best_in_chain.iloc[0]
                
                if row['score'] > best_score:
                    best_score = row['score']
                    best_option = OptionData(
                        symbol=symbol,
                        underlying_price=S0,
                        strike=float(row['strike']),
                        expiration=exp_date,
                        market_price=float(row['lastPrice']),
                        bid=float(row['bid']),
                        ask=float(row['ask']),
                        volume=int(row['volume']),
                        open_interest=int(row['openInterest']),
                        implied_vol=float(row['impliedVolatility']),
                        time_to_expiry=days / 365.25,
                        days_to_expiry=days
                    )
        
        return best_option
    
    def get_multi_strike_options(
        self, 
        symbol: str, 
        n_strikes: int = 5
    ) -> Optional[List[OptionData]]:
        """Get multiple strikes for smile calibration"""
        chain_data = self.fetch_options_chain(symbol)
        
        if not chain_data:
            return None
        
        best_expiry = None
        max_options = 0
        
        for exp_date, chain_info in chain_data['chains'].items():
            n_options = len(chain_info['calls'])
            if n_options > max_options:
                max_options = n_options
                best_expiry = exp_date
        
        if best_expiry is None:
            return None
        
        chain = chain_data['chains'][best_expiry]
        calls = chain['calls'].copy()
        S0 = chain['underlying_price']
        days = chain['days_to_expiry']
        
        calls['atm_distance'] = abs(calls['moneyness'] - 1.0)
        calls = calls.sort_values('atm_distance')
        
        selected_options = []
        
        for _, row in calls.head(n_strikes).iterrows():
            option = OptionData(
                symbol=symbol,
                underlying_price=S0,
                strike=float(row['strike']),
                expiration=best_expiry,
                market_price=float(row['lastPrice']),
                bid=float(row['bid']),
                ask=float(row['ask']),
                volume=int(row['volume']),
                open_interest=int(row['openInterest']),
                implied_vol=float(row['impliedVolatility']),
                time_to_expiry=days / 365.25,
                days_to_expiry=days
            )
            selected_options.append(option)
        
        return selected_options if selected_options else None
    
    def calculate_historical_volatility(
        self, 
        symbol: str, 
        window_days: int = 252
    ) -> Tuple[float, np.ndarray]:
        """Calculate realized historical volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{window_days + 50}d")
            
            if len(hist) < 20:
                return 0.2, np.array([])
            
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            
            if len(returns) < 10:
                return 0.2, np.array([])
            
            hist_vol = returns.std() * np.sqrt(252)
            
            return float(hist_vol), returns.values
        
        except:
            return 0.2, np.array([])


class FuturesDataProvider(DataProvider):
    """Specialized provider for futures markets"""
    
    def __init__(self):
        super().__init__()
        self.universe = FuturesUniverse()
    
    def get_active_futures(self) -> List[Tuple[str, AssetClass, bool]]:
        """Return list of futures with activity status"""
        results = []
        
        for symbol, asset_class in self.universe.get_all_symbols():
            hist = self.fetch_underlying_data(symbol, period="5d")
            
            if hist is not None and len(hist) >= 3:
                avg_volume = hist['Volume'].mean()
                is_active = avg_volume > 1000
                results.append((symbol, asset_class, is_active))
        
        return results
    
    def fetch_futures_option(self, symbol: str) -> Optional[OptionData]:
        """Fetch option on futures contract"""
        return self.select_atm_option(symbol)