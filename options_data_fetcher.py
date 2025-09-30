# options_data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Conservative filters for reliable data
FILTERS = {
    'min_days_to_expiry': 7,
    'max_days_to_expiration': 180,
    'min_volume': 10,
    'max_bid_ask_spread_pct': 20,
    'min_implied_vol': 0.05,
    'max_implied_vol': 2.0,
    'moneyness_range': (0.8, 1.2),
    'min_price': 0.10
}

def get_option_data(symbol):
    """
    Get option data with robust error handling
    Returns format compatible with existing main.py structure
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty or len(hist) < 3:
            return None
        
        S0 = float(hist['Close'].iloc[-1])
        expirations = ticker.options
        
        if not expirations:
            return None
        
        for exp_date in expirations[:8]:
            try:
                chain = ticker.option_chain(exp_date)
                calls = chain.calls
                
                if calls.empty:
                    continue
                
                # Basic filtering
                calls = calls.copy()
                calls['moneyness'] = calls['strike'] / S0
                
                filtered = calls[
                    (calls['volume'] >= FILTERS['min_volume']) &
                    (calls['lastPrice'] >= FILTERS['min_price']) &
                    (calls['impliedVolatility'] >= FILTERS['min_implied_vol']) &
                    (calls['impliedVolatility'] <= FILTERS['max_implied_vol']) &
                    (calls['moneyness'] >= FILTERS['moneyness_range'][0]) &
                    (calls['moneyness'] <= FILTERS['moneyness_range'][1])
                ].copy()
                
                if filtered.empty:
                    continue
                
                # Calculate bid-ask spread
                filtered['spread_pct'] = 999.0
                mask = (filtered['bid'] > 0) & (filtered['ask'] > filtered['bid'])
                filtered.loc[mask, 'spread_pct'] = (
                    (filtered.loc[mask, 'ask'] - filtered.loc[mask, 'bid']) / 
                    filtered.loc[mask, 'lastPrice'] * 100
                )
                
                filtered = filtered[filtered['spread_pct'] <= FILTERS['max_bid_ask_spread_pct']]
                
                if filtered.empty:
                    continue
                
                # Days to expiration
                exp_dt = pd.to_datetime(exp_date)
                days_to_exp = (exp_dt - pd.Timestamp.now()).days
                
                if days_to_exp < FILTERS['min_days_to_expiry'] or days_to_exp > FILTERS['max_days_to_expiration']:
                    continue
                
                # Select ATM option
                filtered['atm_distance'] = abs(filtered['moneyness'] - 1.0)
                atm = filtered.nsmallest(1, 'atm_distance')
                
                if atm.empty:
                    continue
                
                # Return format compatible with main.py expectations
                return {
                    'symbol': symbol,
                    'stock_price': S0,
                    'strike': float(atm['strike'].iloc[0]),
                    'market_price': float(atm['lastPrice'].iloc[0]),
                    'bid': float(atm['bid'].iloc[0]),
                    'ask': float(atm['ask'].iloc[0]),
                    'volume': int(atm['volume'].iloc[0]),
                    'open_interest': int(atm['openInterest'].iloc[0]),
                    'implied_vol': float(atm['impliedVolatility'].iloc[0]),
                    'time_to_expiration': days_to_exp / 365.25,
                    'days_to_expiration': days_to_exp,
                    'expiration': exp_date
                }
                
            except Exception:
                continue
        
        return None
        
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
        return None

def get_real_options_data(symbol):
    """Alias for compatibility with multi_symbol_analysis.py"""
    return get_option_data(symbol)

def get_historical_volatility(symbol, period_days=252):
    """
    Calculate historical volatility for a symbol
    Compatible with multi_symbol_analysis.py requirements
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{period_days+50}d")  # Extra days for returns calc
        
        if len(hist) < 20:
            return 0.2, np.array([])  # Default fallback
        
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        
        if len(returns) < 10:
            return 0.2, np.array([])
        
        hist_vol = returns.std() * np.sqrt(252)
        return float(hist_vol), returns.values
        
    except Exception:
        return 0.2, np.array([])  # Default fallback