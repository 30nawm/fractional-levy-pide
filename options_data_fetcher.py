# options_data_fetcher.py
# Production data acquisition with relaxed but sensible filters

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Relaxed filters for real market conditions
FILTERS = {
    'min_days_to_expiry': 7,
    'max_days_to_expiry': 180,
    'min_volume': 5,              # Reduced from 10
    'max_bid_ask_spread_pct': 50, # Increased from 10
    'min_implied_vol': 0.01,      # Reduced from 0.05
    'max_implied_vol': 5.0,       # Increased from 2.0
    'moneyness_range': (0.7, 1.3), # Wider range
    'min_price': 0.01             # Reduced minimum
}

def get_option_data(symbol):
    """
    Get best available option data with realistic filters
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return None
        
        S0 = float(hist['Close'].iloc[-1])
        expirations = ticker.options
        
        if not expirations:
            return None
        
        # Try each expiration
        for exp_date in expirations[:15]:
            try:
                chain = ticker.option_chain(exp_date)
                calls = chain.calls
                
                if calls.empty:
                    continue
                
                # Calculate metrics
                calls = calls.copy()
                calls['moneyness'] = calls['strike'] / S0
                
                # Apply filters
                filtered = calls[
                    (calls['volume'] >= FILTERS['min_volume']) &
                    (calls['lastPrice'] >= FILTERS['min_price']) &
                    (calls['impliedVolatility'] >= FILTERS['min_implied_vol']) &
                    (calls['impliedVolatility'] <= FILTERS['max_implied_vol']) &
                    (calls['moneyness'] >= FILTERS['moneyness_range'][0]) &
                    (calls['moneyness'] <= FILTERS['moneyness_range'][1])
                ]
                
                if filtered.empty:
                    continue
                
                # Calculate spread where possible
                filtered['spread_pct'] = 999.0
                mask = (filtered['bid'] > 0) & (filtered['ask'] > filtered['bid'])
                filtered.loc[mask, 'spread_pct'] = (filtered.loc[mask, 'ask'] - filtered.loc[mask, 'bid']) / filtered.loc[mask, 'lastPrice'] * 100
                
                # Filter by spread
                filtered = filtered[filtered['spread_pct'] <= FILTERS['max_bid_ask_spread_pct']]
                
                if filtered.empty:
                    continue
                
                # Days to expiration
                exp_dt = pd.to_datetime(exp_date)
                days_to_exp = (exp_dt - pd.Timestamp.now()).days
                
                if days_to_exp < FILTERS['min_days_to_expiry'] or days_to_exp > FILTERS['max_days_to_expiry']:
                    continue
                
                # Select ATM
                atm = filtered.iloc[(filtered['moneyness'] - 1.0).abs().argsort()[:1]]
                
                if atm.empty:
                    continue
                
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
                
            except:
                continue
        
        return None
        
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
        return None