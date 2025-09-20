# real_options_data.py
# Module for fetching and processing real historical options data
# Supports multiple data sources and 5-year historical analysis

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_cboe_options_data(symbol='SPY'):
    """
    Fetch options data from CBOE (Chicago Board Options Exchange)
    Note: Requires specific API access or web scraping
    """
    try:
        # This would require CBOE API access
        # For now, we'll use yfinance as primary source
        print(f"CBOE API not implemented, falling back to yfinance for {symbol}")
        return None
    except Exception as e:
        print(f"CBOE data fetch failed: {e}")
        return None

def get_polygon_options_data(symbol='SPY', api_key=None):
    """
    Fetch historical options data from Polygon.io
    Requires API key: https://polygon.io/
    """
    if not api_key:
        print("Polygon.io requires API key. Skipping...")
        return None
    
    try:
        import requests
        
        # Get options contracts
        url = f"https://api.polygon.io/v3/reference/options/contracts"
        params = {
            'underlying_ticker': symbol,
            'contract_type': 'call',
            'limit': 20,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'results' in data and data['results']:
            print(f"✅ Found {len(data['results'])} option contracts from Polygon")
            return data['results']
        else:
            print("No results from Polygon API")
            return None
            
    except Exception as e:
        print(f"Polygon data fetch failed: {e}")
        return None

def get_alpha_vantage_options(symbol='SPY', api_key=None):
    """
    Get options data from Alpha Vantage
    Requires API key: https://www.alphavantage.co/support/#api-key
    """
    if not api_key:
        print("Alpha Vantage requires API key. Skipping...")
        return None
        
    try:
        import requests
        
        # Alpha Vantage doesn't have direct options API, but has stock data
        # We'll use this for underlying stock analysis
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            print(f"✅ Retrieved stock data from Alpha Vantage for {symbol}")
            return data
        else:
            print("Invalid response from Alpha Vantage")
            return None
            
    except Exception as e:
        print(f"Alpha Vantage fetch failed: {e}")
        return None

def get_yfinance_options_comprehensive(symbol='SPY', max_expirations=5):
    """
    Comprehensive options data retrieval using yfinance
    Gets multiple expirations and detailed options chain
    """
    try:
        import yfinance as yf
        
        print(f"📊 Fetching comprehensive options data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        
        # Get stock info and current price
        info = ticker.info
        hist = ticker.history(period="5d")
        
        if hist.empty:
            raise ValueError("No stock price data available")
        
        current_price = hist['Close'].iloc[-1]
        
        # Get all available option expirations
        expirations = ticker.options
        if not expirations:
            raise ValueError("No options expirations available")
        
        print(f"📅 Found {len(expirations)} expiration dates")
        
        options_data = {
            'symbol': symbol,
            'current_price': current_price,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'options_chains': {},
            'expirations': expirations
        }
        
        # Process multiple expirations
        processed_exps = 0
        for exp_date in expirations[:max_expirations]:
            try:
                print(f"Processing expiration: {exp_date}")
                
                chain = ticker.option_chain(exp_date)
                calls = chain.calls
                puts = chain.puts
                
                # Filter for liquid options (volume > 0, reasonable bid-ask spread)
                calls_liquid = calls[
                    (calls['volume'] > 0) & 
                    (calls['lastPrice'] > 0) &
                    (calls['bid'] > 0) &
                    (calls['ask'] > calls['bid'])
                ].copy()
                
                puts_liquid = puts[
                    (puts['volume'] > 0) & 
                    (puts['lastPrice'] > 0) &
                    (puts['bid'] > 0) &
                    (puts['ask'] > puts['bid'])
                ].copy()
                
                if not calls_liquid.empty:
                    # Calculate additional metrics
                    calls_liquid['mid_price'] = (calls_liquid['bid'] + calls_liquid['ask']) / 2
                    calls_liquid['spread_pct'] = (calls_liquid['ask'] - calls_liquid['bid']) / calls_liquid['mid_price'] * 100
                    calls_liquid['moneyness'] = calls_liquid['strike'] / current_price
                    calls_liquid['intrinsic_value'] = np.maximum(current_price - calls_liquid['strike'], 0)
                    calls_liquid['time_value'] = calls_liquid['lastPrice'] - calls_liquid['intrinsic_value']
                    
                    # Calculate time to expiration
                    exp_datetime = pd.to_datetime(exp_date)
                    days_to_exp = (exp_datetime - pd.Timestamp.now()).days
                    calls_liquid['days_to_expiration'] = days_to_exp
                    calls_liquid['years_to_expiration'] = days_to_exp / 365.25
                    
                if not puts_liquid.empty:
                    puts_liquid['mid_price'] = (puts_liquid['bid'] + puts_liquid['ask']) / 2
                    puts_liquid['spread_pct'] = (puts_liquid['ask'] - puts_liquid['bid']) / puts_liquid['mid_price'] * 100
                    puts_liquid['moneyness'] = puts_liquid['strike'] / current_price
                    puts_liquid['intrinsic_value'] = np.maximum(puts_liquid['strike'] - current_price, 0)
                    puts_liquid['time_value'] = puts_liquid['lastPrice'] - puts_liquid['intrinsic_value']
                    
                    exp_datetime = pd.to_datetime(exp_date)
                    days_to_exp = (exp_datetime - pd.Timestamp.now()).days
                    puts_liquid['days_to_expiration'] = days_to_exp
                    puts_liquid['years_to_expiration'] = days_to_exp / 365.25
                
                options_data['options_chains'][exp_date] = {
                    'calls': calls_liquid,
                    'puts': puts_liquid,
                    'total_call_volume': calls_liquid['volume'].sum() if not calls_liquid.empty else 0,
                    'total_put_volume': puts_liquid['volume'].sum() if not puts_liquid.empty else 0,
                    'days_to_expiration': days_to_exp
                }
                
                processed_exps += 1
                print(f"   ✅ Calls: {len(calls_liquid)}, Puts: {len(puts_liquid)}")
                
            except Exception as e:
                print(f"   ❌ Failed to process {exp_date}: {e}")
                continue
        
        print(f"✅ Successfully processed {processed_exps} option expirations")
        return options_data
        
    except Exception as e:
        print(f"❌ Comprehensive options data fetch failed: {e}")
        return None

def select_best_option_for_analysis(options_data, prefer_atm=True, min_volume=10):
    """
    Select the best option contract for pricing analysis
    Criteria: liquid, near-ATM, reasonable time to expiration
    """
    if not options_data or 'options_chains' not in options_data:
        return None
    
    current_price = options_data['current_price']
    best_option = None
    best_score = -1
    
    print(f"🎯 Selecting best option for analysis (current price: ${current_price:.2f})...")
    
    for exp_date, chain_data in options_data['options_chains'].items():
        calls = chain_data['calls']
        days_to_exp = chain_data['days_to_expiration']
        
        if calls.empty or days_to_exp < 7:  # Skip very short-term options
            continue
        
        # Filter by volume and other criteria
        candidates = calls[
            (calls['volume'] >= min_volume) &
            (calls['spread_pct'] < 20) &  # Reasonable bid-ask spread
            (calls['impliedVolatility'] > 0.05) &  # Reasonable IV
            (calls['impliedVolatility'] < 2.0) &   # Not crazy high IV
            (calls['moneyness'] > 0.8) &   # Not too far OTM
            (calls['moneyness'] < 1.2)     # Not too far ITM
        ]
        
        if candidates.empty:
            continue
        
        # Score each candidate
        for idx, row in candidates.iterrows():
            score = 0
            
            # Prefer ATM options
            atm_distance = abs(row['moneyness'] - 1.0)
            score += max(0, 50 - atm_distance * 100)  # Max 50 points for ATM
            
            # Prefer reasonable time to expiration (30-90 days ideal)
            if 20 <= days_to_exp <= 120:
                score += 30
            elif 7 <= days_to_exp <= 180:
                score += 15
            
            # Prefer high volume
            score += min(20, row['volume'] / 10)  # Max 20 points for volume
            
            # Prefer tight spreads
            spread_penalty = max(0, row['spread_pct'] - 5)
            score -= spread_penalty
            
            # Prefer reasonable implied volatility
            if 0.1 <= row['impliedVolatility'] <= 0.8:
                score += 10
            
            if score > best_score:
                best_score = score
                best_option = {
                    'symbol': options_data['symbol'],
                    'stock_price': current_price,
                    'strike': row['strike'],
                    'expiration': exp_date,
                    'market_price': row['lastPrice'],
                    'mid_price': row['mid_price'],
                    'bid': row['bid'],
                    'ask': row['ask'],
                    'volume': row['volume'],
                    'open_interest': row['openInterest'],
                    'implied_vol': row['impliedVolatility'],
                    'time_to_expiration': row['years_to_expiration'],
                    'days_to_expiration': days_to_exp,
                    'moneyness': row['moneyness'],
                    'intrinsic_value': row['intrinsic_value'],
                    'time_value': row['time_value'],
                    'option_type': 'call',
                    'style': 'american',  # US equity options are typically American
                    'score': best_score
                }
    
    if best_option:
        print(f"✅ Selected option:")
        print(f"   Strike: ${best_option['strike']:.2f}")
        print(f"   Expiration: {best_option['expiration']}")
        print(f"   Market Price: ${best_option['market_price']:.3f}")
        print(f"   Volume: {best_option['volume']:,}")
        print(f"   Implied Vol: {best_option['implied_vol']:.1%}")
        print(f"   Days to Exp: {best_option['days_to_expiration']}")
        print(f"   Selection Score: {best_option['score']:.1f}")
    else:
        print("❌ No suitable option found for analysis")
    
    return best_option

def get_historical_options_analysis(symbol='SPY', years_back=2):
    """
    Analyze historical options data patterns over specified period
    """
    try:
        import yfinance as yf
        
        print(f"📈 Analyzing {years_back} years of options history for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        
        # Get historical stock prices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        hist_prices = ticker.history(start=start_date, end=end_date)
        
        if hist_prices.empty:
            raise ValueError("No historical price data")
        
        # Calculate various statistics
        returns = np.log(hist_prices['Close'] / hist_prices['Close'].shift(1)).dropna()
        
        analysis = {
            'symbol': symbol,
            'period': f"{years_back} years",
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'price_start': hist_prices['Close'].iloc[0],
            'price_end': hist_prices['Close'].iloc[-1],
            'total_return': (hist_prices['Close'].iloc[-1] / hist_prices['Close'].iloc[0] - 1) * 100,
            'annual_return': returns.mean() * 252 * 100,
            'annual_volatility': returns.std() * np.sqrt(252) * 100,
            'max_drawdown': ((hist_prices['Close'] / hist_prices['Close'].cummax() - 1).min()) * 100,
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': np.percentile(returns, 5) * 100,
            'var_99': np.percentile(returns, 1) * 100,
            'trading_days': len(hist_prices)
        }
        
        print(f"📊 Historical Analysis Results:")
        print(f"   Period: {analysis['start_date']} to {analysis['end_date']}")
        print(f"   Total Return: {analysis['total_return']:.1f}%")
        print(f"   Annual Volatility: {analysis['annual_volatility']:.1f}%")
        print(f"   Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {analysis['max_drawdown']:.1f}%")
        print(f"   Skewness: {analysis['skewness']:.3f}")
        print(f"   Kurtosis: {analysis['kurtosis']:.3f}")
        
        return analysis, returns.values
        
    except Exception as e:
        print(f"❌ Historical analysis failed: {e}")
        return None, np.array([])

def main_data_fetch_demo():
    """
    Demonstration of real options data fetching capabilities
    """
    print("🚀 REAL OPTIONS DATA FETCHING DEMONSTRATION")
    print("="*60)
    
    # Test symbols (major ETFs and stocks with active options)
    test_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
    
    for symbol in test_symbols[:2]:  # Test first 2 symbols
        print(f"\n📊 Testing {symbol}...")
        print("-" * 30)
        
        # Get comprehensive options data
        options_data = get_yfinance_options_comprehensive(symbol, max_expirations=3)
        
        if options_data:
            # Select best option for analysis
            best_option = select_best_option_for_analysis(options_data)
            
            if best_option:
                print(f"✅ Successfully identified tradable option for {symbol}")
            
            # Get historical analysis
            hist_analysis, returns = get_historical_options_analysis(symbol, years_back=1)
            
            if hist_analysis:
                print(f"✅ Historical analysis complete")
        
        print(f"{'='*30}")
    
    print("\n🎉 Data fetch demonstration complete!")

if __name__ == "__main__":
    main_data_fetch_demo()