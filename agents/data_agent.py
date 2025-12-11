import yfinance as yf
import pandas as pd
import numpy as np

class DataAgent:
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None
        self.daily_corr = None

    def fetch_data(self):
        """
        Fetches HOURLY data for the last 3 months for model training.
        This enables 'High-Frequency' analysis and increases the dataset size (~2000+ rows).
        """
        # 1. Hourly Data for Model
        df = yf.download(self.symbol, period="3mo", interval="1h", progress=False)
        
        # MultiIndex correction (due to yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 2. Daily Data for Business Analytics (Correlation)
        # Correlation calculation is noisy on hourly data, so we fetch daily data separately.
        tickers = [self.symbol, "SPY", "GC=F"] # SP500 and Gold
        macro_data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
        if isinstance(macro_data.columns, pd.MultiIndex):
            macro_data.columns = macro_data.columns.get_level_values(0)
            
        self.df = df
        self.daily_corr = macro_data
        return df

    def add_indicators(self):
        """Calculates technical indicators (Feature Engineering)."""
        if self.df is None: return None
        
        data = self.df.copy()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Upper_BB'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
        data['Lower_BB'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
        
        # Volatility (Standard Deviation)
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=10).std()
        
        # TARGET CREATION:
        # Is the average price of the next 3 hours higher than the current price?
        # 1: Rise, 0: Fall/Flat
        data['Future_Price'] = data['Close'].shift(-3)
        data['Target'] = (data['Future_Price'] > data['Close']).astype(int)
        
        data.dropna(inplace=True)
        self.df = data
        return data

    def get_market_correlation(self):
        """Calculates SP500 and Gold correlation based on daily data."""
        if self.daily_corr is None:
            return {}
        
        corr = self.daily_corr.corr()
        
        # Match symbol name with column name in dataset
        # yfinance sometimes returns 'BTC-USD' differently, checking just in case.
        symbol_col = self.symbol
        if symbol_col not in corr.columns:
            # If no exact match, assume the first column is crypto
            symbol_col = corr.columns[0]

        return {
            "SP500_Corr": corr.loc[symbol_col, 'SPY'] if 'SPY' in corr.columns else 0,
            "Gold_Corr": corr.loc[symbol_col, 'GC=F'] if 'GC=F' in corr.columns else 0
        }