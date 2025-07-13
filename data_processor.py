import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles real-time market data processing and feature engineering"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.symbols = config.get_trading_symbols()
        self.data_cache = {}
        self.last_update = {}
        self.is_updating = False
        self.usd_inr_rate = None
        self.last_currency_update = None
        
    def get_market_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Optional[pd.DataFrame]:
        """Fetch market data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data received for {symbol}")
                return None
                
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the data
            self.data_cache[symbol] = data
            self.last_update[symbol] = datetime.now()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        try:
            # Moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price action features
            data['Price_Change'] = data['Close'].pct_change()
            data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
            data['Open_Close_Ratio'] = (data['Close'] - data['Open']) / data['Close']
            
            # Volatility
            data['Volatility'] = data['Price_Change'].rolling(window=20).std()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def get_market_overview(self) -> Optional[pd.DataFrame]:
        """Get overview of multiple market symbols"""
        try:
            overview_data = []
            
            for symbol in self.symbols[:5]:  # Limit to 5 symbols for dashboard
                data = self.get_market_data(symbol, period="1d", interval="5m")
                if data is not None and not data.empty:
                    latest = data.iloc[-1]
                    overview_data.append({
                        'symbol': symbol,
                        'price': latest['Close'],
                        'change': latest['Price_Change'] * 100,
                        'volume': latest['Volume'],
                        'time': latest.name
                    })
            
            if overview_data:
                return pd.DataFrame(overview_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market overview: {str(e)}")
            return None
    
    def update_market_data(self):
        """Update market data for all symbols"""
        if self.is_updating:
            return
            
        self.is_updating = True
        try:
            for symbol in self.symbols:
                # Update if data is older than 1 minute
                if (symbol not in self.last_update or 
                    datetime.now() - self.last_update[symbol] > timedelta(minutes=1)):
                    
                    self.get_market_data(symbol)
                    time.sleep(0.1)  # Rate limiting
                    
        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")
        finally:
            self.is_updating = False
    
    def get_features_for_ml(self, symbol: str, lookback: int = 100) -> Optional[pd.DataFrame]:
        """Get processed features for machine learning models"""
        try:
            data = self.data_cache.get(symbol)
            if data is None or data.empty:
                data = self.get_market_data(symbol, period="1mo", interval="1h")
                
            if data is None or len(data) < lookback:
                return None
            
            # Select features for ML
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'MACD', 'MACD_Signal', 'RSI',
                'BB_Upper', 'BB_Lower', 'Volume_Ratio',
                'High_Low_Ratio', 'Open_Close_Ratio', 'Volatility'
            ]
            
            # Get last N rows
            features = data[feature_columns].tail(lookback).copy()
            
            # Handle missing values
            features = features.ffill().fillna(0)
            
            # Normalize features (except volume which is already ratio-based)
            price_columns = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'BB_Upper', 'BB_Lower']
            for col in price_columns:
                if col in features.columns:
                    features[f'{col}_norm'] = features[col] / features['Close'].iloc[-1]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data for backtesting"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1h")
            
            if data.empty:
                return None
                
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def calculate_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for portfolio symbols"""
        try:
            price_data = {}
            
            for symbol in self.symbols:
                data = self.data_cache.get(symbol)
                if data is not None and not data.empty:
                    price_data[symbol] = data['Close']
            
            if len(price_data) < 2:
                return None
                
            correlation_df = pd.DataFrame(price_data)
            return correlation_df.corr()
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return None
    
    def get_usd_inr_rate(self) -> float:
        """Get current USD to INR exchange rate"""
        try:
            # Check if we need to update the rate (update every 15 minutes)
            current_time = datetime.now()
            if (self.last_currency_update is None or 
                (current_time - self.last_currency_update).total_seconds() > 900):
                
                # Fetch USD/INR rate from Yahoo Finance
                ticker = yf.Ticker("USDINR=X")
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    self.usd_inr_rate = data['Close'].iloc[-1]
                    self.last_currency_update = current_time
                    self.logger.info(f"Updated USD/INR rate: {self.usd_inr_rate:.4f}")
                else:
                    # Fallback to approximate rate if API fails
                    if self.usd_inr_rate is None:
                        self.usd_inr_rate = 83.0  # Approximate rate
                        self.logger.warning("Using fallback USD/INR rate: 83.0")
            
            return self.usd_inr_rate or 83.0
            
        except Exception as e:
            self.logger.error(f"Error fetching USD/INR rate: {str(e)}")
            # Return fallback rate
            return 83.0
    
    def convert_usd_to_inr(self, usd_amount: float) -> float:
        """Convert USD amount to INR"""
        try:
            rate = self.get_usd_inr_rate()
            inr_amount = usd_amount * rate
            return inr_amount
        except Exception as e:
            self.logger.error(f"Error converting USD to INR: {str(e)}")
            return usd_amount * 83.0  # Fallback conversion
