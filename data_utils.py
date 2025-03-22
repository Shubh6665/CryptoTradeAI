import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, TIMEFRAME, HISTORICAL_DATA_LIMIT, DATA_SAVE_PATH
import ta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_binance_client():
    """Initialize and return a Binance client instance"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("❌ Binance API credentials not found in environment variables")
        raise ValueError("Binance API credentials are required")
    
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        # Test connectivity
        client.ping()
        logger.info("✅ Successfully connected to Binance API")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Binance API: {str(e)}")
        raise

def fetch_historical_data(client, symbol=SYMBOL, interval=TIMEFRAME, limit=HISTORICAL_DATA_LIMIT, save_csv=True):
    """
    Fetch historical kline/candlestick data from Binance
    
    Args:
        client: Binance client instance
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Candle interval (e.g., '15m')
        limit: Number of candles to fetch
        save_csv: Whether to save the data to a CSV file
        
    Returns:
        Pandas DataFrame with OHLCV data and calculated indicators
    """
    try:
        logger.info(f"🔍 Fetching {limit} {interval} candles for {symbol}...")
        
        # Fetch klines (candlestick data)
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string columns to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Save to CSV if requested
        if save_csv:
            filename = f"{DATA_SAVE_PATH}{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"✅ Data saved to {filename}")
        
        logger.info(f"✅ Successfully fetched historical data: {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"❌ Error fetching historical data: {str(e)}")
        raise

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df: Pandas DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(
            df['close'],
             window_slow=26, 
             window_fast=12, 
             window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        
        # EMA (Exponential Moving Average)
        df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
        df['ema_medium'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

        
        # Fill any NaN values created by indicators that need more data than available
        df.fillna(method='bfill', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error adding technical indicators: {str(e)}")
        raise

def normalize_data(df):
    """
    Normalize the data for the neural network
    
    Args:
        df: Pandas DataFrame with features
        
    Returns:
        DataFrame with normalized features
    """
    df_norm = df.copy()
    
    # Normalize price and volume data using percentages relative to first value
    for col in ['open', 'high', 'low', 'close']:
        first_val = df_norm[col].iloc[0]
        df_norm[col] = df_norm[col] / first_val - 1.0
    
    # Normalize volume
    first_vol = df_norm['volume'].iloc[0]
    df_norm['volume'] = df_norm['volume'] / first_vol - 1.0
    
    # Normalize indicators to [-1, 1] range
    for col in ['rsi']:
        df_norm[col] = df_norm[col] / 100.0 * 2 - 1
    
    for col in ['macd', 'macd_signal', 'macd_hist', 'ema_short', 'ema_medium', 'ema_long']:
        # Normalize based on historic range or using min-max scaling
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:  # Avoid division by zero
            df_norm[col] = 2 * (df_norm[col] - min_val) / (max_val - min_val) - 1
    
    return df_norm

def prepare_train_test_data(df, train_test_split=0.8):
    """
    Split the data into training and testing sets
    
    Args:
        df: Pandas DataFrame with features
        train_test_split: Fraction of data to use for training
        
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
    # Normalize the data
    df_normalized = normalize_data(df)
    
    # Determine split point
    split_idx = int(len(df_normalized) * train_test_split)
    
    # Split the data
    train_df = df_normalized.iloc[:split_idx].reset_index(drop=True)
    test_df = df_normalized.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"✅ Data split: Training set: {len(train_df)} rows, Testing set: {len(test_df)} rows")
    
    return train_df, test_df

def get_live_market_data(client, symbol=SYMBOL):
    """
    Get current market data for a symbol
    
    Args:
        client: Binance client instance
        symbol: Trading pair symbol
        
    Returns:
        Current market data including latest price
    """
    try:
        # Get current ticker
        ticker = client.get_ticker(symbol=symbol)
        
        # Get latest klines (most recent candles)
        klines = client.get_klines(
            symbol=symbol,
            interval=TIMEFRAME,
            limit=50  # Get enough data to calculate indicators
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string columns to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Get the most recent candle with indicators
        latest_data = df.iloc[-1].to_dict()
        latest_data['current_price'] = float(ticker['lastPrice'])
        
        return latest_data, df
        
    except Exception as e:
        logger.error(f"❌ Error fetching live market data: {str(e)}")
        raise

def visualize_data(df, title="BTC/USDT Price History", save_path=None):
    """
    Visualize price and volume data
    
    Args:
        df: Pandas DataFrame with OHLCV data
        title: Plot title
        save_path: Path to save the figure, if specified
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    ax1.set_title(title)
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot volume
    ax2.bar(df['timestamp'], df['volume'], color='green', alpha=0.5, label='Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"✅ Chart saved to {save_path}")
    
    return fig
