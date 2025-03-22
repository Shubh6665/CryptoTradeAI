import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Trading Configuration
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
TRADE_QUANTITY = 0.001  # Default trade size in BTC
INITIAL_BALANCE = 10000.0  # Initial USD balance for backtesting

# Historical Data Parameters
HISTORICAL_DATA_LIMIT = 1000  # Number of candles to fetch
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing

# RL Model Parameters
MODELS_TO_TRAIN = ["SAC", "A2C", "PPO"]  # Models to train
TRAINING_TIMESTEPS = 20000  # Training steps for each model
LEARNING_RATE = 0.0003  # Learning rate for RL models

# Environment Parameters
STATE_LOOKBACK_WINDOW = 10  # Number of previous candles to include in state
FEATURES = [
    "close",
    "high",
    "low",
    "volume",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_short",
    "ema_medium",
    "ema_long"
]

# Risk Management
MAX_POSITION_SIZE = 0.2  # Maximum position size as a fraction of total capital
STOP_LOSS_THRESHOLD = 0.05  # Stop loss at 5% drawdown
TAKE_PROFIT_THRESHOLD = 0.1  # Take profit at 10% gain

# Path Configuration
MODEL_SAVE_PATH = "./trained_models/"
DATA_SAVE_PATH = "./data/"

# Ensure necessary directories exist
import os
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_SAVE_PATH, exist_ok=True)
