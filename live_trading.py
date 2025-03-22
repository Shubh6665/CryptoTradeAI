import time
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import threading
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, TIMEFRAME, TRADE_QUANTITY
from data_utils import initialize_binance_client, get_live_market_data, normalize_data
from risk_management import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTrader:
    """
    Live trading module for executing trades based on RL model predictions
    """
    
    def __init__(self, model, initial_balance=10000.0):
        """
        Initialize the live trader
        
        Args:
            model: Trained RL model
            initial_balance: Initial account balance for risk calculation
        """
        self.model = model
        self.initial_balance = initial_balance
        self.symbol = SYMBOL
        self.trade_quantity = TRADE_QUANTITY
        
        # Initialize Binance client
        self.client = initialize_binance_client()
        
        # Verify connection and get account info
        self.account_info = self.client.get_account()
        logger.info(f"✅ Account connected. Status: {self.account_info['accountType']}")
        
        # Initialize risk manager
        self.risk_manager = RiskManager(initial_balance)
        
        # Trading state
        self.is_trading_active = False
        self.last_action_time = None
        self.min_time_between_trades = 60  # seconds
        
        # Performance tracking
        self.trades = []
        self.portfolio_history = []
        self.current_portfolio_value = self.calculate_portfolio_value()
        self.portfolio_history.append(self.current_portfolio_value)
        
        # Latest observation/action
        self.latest_observation = None
        self.latest_action = None
        self.latest_market_data = None
        
        logger.info(f"🚀 Live trader initialized for {self.symbol}")
    
    def preprocess_observation(self, market_data_df):
        """
        Convert market data to a format suitable for the model
        
        Args:
            market_data_df: DataFrame with market data
            
        Returns:
            Preprocessed observation array
        """
        # Normalize data
        normalized_df = normalize_data(market_data_df.copy())
        
        # Extract features from the last `lookback_window_size` rows
        feature_data = []
        for feature in FEATURES:
            if feature in normalized_df.columns:
                feature_data.extend(normalized_df[feature].tolist())
            else:
                logger.warning(f"⚠️ Feature {feature} not found in market data")
                feature_data.extend([0] * len(normalized_df))
        
        # Add account information
        account_features = [
            self.risk_manager.last_balance / self.initial_balance,  # Normalized balance
            self.risk_manager.position * self.latest_market_data.get('current_price', 0) / self.initial_balance  # Normalized position value
        ]
        
        # Combine features
        observation = np.array(feature_data + account_features, dtype=np.float32)
        
        return observation
    
    def calculate_portfolio_value(self):
        """
        Calculate current portfolio value
        
        Returns:
            Current portfolio value
        """
        try:
            # Get current ticker price
            ticker = self.client.get_ticker(symbol=self.symbol)
            current_price = float(ticker['lastPrice'])
            
            # Get balances
            account = self.client.get_account()
            
            # Extract base and quote asset from symbol (e.g., 'BTCUSDT' -> 'BTC' and 'USDT')
            base_asset = self.symbol[:-4] if self.symbol.endswith('USDT') else self.symbol[:3]
            quote_asset = 'USDT' if self.symbol.endswith('USDT') else self.symbol[3:]
            
            # Find balances
            base_balance = 0
            quote_balance = 0
            
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    base_balance = float(balance['free']) + float(balance['locked'])
                elif balance['asset'] == quote_asset:
                    quote_balance = float(balance['free']) + float(balance['locked'])
            
            # Calculate total value in quote asset
            total_value = quote_balance + (base_balance * current_price)
            
            logger.info(f"💰 Portfolio Value: {total_value:.2f} {quote_asset} | {base_asset}: {base_balance:.8f} | {quote_asset}: {quote_balance:.2f}")
            
            return total_value
            
        except Exception as e:
            logger.error(f"❌ Error calculating portfolio value: {str(e)}")
            return self.current_portfolio_value  # Return last known value
    
    def execute_trade(self, action_type, amount):
        """
        Execute a trade on the exchange
        
        Args:
            action_type: 1.0 for buy, -1.0 for sell
            amount: Amount to trade
            
        Returns:
            Dictionary with trade result
        """
        if amount <= 0:
            logger.warning("⚠️ Cannot execute trade with zero amount")
            return None
            
        side = 'BUY' if action_type > 0 else 'SELL'
        
        try:
            # Round amount to appropriate precision
            symbol_info = self.client.get_symbol_info(self.symbol)
            step_size = 0.00001  # Default fallback
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
            
            # Calculate precision from step size
            precision = 0
            if step_size < 1:
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                
            # Round amount to precision
            amount = round(amount, precision)
            
            # Ensure minimum notional value
            min_notional = None
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(filter['minNotional'])
            
            # Get current price
            ticker = self.client.get_ticker(symbol=self.symbol)
            current_price = float(ticker['lastPrice'])
            
            # Check if order meets minimum notional value
            if min_notional and amount * current_price < min_notional:
                logger.warning(f"⚠️ Order value {amount * current_price:.2f} below minimum notional {min_notional}")
                return None
                
            # Execute the order
            logger.info(f"🔄 Executing {side} order for {amount} {self.symbol} at ~{current_price}")
            
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=amount
            )
            
            # Record the trade
            trade = {
                'time': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': side,
                'amount': amount,
                'price': current_price,  # Approximate price
                'order_id': order['orderId'],
                'status': order['status']
            }
            
            self.trades.append(trade)
            logger.info(f"✅ Order executed: {side} {amount} {self.symbol} at ~{current_price}")
            
            # Update risk manager position
            self.risk_manager.update_position(action_type, amount, current_price)
            
            return trade
            
        except BinanceAPIException as e:
            logger.error(f"❌ Binance API error: {e.message}")
            return None
        except Exception as e:
            logger.error(f"❌ Error executing trade: {str(e)}")
            return None
    
    def check_risk_management_triggers(self, current_price):
        """
        Check if risk management should trigger a trade
        
        Args:
            current_price: Current market price
            
        Returns:
            action_type, amount tuple if risk management triggers a trade, (0, 0) otherwise
        """
        # Check stop-loss
        stop_loss_triggered, stop_loss_amount = self.risk_manager.check_stop_loss(
            current_price, 
            self.current_portfolio_value
        )
        
        if stop_loss_triggered and stop_loss_amount > 0:
            logger.warning(f"🛑 Stop-loss triggered at ${current_price:.2f}")
            return -1.0, stop_loss_amount
            
        # Check take-profit
        take_profit_triggered, take_profit_amount = self.risk_manager.check_take_profit(
            current_price
        )
        
        if take_profit_triggered and take_profit_amount > 0:
            logger.info(f"💰 Take-profit triggered at ${current_price:.2f}")
            return -1.0, take_profit_amount
            
        return 0.0, 0.0
    
    def start_trading(self):
        """Start the live trading loop"""
        if self.is_trading_active:
            logger.warning("⚠️ Trading already active")
            return
            
        self.is_trading_active = True
        logger.info("🚀 Starting live trading...")
        
        # Start trading in a separate thread to avoid blocking
        trading_thread = threading.Thread(target=self._trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
    
    def stop_trading(self):
        """Stop the live trading loop"""
        if not self.is_trading_active:
            logger.warning("⚠️ Trading already stopped")
            return
            
        logger.info("⏹️ Stopping live trading...")
        self.is_trading_active = False
    
def _trading_loop(self):
    """
    Main trading loop that automatically checks market data,
    gets model predictions, evaluates risk management, and executes trades.
    """
    while self.is_trading_active:
        try:
            # 1. Update market data
            self.latest_market_data, market_data_df = get_live_market_data(self.client, self.symbol)
            if not self.latest_market_data or 'current_price' not in self.latest_market_data:
                logger.error("Market data unavailable; skipping iteration.")
                time.sleep(10)
                continue
            current_price = self.latest_market_data['current_price']
            
            # 2. Update portfolio value and history
            self.current_portfolio_value = self.calculate_portfolio_value()
            self.portfolio_history.append(self.current_portfolio_value)
            
            # 3. Update risk metrics
            self.risk_manager.update_metrics(self.current_portfolio_value)
            
            # 4. Check risk management triggers (e.g., stop-loss or take-profit)
            action_type, amount = self.check_risk_management_triggers(current_price)
            if action_type != 0 and amount > 0:
                logger.info("Risk trigger activated; executing risk-based trade.")
                self.execute_trade(action_type, amount)
                self.last_action_time = datetime.now()
            else:
                # 5. Preprocess observation for the model
                observation = self.preprocess_observation(market_data_df)
                self.latest_observation = observation
                
                # 6. Get model prediction
                model_action, _ = self.model.predict(observation, deterministic=True)
                self.latest_action = model_action
                
                # Interpret model action value (e.g., >0.2: BUY, < -0.2: SELL, else HOLD)
                action_value = model_action[0]
                if action_value > 0.2:
                    signal = "BUY"
                elif action_value < -0.2:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                logger.info(f"Model prediction: {signal} ({action_value:.4f}) at price ${current_price:.2f}")
                
                # 7. Check if enough time has passed since last action
                can_trade = (
                    self.last_action_time is None or 
                    (datetime.now() - self.last_action_time).total_seconds() >= self.min_time_between_trades
                )
                
                if can_trade and signal != "HOLD":
                    # Calculate trade size via risk manager
                    adjusted_action, trade_amount = self.risk_manager.calculate_position_size(
                        model_action, self.current_portfolio_value, current_price
                    )
                    
                    if adjusted_action != 0 and trade_amount > 0:
                        result = self.execute_trade(adjusted_action, trade_amount)
                        if result:
                            self.last_action_time = datetime.now()
            
            # 8. Optionally, dynamically adjust risk parameters
            risk_metrics = self.risk_manager.get_risk_metrics()
            self.risk_manager.adjust_thresholds(risk_metrics)
            
            # 9. Sleep before next iteration
            time.sleep(10)  # Check every 10 seconds; adjust as needed
            
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {str(e)}")
            time.sleep(30)  # Wait longer on error
    
    def get_trading_status(self):
        """
        Get current trading status
        
        Returns:
            Dictionary with current trading status
        """
        return {
            'is_active': self.is_trading_active,
            'current_price': self.latest_market_data['current_price'] if self.latest_market_data else None,
            'portfolio_value': self.current_portfolio_value,
            'position': self.risk_manager.position,
            'entry_price': self.risk_manager.entry_price,
            'last_action': float(self.latest_action[0]) if self.latest_action is not None else None,
            'trade_count': len(self.trades),
            'last_trade': self.trades[-1] if self.trades else None,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
    
    def save_trading_history(self, filename=None):
        """
        Save trading history to a file
        
        Args:
            filename: Output filename, defaults to timestamp
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_history_{timestamp}.json"
            
        history = {
            'symbol': self.symbol,
            'start_time': self.trades[0]['time'] if self.trades else None,
            'end_time': datetime.now().isoformat(),
            'trades': self.trades,
            'portfolio_history': self.portfolio_history
        }
        
        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)
            
        logger.info(f"✅ Trading history saved to {filename}")
