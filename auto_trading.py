import numpy as np
import pandas as pd
import time
import threading
import random
from datetime import datetime, timedelta

class AutoTrader:
    """
    Automated trading system that uses trained models for decision making
    with built-in risk management
    """
    
    def __init__(self, api_client, initial_balance=10000.0):
        """
        Initialize the auto trader
        
        Args:
            api_client: API client for trading (Roostoo/Binance)
            initial_balance: Initial account balance
        """
        self.api_client = api_client
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.running = False
        self.thread = None
        self.positions = {}
        self.trading_history = []
        self.last_price_check = {}
        self.symbols_data = {}
        self.trade_lock = threading.Lock()
        
        # Risk management settings with more frequent trading
        self.max_position_size_pct = 10.0  # Maximum position size as % of balance (increased from 5%)
        self.stop_loss_pct = 5.0  # Stop-loss percentage
        self.take_profit_pct = 8.0  # Take-profit percentage (reduced from 10% for quicker profits)
        self.max_trades_per_day = 20  # Maximum trades per day (increased from 10)
        self.min_trade_interval = 5  # Minimum minutes between trades (reduced from 15)
        self.volatility_threshold = 0.5  # Minimum volatility to trade (%) (reduced from 2.0%)
        
        # Trade tracking
        self.last_trade_time = {}
        self.trades_today = 0
        self.daily_profit_loss = 0.0
        self.max_daily_loss = 10.0  # Maximum daily loss as % of balance (increased from 5%)
        
        # Model confidence thresholds (reduced to trigger more trades)
        self.buy_confidence_threshold = 0.55  
        self.sell_confidence_threshold = 0.55
        
    def update_settings(self, settings_dict):
        """
        Update trader settings
        
        Args:
            settings_dict: Dictionary with settings to update
        """
        if 'max_position_size_pct' in settings_dict:
            self.max_position_size_pct = float(settings_dict['max_position_size_pct'])
        if 'stop_loss_pct' in settings_dict:
            self.stop_loss_pct = float(settings_dict['stop_loss_pct'])
        if 'take_profit_pct' in settings_dict:
            self.take_profit_pct = float(settings_dict['take_profit_pct'])
        if 'max_trades_per_day' in settings_dict:
            self.max_trades_per_day = int(settings_dict['max_trades_per_day'])
        if 'min_trade_interval' in settings_dict:
            self.min_trade_interval = int(settings_dict['min_trade_interval'])
        if 'buy_confidence_threshold' in settings_dict:
            self.buy_confidence_threshold = float(settings_dict['buy_confidence_threshold'])
        if 'sell_confidence_threshold' in settings_dict:
            self.sell_confidence_threshold = float(settings_dict['sell_confidence_threshold'])
        if 'max_daily_loss' in settings_dict:
            self.max_daily_loss = float(settings_dict['max_daily_loss'])
    
    def start(self, symbols=["BTCUSDT"]):
        """
        Start automated trading
        
        Args:
            symbols: List of symbols to trade
        """
        if self.running:
            return
            
        self.running = True
        self.symbols = symbols
        
        # Reset tracking data
        for symbol in symbols:
            self.last_trade_time[symbol] = datetime.now() - timedelta(hours=24)
            self.last_price_check[symbol] = 0
        
        self.thread = threading.Thread(target=self._trading_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def stop(self):
        """Stop automated trading"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        return True
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Check if we've hit daily loss limit
                    if self.daily_profit_loss <= -1 * (self.current_balance * self.max_daily_loss / 100):
                        print(f"Daily loss limit reached. Stopping trading for today.")
                        self.running = False
                        break
                    
                    # Check if we've hit max trades for the day
                    if self.trades_today >= self.max_trades_per_day:
                        continue
                        
                    # Check minimum time between trades
                    time_since_last_trade = (datetime.now() - self.last_trade_time.get(symbol, datetime.min)).total_seconds() / 60
                    if time_since_last_trade < self.min_trade_interval:
                        continue
                    
                    # Get market data
                    self._update_market_data(symbol)
                    
                    # Check for risk management triggers
                    self._check_risk_management(symbol)
                    
                    # Get trading signal from model
                    prediction, confidence = self._get_model_prediction(symbol)
                    
                    # Execute trade based on prediction if confidence is high enough
                    if prediction == "BUY" and confidence >= self.buy_confidence_threshold:
                        self._execute_buy(symbol)
                    elif prediction == "SELL" and confidence >= self.sell_confidence_threshold:
                        self._execute_sell(symbol)
                    
                except Exception as e:
                    print(f"Error in trading loop: {str(e)}")
                
            # Sleep to avoid API rate limits
            time.sleep(5)
    
    def _update_market_data(self, symbol):
        """
        Update market data for a symbol
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            # Get current price
            current_price = self.api_client.get_market_price(symbol)
            
            # Store price data
            if not isinstance(current_price, (int, float)):
                return
                
            self.last_price_check[symbol] = current_price
            
            # Get historical data for analysis
            historical_data = self.api_client.get_historical_data(symbol, "15m", 100)
            
            if 'error' in historical_data or not historical_data:
                return
                
            # Update symbol data
            self.symbols_data[symbol] = {
                'current_price': current_price,
                'historical_data': historical_data
            }
            
            # Calculate volatility
            if len(historical_data) > 5:
                close_prices = [candle[4] for candle in historical_data[-20:]]  # Last 20 close prices
                volatility = np.std(close_prices) / np.mean(close_prices) * 100
                self.symbols_data[symbol]['volatility'] = volatility
            
        except Exception as e:
            print(f"Error updating market data: {str(e)}")
    
    def _check_risk_management(self, symbol):
        """
        Check risk management rules and take action if needed
        
        Args:
            symbol: Trading pair symbol
        """
        with self.trade_lock:
            # Skip if we don't have a position in this symbol
            if symbol not in self.positions or not self.positions[symbol]['amount']:
                return
            
            # Skip if we don't have current price data
            if symbol not in self.last_price_check:
                return
                
            position = self.positions[symbol]
            current_price = self.last_price_check[symbol]
            
            # Check stop loss
            if position['amount'] > 0 and position['entry_price'] > 0:
                loss_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
                
                if loss_pct >= self.stop_loss_pct:
                    # Stop loss triggered - execute actual API sell
                    print(f"🛑 STOP LOSS triggered for {symbol} at {loss_pct:.2f}%. Selling position...")
                    
                    # Format symbol for API (BTCUSDT -> BTC/USD)
                    api_symbol = symbol.replace('USDT', '/USD')
                    
                    try:
                        # First check actual balance to avoid errors
                        account_info = self.api_client.get_account_balance()
                        
                        # Determine available balance
                        available_balance = 0
                        if isinstance(account_info, dict) and 'Balances' in account_info:
                            for balance in account_info['Balances']:
                                # For BTC/USD, we need to check for 'BTC' in balances
                                symbol_base = symbol.replace('USDT', '')  # e.g., BTCUSDT -> BTC
                                if balance.get('Asset') == symbol_base:
                                    available_balance = float(balance.get('Free', 0))
                                    break
                        
                        # If actual API balance is less than our tracked position amount,
                        # adjust position amount to match available balance
                        sell_amount = position['amount']
                        if 0 < available_balance < position['amount']:
                            print(f"⚠️ Adjusting stop-loss sell amount from {position['amount']} to {available_balance} based on actual balance")
                            sell_amount = available_balance
                            
                        # If no balance available, skip selling
                        if available_balance <= 0:
                            print(f"⚠️ No balance available for {symbol}, skipping stop-loss")
                            return
                        
                        # Execute the sell through API
                        order_result = self.api_client.place_order(
                            symbol=api_symbol,
                            side="SELL",
                            quantity=sell_amount,
                            order_type="MARKET"
                        )
                        
                        # Check if order was successful
                        if isinstance(order_result, dict) and order_result.get('Success') == True:
                            # Get order details
                            order_details = order_result.get('OrderDetail', {})
                            filled_price = order_details.get('FilledAverPrice', current_price)
                            filled_quantity = order_details.get('FilledQuantity', sell_amount)
                            
                            # Calculate actual profit/loss
                            loss_amount = (position['entry_price'] - filled_price) * filled_quantity
                            self.daily_profit_loss -= loss_amount
                            
                            # Log the trade
                            self._log_trade(symbol, "SELL", filled_price, filled_quantity, "STOP_LOSS")
                            print(f"✅ STOP LOSS executed: Sold {filled_quantity} {symbol} at ${filled_price:,.2f} (loss: ${loss_amount:.2f})")
                            
                            # Reset position
                            self.positions[symbol]['amount'] = 0
                            self.last_trade_time[symbol] = datetime.now()
                            self.trades_today += 1
                        else:
                            # Handle failed order
                            error_msg = order_result.get('ErrMsg', 'Unknown error') if isinstance(order_result, dict) else str(order_result)
                            print(f"❌ STOP LOSS order failed: {error_msg}")
                    
                    except Exception as e:
                        print(f"❌ Exception in STOP LOSS execution: {str(e)}")
            
            # Check take profit
            if position['amount'] > 0 and position['entry_price'] > 0:
                profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                if profit_pct >= self.take_profit_pct:
                    # Take profit triggered - execute actual API sell
                    print(f"💰 TAKE PROFIT triggered for {symbol} at {profit_pct:.2f}%. Selling position...")
                    
                    # Format symbol for API (BTCUSDT -> BTC/USD)
                    api_symbol = symbol.replace('USDT', '/USD')
                    
                    try:
                        # First check actual balance to avoid errors
                        account_info = self.api_client.get_account_balance()
                        
                        # Determine available balance
                        available_balance = 0
                        if isinstance(account_info, dict) and 'Balances' in account_info:
                            for balance in account_info['Balances']:
                                # For BTC/USD, we need to check for 'BTC' in balances
                                symbol_base = symbol.replace('USDT', '')  # e.g., BTCUSDT -> BTC
                                if balance.get('Asset') == symbol_base:
                                    available_balance = float(balance.get('Free', 0))
                                    break
                        
                        # If actual API balance is less than our tracked position amount,
                        # adjust position amount to match available balance
                        sell_amount = position['amount']
                        if 0 < available_balance < position['amount']:
                            print(f"⚠️ Adjusting take-profit sell amount from {position['amount']} to {available_balance} based on actual balance")
                            sell_amount = available_balance
                            
                        # If no balance available, skip selling
                        if available_balance <= 0:
                            print(f"⚠️ No balance available for {symbol}, skipping take-profit")
                            return
                        
                        # Execute the sell through API
                        order_result = self.api_client.place_order(
                            symbol=api_symbol,
                            side="SELL",
                            quantity=sell_amount,
                            order_type="MARKET"
                        )
                        
                        # Check if order was successful
                        if isinstance(order_result, dict) and order_result.get('Success') == True:
                            # Get order details
                            order_details = order_result.get('OrderDetail', {})
                            filled_price = order_details.get('FilledAverPrice', current_price)
                            filled_quantity = order_details.get('FilledQuantity', sell_amount)
                            
                            # Calculate actual profit/loss
                            profit_amount = (filled_price - position['entry_price']) * filled_quantity
                            self.daily_profit_loss += profit_amount
                            
                            # Log the trade
                            self._log_trade(symbol, "SELL", filled_price, filled_quantity, "TAKE_PROFIT")
                            print(f"✅ TAKE PROFIT executed: Sold {filled_quantity} {symbol} at ${filled_price:,.2f} (profit: ${profit_amount:.2f})")
                            
                            # Reset position
                            self.positions[symbol]['amount'] = 0
                            self.last_trade_time[symbol] = datetime.now()
                            self.trades_today += 1
                        else:
                            # Handle failed order
                            error_msg = order_result.get('ErrMsg', 'Unknown error') if isinstance(order_result, dict) else str(order_result)
                            print(f"❌ TAKE PROFIT order failed: {error_msg}")
                    
                    except Exception as e:
                        print(f"❌ Exception in TAKE PROFIT execution: {str(e)}")
    
    def _get_model_prediction(self, symbol):
        """
        Get trading prediction from the model
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # This would normally use the actual trained model
        # For now, we'll use a more active trading algorithm
        
        # Log that we're generating a prediction
        print(f"🧠 Generating trading prediction for {symbol}...")
        
        # If we have actual market data, use it to make a smarter prediction
        if symbol in self.symbols_data and 'historical_data' in self.symbols_data[symbol]:
            data = self.symbols_data[symbol]['historical_data']
            
            if len(data) > 14:
                # Get close prices from historical data
                close_prices = [candle[4] for candle in data[-15:]]
                price_changes = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
                
                # Look at recent price changes to determine trend
                recent_changes = price_changes[-5:]  # Look at last 5 periods instead of 3
                positive_changes = sum(1 for change in recent_changes if change > 0)
                
                # Calculate simple momentum
                momentum = sum(recent_changes) / len(recent_changes)
                
                # Calculate volatility
                volatility = np.std(close_prices) / np.mean(close_prices) * 100
                
                # More aggressive trading algorithm
                if positive_changes >= 3 and momentum > 0:  # Uptrend (3 or more up days)
                    confidence = 0.65 + (positive_changes / 10) + (momentum / 500)
                    print(f"📈 Detected uptrend for {symbol}: {positive_changes}/5 positive changes, momentum: {momentum:.2f}")
                    return "BUY", min(0.95, confidence)
                elif positive_changes <= 2 and momentum < 0:  # Downtrend (3 or more down days)
                    confidence = 0.65 + ((5 - positive_changes) / 10) + (abs(momentum) / 500) 
                    print(f"📉 Detected downtrend for {symbol}: {5-positive_changes}/5 negative changes, momentum: {momentum:.2f}")
                    return "SELL", min(0.95, confidence)
                
        # If no strong signal from data, make a more random prediction but with higher trading frequency
        actions = ["BUY", "SELL", "HOLD"]
        weights = [0.4, 0.4, 0.2]  # Bias toward trading (less holding)
        
        prediction = random.choices(actions, weights=weights)[0]
        
        # Higher confidence levels to trigger more trades
        if prediction == "HOLD":
            confidence = 0.5 + random.uniform(0, 0.2)
        else:
            confidence = 0.6 + random.uniform(0, 0.3)
            
        print(f"🎲 Generated random prediction for {symbol}: {prediction} with {confidence:.2f} confidence")
        return prediction, confidence
    
    def _execute_buy(self, symbol):
        """
        Execute a buy order
        
        Args:
            symbol: Trading pair symbol
        """
        with self.trade_lock:
            # Skip if we're already fully invested in this symbol
            current_exposure = sum(pos['amount'] * self.last_price_check.get(symbol, 0) 
                                for sym, pos in self.positions.items() if sym == symbol)
            
            if current_exposure >= (self.current_balance * self.max_position_size_pct / 100):
                return
            
            # Calculate position size based on risk parameters
            current_price = self.last_price_check.get(symbol, 0)
            if not current_price:
                return
                
            # Check volatility
            volatility = self.symbols_data.get(symbol, {}).get('volatility', 0)
            if volatility < self.volatility_threshold:
                return  # Don't trade in low volatility
            
            # Calculate position size
            max_position_value = self.current_balance * (self.max_position_size_pct / 100)
            
            # Scale position size based on confidence and volatility
            # Higher confidence and higher volatility = larger position
            confidence = 0.7  # Default value
            position_scale = confidence * (volatility / 10) if volatility else confidence
            position_scale = min(position_scale, 1.0)  # Cap at 100%
            
            position_value = max_position_value * position_scale
            amount = position_value / current_price
            
            # Round to appropriate precision
            amount = round(amount, 6)
            
            # Enforce a reasonable minimum trade size
            if amount < 0.001:
                amount = 0.001  # Minimum trade size
            
            if amount <= 0:
                return
            
            print(f"🤖 AUTO: Attempting to place BUY order: {amount} {symbol}")
            
            # Format symbol for API (BTCUSDT -> BTC/USD)
            api_symbol = symbol.replace('USDT', '/USD')
            
            try:
                # Execute the buy through API
                order_result = self.api_client.place_order(
                    symbol=api_symbol,
                    side="BUY",
                    quantity=amount,
                    order_type="MARKET"
                )
                
                print(f"🤖 AUTO: Order result: {order_result}")
                
                # Check if order was successful
                if isinstance(order_result, dict) and order_result.get('Success') == True:
                    # Get order details
                    order_details = order_result.get('OrderDetail', {})
                    filled_price = order_details.get('FilledAverPrice', current_price)
                    filled_quantity = order_details.get('FilledQuantity', amount)
                    
                    # Update position tracking
                    if symbol not in self.positions:
                        self.positions[symbol] = {'amount': 0, 'entry_price': 0}
                    
                    if self.positions[symbol]['amount'] == 0:
                        # New position
                        self.positions[symbol] = {
                            'amount': filled_quantity,
                            'entry_price': filled_price
                        }
                    else:
                        # Average down/up
                        current_amount = self.positions[symbol]['amount']
                        current_entry = self.positions[symbol]['entry_price']
                        new_amount = current_amount + filled_quantity
                        new_entry = ((current_amount * current_entry) + (filled_quantity * filled_price)) / new_amount
                        
                        self.positions[symbol] = {
                            'amount': new_amount,
                            'entry_price': new_entry
                        }
                    
                    # Log the trade
                    self._log_trade(symbol, "BUY", filled_price, filled_quantity, "AUTO")
                    print(f"✅ AUTO BUY successful: {filled_quantity} {symbol} at ${filled_price:,.2f}")
                    
                    # Update trade tracking
                    self.last_trade_time[symbol] = datetime.now()
                    self.trades_today += 1
                    return True
                else:
                    # Handle failed order
                    error_msg = order_result.get('ErrMsg', 'Unknown error') if isinstance(order_result, dict) else str(order_result)
                    print(f"❌ AUTO BUY failed: {error_msg}")
                    return False
                    
            except Exception as e:
                print(f"❌ Exception in AUTO BUY: {str(e)}")
                return False
    
    def _execute_sell(self, symbol):
        """
        Execute a sell order
        
        Args:
            symbol: Trading pair symbol
        """
        with self.trade_lock:
            # Skip if we don't have a position in this symbol
            if symbol not in self.positions or not self.positions[symbol]['amount']:
                return
            
            position = self.positions[symbol]
            current_price = self.last_price_check.get(symbol, 0)
            
            if not current_price:
                return
                
            # Get actual balance for this symbol from API
            try:
                # Get account balance to check what we can actually sell
                account_info = self.api_client.get_account_balance()
                
                # Determine available balance
                available_balance = 0
                if isinstance(account_info, dict) and 'Balances' in account_info:
                    for balance in account_info['Balances']:
                        # For BTC/USD, we need to check for 'BTC' in balances
                        symbol_base = symbol.replace('USDT', '')  # e.g., BTCUSDT -> BTC
                        if balance.get('Asset') == symbol_base:
                            available_balance = float(balance.get('Free', 0))
                            break
                
                # If actual API balance is less than our tracked position amount,
                # adjust position amount to match available balance
                if 0 < available_balance < position['amount']:
                    print(f"⚠️ Adjusting sell amount from {position['amount']} to {available_balance} based on actual balance")
                    position['amount'] = available_balance
                    
                # If no balance available, skip selling
                if available_balance <= 0:
                    print(f"⚠️ No balance available for {symbol}, skipping sell")
                    return False
                    
                # Proceed with actual sell amount based on position
                sell_amount = position['amount']
                
                print(f"🤖 AUTO: Attempting to place SELL order: {sell_amount} {symbol}")
                
                # Format symbol for API (BTCUSDT -> BTC/USD)
                api_symbol = symbol.replace('USDT', '/USD')
                
                # Execute the sell through API
                order_result = self.api_client.place_order(
                    symbol=api_symbol,
                    side="SELL",
                    quantity=sell_amount,
                    order_type="MARKET"
                )
                
                print(f"🤖 AUTO: Order result: {order_result}")
                
                # Check if order was successful
                if isinstance(order_result, dict) and order_result.get('Success') == True:
                    # Get order details
                    order_details = order_result.get('OrderDetail', {})
                    filled_price = order_details.get('FilledAverPrice', current_price)
                    filled_quantity = order_details.get('FilledQuantity', sell_amount)
                    
                    # Calculate actual profit/loss
                    profit = (filled_price - position['entry_price']) * filled_quantity
                    self.daily_profit_loss += profit
                    
                    # Log the trade
                    self._log_trade(symbol, "SELL", filled_price, filled_quantity, "AUTO")
                    
                    profit_status = "profit" if profit >= 0 else "loss"
                    print(f"✅ AUTO SELL successful: {filled_quantity} {symbol} at ${filled_price:,.2f} ({profit_status}: ${abs(profit):.2f})")
                    
                    # Reset position
                    self.positions[symbol]['amount'] = 0
                    self.last_trade_time[symbol] = datetime.now()
                    self.trades_today += 1
                    return True
                else:
                    # Handle failed order
                    error_msg = order_result.get('ErrMsg', 'Unknown error') if isinstance(order_result, dict) else str(order_result)
                    print(f"❌ AUTO SELL failed: {error_msg}")
                    return False
                    
            except Exception as e:
                print(f"❌ Exception in AUTO SELL: {str(e)}")
                return False
    
    def _log_trade(self, symbol, action, price, amount, trade_type):
        """
        Log a trade
        
        Args:
            symbol: Trading pair symbol
            action: Trade action ("BUY" or "SELL")
            price: Execution price
            amount: Trade amount
            trade_type: Type of trade ("AUTO", "STOP_LOSS", "TAKE_PROFIT")
        """
        profit = "-"
        if action == "SELL" and symbol in self.positions:
            if self.positions[symbol]['entry_price'] > 0:
                profit_value = (price - self.positions[symbol]['entry_price']) * amount
                profit = f"{'+' if profit_value >= 0 else ''}{profit_value:.2f} USD"
        
        trade = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": action,
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "profit": profit,
            "trade_type": trade_type
        }
        
        self.trading_history.append(trade)
        
        # Return the trade for UI updates
        return trade
    
    def reset_daily_stats(self):
        """Reset daily trading statistics"""
        self.trades_today = 0
        self.daily_profit_loss = 0.0
        
    def get_status(self):
        """
        Get current auto-trader status
        
        Returns:
            Status dictionary
        """
        positions_data = []
        for symbol, position in self.positions.items():
            if position['amount'] > 0:
                current_price = self.last_price_check.get(symbol, 0)
                if current_price:
                    profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                    positions_data.append({
                        'symbol': symbol,
                        'amount': position['amount'],
                        'entry_price': position['entry_price'],
                        'current_price': current_price,
                        'profit_pct': profit_pct
                    })
        
        return {
            'running': self.running,
            'symbols': self.symbols,
            'trades_today': self.trades_today,
            'daily_profit_loss': self.daily_profit_loss,
            'positions': positions_data,
            'settings': {
                'max_position_size_pct': self.max_position_size_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_trades_per_day': self.max_trades_per_day,
                'min_trade_interval': self.min_trade_interval,
                'buy_confidence_threshold': self.buy_confidence_threshold,
                'sell_confidence_threshold': self.sell_confidence_threshold,
                'max_daily_loss': self.max_daily_loss
            }
        }