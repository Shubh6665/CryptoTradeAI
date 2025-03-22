import numpy as np
from config import MAX_POSITION_SIZE, STOP_LOSS_THRESHOLD, TAKE_PROFIT_THRESHOLD
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management module for the trading bot
    
    Handles position sizing, stop-loss, and take-profit logic
    """
    
    def __init__(self, initial_balance, max_position_size=MAX_POSITION_SIZE, 
                 stop_loss_threshold=STOP_LOSS_THRESHOLD, take_profit_threshold=TAKE_PROFIT_THRESHOLD):
        """
        Initialize the risk manager
        
        Args:
            initial_balance: Initial account balance
            max_position_size: Maximum position size as a fraction of total capital
            stop_loss_threshold: Stop loss threshold as a fraction of entry price
            take_profit_threshold: Take profit threshold as a fraction of entry price
        """
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        
        # Position tracking
        self.position = 0.0
        self.entry_price = 0.0
        self.last_balance = initial_balance
        
        # Performance metrics
        self.returns = []
        self.drawdowns = []
        
        logger.info(f"🛡️ Risk Manager initialized with max position size: {max_position_size*100}%, " 
                    f"stop-loss: {stop_loss_threshold*100}%, take-profit: {take_profit_threshold*100}%")
    
    def calculate_position_size(self, action, balance, current_price):
        """
        Calculate the appropriate position size based on risk parameters
        
        Args:
            action: Model's action value [-1.0, 1.0]
            balance: Current account balance
            current_price: Current asset price
            
        Returns:
            Adjusted action, amount to trade
        """
        # Scale the action to determine what percentage of available funds to use
        action_scale = abs(action[0])  # Use absolute value to determine magnitude
        
        # If action is close to zero, don't trade
        if action_scale < 0.1:
            return 0.0, 0.0
        
        # Calculate maximum allowable position in crypto units
        max_position_units = (balance * self.max_position_size) / current_price
        
        # Scale the position size based on action magnitude
        position_units = max_position_units * action_scale
        
        # Determine action direction
        if action[0] > 0:  # Buy
            if self.position >= max_position_units:
                # Already at or above maximum position
                return 0.0, 0.0
            
            # Adjust buy amount to not exceed maximum position
            max_buy = max(0, max_position_units - self.position)
            amount = min(position_units, max_buy)
            return 1.0, amount
            
        elif action[0] < 0:  # Sell
            if self.position <= 0:
                # No position to sell
                return 0.0, 0.0
                
            # Adjust sell amount to not exceed current position
            amount = min(position_units, self.position)
            return -1.0, amount
            
        return 0.0, 0.0
    
    def update_position(self, action_type, amount, price):
        """
        Update the current position after a trade
        
        Args:
            action_type: 1.0 for buy, -1.0 for sell, 0.0 for hold
            amount: Amount of crypto traded
            price: Trade execution price
        """
        if action_type > 0:  # Buy
            # Update entry price (weighted average if adding to position)
            if self.position > 0:
                self.entry_price = (self.entry_price * self.position + price * amount) / (self.position + amount)
            else:
                self.entry_price = price
                
            # Update position
            self.position += amount
            
        elif action_type < 0:  # Sell
            # Update position
            self.position -= amount
            
            # If position is closed, reset entry price
            if self.position <= 0:
                self.position = 0
                self.entry_price = 0
    
    def check_stop_loss(self, current_price, current_balance):
        """
        Check if stop-loss should be triggered
        
        Args:
            current_price: Current asset price
            current_balance: Current account balance
            
        Returns:
            Boolean indicating whether stop-loss is triggered, amount to sell
        """
        if self.position <= 0 or self.entry_price <= 0:
            return False, 0
            
        # Calculate current drawdown
        if current_price < self.entry_price:
            drawdown = (self.entry_price - current_price) / self.entry_price
            self.drawdowns.append(drawdown)
            
            # Check if stop-loss is triggered
            if drawdown >= self.stop_loss_threshold:
                logger.warning(f"🛑 Stop-loss triggered! Drawdown: {drawdown*100:.2f}%, "
                             f"Entry: ${self.entry_price:.2f}, Current: ${current_price:.2f}")
                return True, self.position
        
        # Check for large account drawdown
        if self.last_balance > 0 and current_balance < self.last_balance:
            account_drawdown = (self.last_balance - current_balance) / self.last_balance
            if account_drawdown >= self.stop_loss_threshold * 2:  # More aggressive account-level stop
                logger.warning(f"🛑 Account stop-loss triggered! Drawdown: {account_drawdown*100:.2f}%")
                return True, self.position
                
        return False, 0
    
    def check_take_profit(self, current_price):
        """
        Check if take-profit should be triggered
        
        Args:
            current_price: Current asset price
            
        Returns:
            Boolean indicating whether take-profit is triggered, amount to sell
        """
        if self.position <= 0 or self.entry_price <= 0:
            return False, 0
            
        # Calculate current profit
        if current_price > self.entry_price:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Check if take-profit is triggered
            if profit_pct >= self.take_profit_threshold:
                logger.info(f"💰 Take-profit triggered! Profit: {profit_pct*100:.2f}%, "
                           f"Entry: ${self.entry_price:.2f}, Current: ${current_price:.2f}")
                return True, self.position * 0.5  # Sell half of the position
        
        return False, 0
    
    def update_metrics(self, current_balance):
        """
        Update performance metrics
        
        Args:
            current_balance: Current account balance
        """
        # Calculate return
        if self.last_balance > 0:
            return_pct = (current_balance - self.last_balance) / self.last_balance
            self.returns.append(return_pct)
        
        self.last_balance = current_balance
    
    def get_risk_metrics(self):
        """
        Calculate risk metrics
        
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Calculate Sharpe ratio
        if self.returns:
            avg_return = np.mean(self.returns)
            std_return = np.std(self.returns) if len(self.returns) > 1 else 1
            metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
            
        # Calculate maximum drawdown
        if self.drawdowns:
            metrics['max_drawdown'] = max(self.drawdowns)
        else:
            metrics['max_drawdown'] = 0
            
        # Calculate current position as a percentage of initial balance
        metrics['position_pct'] = self.position * self.entry_price / self.initial_balance if self.entry_price > 0 else 0
        
        return metrics
    
    def adjust_thresholds(self, performance_metrics):
        """
        Dynamically adjust risk thresholds based on performance
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # If we have severe drawdowns, reduce position size
        if 'max_drawdown' in performance_metrics and performance_metrics['max_drawdown'] > self.stop_loss_threshold * 2:
            old_max_size = self.max_position_size
            self.max_position_size = max(0.05, self.max_position_size * 0.8)
            logger.info(f"⚠️ Reducing max position size from {old_max_size*100:.1f}% to {self.max_position_size*100:.1f}% "
                       f"due to high drawdown ({performance_metrics['max_drawdown']*100:.1f}%)")
            
        # If Sharpe ratio is good, we can be a bit more aggressive
        if 'sharpe_ratio' in performance_metrics and performance_metrics['sharpe_ratio'] > 2.0:
            if self.max_position_size < MAX_POSITION_SIZE:
                old_max_size = self.max_position_size
                self.max_position_size = min(MAX_POSITION_SIZE, self.max_position_size * 1.1)
                logger.info(f"📈 Increasing max position size from {old_max_size*100:.1f}% to {self.max_position_size*100:.1f}% "
                           f"due to good Sharpe ratio ({performance_metrics['sharpe_ratio']:.2f})")
