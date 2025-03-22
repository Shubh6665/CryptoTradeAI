import gym
import numpy as np
import pandas as pd
from gym import spaces
from config import INITIAL_BALANCE, FEATURES, STATE_LOOKBACK_WINDOW, MAX_POSITION_SIZE
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoTradingEnv(gym.Env):
    """Custom Gym environment for cryptocurrency trading with RL"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, lookback_window_size=STATE_LOOKBACK_WINDOW, initial_balance=INITIAL_BALANCE, commission=0.001):
        """
        Initialize the trading environment
        
        Args:
            df: Pandas DataFrame with historical data and features
            lookback_window_size: Number of previous observations to include in state
            initial_balance: Initial account balance
            commission: Trading fee as a percentage (e.g., 0.001 = 0.1%)
        """
        super(CryptoTradingEnv, self).__init__()
        
        # Data setup
        self.df = df.reset_index(drop=True)
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Account state
        self.balance = None
        self.net_worth = None
        self.crypto_held = None
        self.current_step = None
        
        # Orders memory
        self.trades = []
        
        # Performance tracking
        self.returns = []
        self.trade_profits = []
        
        # Define action and observation space
        # Action: sell, hold, buy (-1, 0, 1) with a continuous range for amount
        self.action_space = spaces.Box(
            low=np.array([-1.0]), 
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # State space: previous candles + account information
        feature_count = len(FEATURES) * lookback_window_size
        account_features = 2  # balance and crypto_held
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(feature_count + account_features,),
            dtype=np.float32
        )
        
        # Episode complete
        self.done = False
        
        # Print init message
        logger.info(f"🚀 Environment initialized with {len(df)} timesteps, {lookback_window_size} lookback window")
        
    def reset(self):
        """
        Reset the environment for a new episode
        
        Returns:
            The initial observation
        """
        # Reset account state
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window_size
        
        # Clear trade history
        self.trades = []
        self.returns = []
        self.trade_profits = []
        
        # Reset episode complete flag
        self.done = False
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: The action to take [-1.0, 1.0] where:
                   -1.0 = sell maximum allowed
                   0.0 = hold
                   1.0 = buy maximum allowed
                   
        Returns:
            observation, reward, done, info
        """
        self.current_step += 1
        current_price = self.df.loc[self.current_step, 'close']
        
        # The environment will transition to the next state after this step.
        self.done = self.current_step >= len(self.df) - 1
        
        # Execute the trade action
        action_type = action[0]
        prev_net_worth = self.net_worth
        
        # Execute the trade
        self._take_action(action_type, current_price)
        
        # Update net worth (balance + crypto_held * current_price)
        self.net_worth = self.balance + self.crypto_held * current_price
        
        # Calculate reward
        reward = self._calculate_reward(prev_net_worth)
        
        # Append current return
        self.returns.append(self.net_worth / self.initial_balance - 1.0)
        
        # Get the new observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'current_step': self.current_step,
            'current_price': current_price,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'net_worth': self.net_worth,
            'trade_count': len(self.trades),
            'action': action[0],
            'reward': reward
        }
        
        return observation, reward, self.done, info
    
    def _take_action(self, action, current_price):
        """
        Execute trading action
        
        Args:
            action: Action value [-1.0, 1.0]
            current_price: Current cryptocurrency price
        """
        # Calculate the amount to buy or sell based on the action
        # Action is scaled from [-1, 1] where -1 = sell max, 0 = hold, 1 = buy max
        available_amount = min(
            self.balance / current_price * (1 - self.commission) if action > 0 else self.crypto_held,
            self.net_worth * MAX_POSITION_SIZE / current_price if action > 0 else self.crypto_held
        )
        
        # Scale the action to determine what percentage of available_amount to use
        action_amount = abs(action) * available_amount
        
        # Compute the cost or revenue of the trade
        if action > 0:  # Buy
            cost = action_amount * current_price * (1 + self.commission)
            if cost <= self.balance:
                self.balance -= cost
                self.crypto_held += action_amount
                # Log the trade
                self.trades.append({
                    'step': self.current_step,
                    'price': current_price,
                    'type': 'buy',
                    'amount': action_amount,
                    'cost': cost,
                    'balance': self.balance,
                    'crypto_held': self.crypto_held,
                    'net_worth': self.net_worth
                })
                
        elif action < 0:  # Sell
            revenue = action_amount * current_price * (1 - self.commission)
            if action_amount <= self.crypto_held:
                self.balance += revenue
                self.crypto_held -= action_amount
                # Calculate profit for this trade
                trade_profit = revenue - (action_amount / self.crypto_held) * self._calculate_cost_basis()
                self.trade_profits.append(trade_profit)
                # Log the trade
                self.trades.append({
                    'step': self.current_step,
                    'price': current_price,
                    'type': 'sell',
                    'amount': action_amount,
                    'revenue': revenue,
                    'profit': trade_profit,
                    'balance': self.balance,
                    'crypto_held': self.crypto_held,
                    'net_worth': self.net_worth
                })
    
    def _get_observation(self):
        """
        Construct the state representation for the agent
        
        Returns:
            Numpy array containing the state
        """
        # Get market features from lookback window
        market_features = []
        for i in range(self.current_step - self.lookback_window_size + 1, self.current_step + 1):
            for feature in FEATURES:
                market_features.append(self.df.loc[i, feature])
        
        # Add account information
        account_features = [
            self.balance / self.initial_balance,  # Normalized balance
            self.crypto_held * self.df.loc[self.current_step, 'close'] / self.initial_balance  # Normalized position value
        ]
        
        # Combine features
        observation = np.array(market_features + account_features, dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, prev_net_worth):
        """
        Calculate reward for the current step
        
        Args:
            prev_net_worth: Net worth before this step
            
        Returns:
            Calculated reward value
        """
        # Core reward is net worth change
        net_worth_change = self.net_worth - prev_net_worth
        pct_change = net_worth_change / prev_net_worth if prev_net_worth > 0 else 0
        
        # Basic reward is percent change in net worth
        reward = pct_change
        
        # Add penalty for excessive trading (to discourage overtrading)
        if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step:
            reward -= 0.0001  # Small penalty for trading
        
        return reward
    
    def _calculate_cost_basis(self):
        """Calculate the average cost basis of current crypto holdings"""
        # This is a simple implementation; a more realistic one would track each purchase
        buys = [t for t in self.trades if t['type'] == 'buy']
        if not buys:
            return 0
        
        total_cost = sum(t['cost'] for t in buys)
        total_amount = sum(t['amount'] for t in buys)
        
        if total_amount == 0:
            return 0
            
        return total_cost / total_amount
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode: The rendering mode
        """
        if mode != 'human':
            raise NotImplementedError(f"Rendering mode {mode} is not supported")
        
        current_price = self.df.loc[self.current_step, 'close']
        
        # Get current trade if any
        current_trade = None
        if self.trades and self.trades[-1]['step'] == self.current_step:
            current_trade = self.trades[-1]
        
        # Calculate performance metrics
        if len(self.returns) > 0:
            # Sharpe ratio (using returns and assuming risk-free rate of 0 for simplicity)
            returns_std = np.std(self.returns) if len(self.returns) > 1 else 1
            sharpe_ratio = np.mean(self.returns) / returns_std if returns_std != 0 else 0
            
            # Profit factor (sum of profitable trades / sum of losing trades)
            profitable_trades = [p for p in self.trade_profits if p > 0]
            losing_trades = [p for p in self.trade_profits if p <= 0]
            
            profit_factor = sum(profitable_trades) / abs(sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
            
            # Win rate
            win_rate = len(profitable_trades) / len(self.trade_profits) if self.trade_profits else 0
        else:
            sharpe_ratio = 0
            profit_factor = 0
            win_rate = 0
        
        # Print step information
        print("\n" + "="*50)
        print(f"🕒 Step: {self.current_step}/{len(self.df)-1} | 💰 Net Worth: ${self.net_worth:.2f}")
        print(f"💵 Balance: ${self.balance:.2f} | 🪙 Crypto: {self.crypto_held:.6f} | 📈 Price: ${current_price:.2f}")
        print(f"📊 Performance | Sharpe: {sharpe_ratio:.2f} | Profit Factor: {profit_factor:.2f} | Win Rate: {win_rate:.2f}")
        
        if current_trade:
            trade_type = current_trade['type'].upper()
            icon = "🔴 SELL" if trade_type == "SELL" else "🟢 BUY"
            amount = current_trade['amount']
            value = amount * current_price
            print(f"🔄 Trade: {icon} | Amount: {amount:.6f} | Value: ${value:.2f}")
        
        print("="*50)
        
        return
