import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_returns(portfolio_values):
    """
    Calculate daily returns from portfolio values
    
    Args:
        portfolio_values: List or array of portfolio values
        
    Returns:
        Array of daily returns (percentage)
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return []
    
    returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] * 100
        returns.append(daily_return)
    
    return returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio based on returns
    
    Args:
        returns: List or array of returns (percentage)
        risk_free_rate: Annual risk-free rate (percentage)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_risk_free = risk_free_rate / 365
    
    excess_returns = [r - daily_risk_free for r in returns]
    
    # Calculate annualized Sharpe ratio
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
    
    return sharpe

def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown
    
    Args:
        portfolio_values: List or array of portfolio values
        
    Returns:
        Maximum drawdown (percentage)
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return 0.0
    
    max_drawdown = 0
    peak = portfolio_values[0]
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sortino ratio based on returns
    
    Args:
        returns: List or array of returns (percentage)
        risk_free_rate: Annual risk-free rate (percentage)
        
    Returns:
        Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_risk_free = risk_free_rate / 365
    
    excess_returns = [r - daily_risk_free for r in returns]
    
    # Calculate downside deviation (only negative returns)
    negative_returns = [r for r in excess_returns if r < 0]
    
    if not negative_returns:
        return 0.0  # No negative returns, avoid division by zero
    
    downside_deviation = np.std(negative_returns)
    
    # Calculate annualized Sortino ratio
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(365)
    
    return sortino

def calculate_winning_rate(trades):
    """
    Calculate win/loss ratio from trades
    
    Args:
        trades: List of trade dictionaries with 'profit' field
        
    Returns:
        Winning rate (percentage)
    """
    if not trades:
        return 0.0
    
    winning_trades = 0
    
    for trade in trades:
        if trade['type'] == 'SELL':
            profit_str = trade.get('profit', '0')
            
            try:
                # Extract numeric value from profit string
                if isinstance(profit_str, str):
                    profit_str = profit_str.replace('+', '').replace('$', '').replace('USD', '').strip()
                    if profit_str == '-':
                        profit = 0
                    else:
                        profit = float(profit_str)
                else:
                    profit = float(profit_str)
                
                if profit > 0:
                    winning_trades += 1
            except:
                pass
    
    # Count only SELL trades
    total_sell_trades = sum(1 for trade in trades if trade['type'] == 'SELL')
    
    if total_sell_trades == 0:
        return 0.0
    
    return (winning_trades / total_sell_trades) * 100

def generate_portfolio_history(trades, initial_balance=10000.0, days=30):
    """
    Generate portfolio value history based on trades
    
    Args:
        trades: List of trade dictionaries
        initial_balance: Initial account balance
        days: Number of days to generate history for
        
    Returns:
        DataFrame with dates and portfolio values
    """
    if not trades:
        # Generate demo data if no trades
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date)
        values = [initial_balance]
        
        for i in range(1, len(dates)):
            # Add some random fluctuation
            prev_value = values[-1]
            change = prev_value * np.random.uniform(-0.01, 0.02) 
            values.append(prev_value + change)
        
        return pd.DataFrame({'date': dates, 'value': values})
    
    # Process real trades
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    if not sell_trades:
        return generate_portfolio_history([], initial_balance, days)
    
    # Sort trades by time
    try:
        sorted_trades = sorted(trades, key=lambda t: datetime.strptime(t.get('time', '2023-01-01'), '%Y-%m-%d %H:%M:%S'))
    except:
        sorted_trades = trades
    
    # Get the earliest and latest trade dates
    try:
        start_date = datetime.strptime(sorted_trades[0].get('time', '2023-01-01'), '%Y-%m-%d %H:%M:%S')
        end_date = datetime.now()
    except:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    
    # Ensure at least 'days' days of history
    if (end_date - start_date).days < days:
        start_date = end_date - timedelta(days=days)
    
    # Generate daily dates
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Initialize portfolio values with initial balance
    portfolio_values = [initial_balance]
    current_value = initial_balance
    
    # Track the dates of trades
    trade_dates = []
    
    for trade in sorted_trades:
        if trade['type'] == 'SELL':
            try:
                trade_date = datetime.strptime(trade.get('time', '2023-01-01'), '%Y-%m-%d %H:%M:%S')
                profit_str = trade.get('profit', '0')
                
                # Extract profit value
                if isinstance(profit_str, str):
                    profit_str = profit_str.replace('+', '').replace('$', '').replace('USD', '').strip()
                    if profit_str == '-':
                        profit = 0
                    else:
                        profit = float(profit_str)
                else:
                    profit = float(profit_str)
                
                # Update current value with profit
                current_value += profit
                
                # Record the trade date
                trade_dates.append((trade_date, current_value))
            except:
                pass
    
    # Generate portfolio values for each date
    portfolio_df = pd.DataFrame({'date': dates})
    
    # Interpolate values between trade dates
    values = []
    last_value = initial_balance
    
    for date in dates:
        # Find the closest trade value that's earlier than or equal to this date
        trade_before = [(d, v) for d, v in trade_dates if d <= date]
        
        if trade_before:
            # Use the most recent trade value
            last_value = trade_before[-1][1]
        
        values.append(last_value)
    
    portfolio_df['value'] = values
    
    return portfolio_df

def calculate_all_metrics(portfolio_values, trades, risk_free_rate=1.0):
    """
    Calculate all trading performance metrics
    
    Args:
        portfolio_values: List or array of portfolio values
        trades: List of trade dictionaries
        risk_free_rate: Annual risk-free rate (percentage)
        
    Returns:
        Dictionary of performance metrics
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'winning_rate': 0.0,
            'trade_count': 0
        }
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Calculate total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    
    # Calculate annualized return (assuming daily values)
    days = len(portfolio_values) - 1
    annualized_return = ((1 + (total_return / 100)) ** (365 / days) - 1) * 100 if days > 0 else 0
    
    # Calculate other metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(portfolio_values)
    win_rate = calculate_winning_rate(trades)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'winning_rate': win_rate,
        'trade_count': len([t for t in trades if t['type'] == 'SELL'])
    }