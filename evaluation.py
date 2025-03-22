import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_performance_metrics(portfolio_values, initial_balance=10000.0, risk_free_rate=0.001):
    """
    Calculate trading performance metrics
    
    Args:
        portfolio_values: List or array of portfolio values over time
        initial_balance: Initial account balance
        risk_free_rate: Annualized risk-free rate (e.g., 0.001 = 0.1%)
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert to numpy array if needed
    portfolio_values = np.array(portfolio_values)
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate metrics
    total_return = (portfolio_values[-1] / initial_balance) - 1
    daily_returns_mean = np.mean(returns)
    daily_returns_std = np.std(returns) if len(returns) > 1 else 1
    
    # Adjust risk-free rate to match the timeframe of returns (assuming daily)
    daily_risk_free = risk_free_rate / 252  # Approximation for daily
    
    # Sharpe ratio calculation
    excess_returns = daily_returns_mean - daily_risk_free
    sharpe_ratio = excess_returns / daily_returns_std if daily_returns_std > 0 else 0
    
    # Maximum drawdown calculation
    cumulative_returns = np.cumprod(1 + np.append([0], returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Compile metrics
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': daily_returns_std,
        'final_balance': portfolio_values[-1]
    }
    
    return metrics

def analyze_trades(trades, initial_balance=10000.0):
    """
    Analyze trading history
    
    Args:
        trades: List of trade dictionaries
        initial_balance: Initial account balance
        
    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_profit_per_trade': 0,
            'avg_trade_duration': 0
        }
    
    # Separate buys and sells
    buys = [t for t in trades if t['type'] == 'buy']
    sells = [t for t in trades if t['type'] == 'sell']
    
    # Calculate basic metrics
    total_trades = len(buys) + len(sells)
    
    # Calculate profit metrics if we have sell trades
    if sells:
        profitable_trades = [t for t in sells if t.get('profit', 0) > 0]
        losing_trades = [t for t in sells if t.get('profit', 0) <= 0]
        
        win_rate = len(profitable_trades) / len(sells) if sells else 0
        
        total_profit = sum(t.get('profit', 0) for t in profitable_trades)
        total_loss = abs(sum(t.get('profit', 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_profit_per_trade = sum(t.get('profit', 0) for t in sells) / len(sells)
    else:
        win_rate = 0
        profit_factor = 0
        avg_profit_per_trade = 0
    
    # Calculate average trade duration (if time information is available)
    # This assumes trades have 'time' or 'timestamp' fields, adjust as needed
    # Here we're using 'step' as a proxy for time
    if len(buys) > 0 and len(sells) > 0:
        # Pair buys and sells sequentially (simple approach)
        paired_trades = min(len(buys), len(sells))
        durations = [sells[i]['step'] - buys[i]['step'] for i in range(paired_trades)]
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
    else:
        avg_trade_duration = 0
    
    metrics = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_trade_duration': avg_trade_duration
    }
    
    return metrics

def plot_portfolio_performance(portfolio_values, benchmark_values=None, title="Portfolio Performance"):
    """
    Plot portfolio performance over time
    
    Args:
        portfolio_values: List or array of portfolio values
        benchmark_values: Optional list or array of benchmark values (e.g., buy-and-hold)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot portfolio values
    x = range(len(portfolio_values))
    ax.plot(x, portfolio_values, label='Trading Strategy', color='blue', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_values is not None:
        ax.plot(x, benchmark_values, label='Buy and Hold', color='green', linestyle='--', linewidth=1.5)
    
    # Calculate returns for annotation
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = ((final_value / initial_value) - 1) * 100
    
    # Add return annotation
    ax.annotate(f'Return: {total_return:.2f}%', 
                xy=(0.02, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    
    # Add labels and grid
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Portfolio Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_trade_analysis(trades, prices):
    """
    Plot trade entry and exit points on price chart
    
    Args:
        trades: List of trade dictionaries
        prices: List or array of asset prices
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price history
    x = range(len(prices))
    ax.plot(x, prices, label='Asset Price', color='black', alpha=0.7)
    
    # Extract buy and sell points
    buys = [(t['step'], t['price']) for t in trades if t['type'] == 'buy']
    sells = [(t['step'], t['price']) for t in trades if t['type'] == 'sell']
    
    # Plot buy points
    if buys:
        buy_x, buy_y = zip(*buys)
        ax.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy', alpha=0.7)
    
    # Plot sell points
    if sells:
        sell_x, sell_y = zip(*sells)
        ax.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell', alpha=0.7)
    
    # Add labels and grid
    ax.set_title('Trading Activity')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def generate_performance_report(metrics, trade_metrics, benchmark_metrics=None):
    """
    Generate a performance report as a formatted string
    
    Args:
        metrics: Dictionary of performance metrics
        trade_metrics: Dictionary of trade metrics
        benchmark_metrics: Optional dictionary of benchmark metrics
        
    Returns:
        Formatted report string
    """
    report = [
        "=" * 50,
        "📊 TRADING PERFORMANCE REPORT",
        "=" * 50,
        "",
        "🔹 PERFORMANCE METRICS:",
        f"  • Total Return: {metrics['total_return']*100:.2f}%",
        f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
        f"  • Max Drawdown: {metrics['max_drawdown']*100:.2f}%",
        f"  • Volatility: {metrics['volatility']*100:.2f}%",
        f"  • Final Balance: ${metrics['final_balance']:.2f}",
        "",
        "🔹 TRADING STATISTICS:",
        f"  • Total Trades: {trade_metrics['total_trades']}",
        f"  • Win Rate: {trade_metrics['win_rate']*100:.2f}%",
        f"  • Profit Factor: {trade_metrics['profit_factor']:.2f}",
        f"  • Avg. Profit per Trade: ${trade_metrics['avg_profit_per_trade']:.2f}",
        f"  • Avg. Trade Duration: {trade_metrics['avg_trade_duration']:.2f} periods",
        ""
    ]
    
    # Add benchmark comparison if provided
    if benchmark_metrics:
        outperformance = metrics['total_return'] - benchmark_metrics['total_return']
        report.extend([
            "🔹 BENCHMARK COMPARISON (BUY & HOLD):",
            f"  • Benchmark Return: {benchmark_metrics['total_return']*100:.2f}%",
            f"  • Outperformance: {outperformance*100:.2f}%",
            f"  • Benchmark Sharpe: {benchmark_metrics['sharpe_ratio']:.2f}",
            f"  • Benchmark Max Drawdown: {benchmark_metrics['max_drawdown']*100:.2f}%",
            ""
        ])
    
    report.append("=" * 50)
    
    return "\n".join(report)
