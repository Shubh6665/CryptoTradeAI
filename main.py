import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from data_utils import initialize_binance_client, fetch_historical_data, add_technical_indicators, prepare_train_test_data, visualize_data
from trading_env import CryptoTradingEnv
from models import train_models, evaluate_model, compare_models, visualize_model_comparison
from live_trading import LiveTrader
from evaluation import calculate_performance_metrics, analyze_trades, plot_portfolio_performance, plot_trade_analysis, generate_performance_report
from config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, TIMEFRAME, MODELS_TO_TRAIN, TRAINING_TIMESTEPS, INITIAL_BALANCE, MODEL_SAVE_PATH

# Page configuration
st.set_page_config(
    page_title="RL Crypto Trading Bot",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📈 Reinforcement Learning Crypto Trading Bot")
st.markdown("""
This application trains reinforcement learning models (SAC, A2C, PPO) on historical cryptocurrency data
and provides a dashboard for monitoring live trading performance.
""")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'trading_status' not in st.session_state:
    st.session_state.trading_status = {}
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Function to check Binance API credentials
def check_api_credentials():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        st.error("❌ Binance API credentials not found in environment variables. Please set them before proceeding.")
        return False
    return True

# Function to update trading status
def update_trading_status():
    while st.session_state.trader and st.session_state.trader.is_trading_active:
        try:
            st.session_state.trading_status = st.session_state.trader.get_trading_status()
            time.sleep(2)
        except Exception as e:
            st.error(f"Error updating trading status: {str(e)}")
            time.sleep(10)

# Training function (runs in a separate thread)
def train_models_thread(train_df, test_df, models_to_train, timesteps):
    try:
        st.session_state.is_training = True
        
        # Train models
        st.session_state.models = train_models(train_df, test_df, models=models_to_train, timesteps=timesteps)
        
        # Compare models
        if st.session_state.models:
            test_env = CryptoTradingEnv(test_df)
            st.session_state.comparison_results, st.session_state.best_model_name, st.session_state.best_model = compare_models(
                st.session_state.models, 
                test_env
            )
            
            # Get evaluation results for the best model
            if st.session_state.best_model:
                st.session_state.evaluation_results = evaluate_model(
                    st.session_state.best_model,
                    test_env,
                    n_eval_episodes=5
                )
        
        st.session_state.training_complete = True
    except Exception as e:
        st.error(f"Training error: {str(e)}")
    finally:
        st.session_state.is_training = False
        st.rerun()

# Sidebar navigation
st.sidebar.title("Navigation")
steps = [
    "1. Data Collection", 
    "2. Model Training", 
    "3. Model Evaluation",
    "4. Live Trading"
]
selected_step = st.sidebar.radio("Go to step:", steps, index=st.session_state.step-1)
st.session_state.step = steps.index(selected_step) + 1

# Sidebar info
with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("""
    ### About this app
    
    This application uses deep reinforcement learning to trade cryptocurrencies. 
    It trains models on historical data and can execute trades via the Binance API.
    
    **Key Features:**
    - Fetches and preprocesses historical data
    - Trains multiple RL models (SAC, A2C, PPO)
    - Evaluates model performance with key metrics
    - Provides real-time trading dashboard
    - Implements risk management strategies
    
    Developed by RL Trader
    """)

# Step 1: Data Collection
if st.session_state.step == 1:
    st.header("Step 1: Historical Data Collection")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Data Parameters")
        symbol = st.text_input("Trading Pair", SYMBOL)
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        data_limit = st.number_input("Number of Candles", min_value=100, max_value=1000, value=500, step=100)
        
        if st.button("Fetch Historical Data"):
            if check_api_credentials():
                with st.spinner("Fetching historical data from Binance..."):
                    try:
                        # Initialize client
                        client = initialize_binance_client()
                        
                        # Fetch data
                        df = fetch_historical_data(client, symbol=symbol, interval=timeframe, limit=data_limit)
                        
                        # Store data in session state
                        st.session_state.historical_data = df
                        
                        # Prepare train/test data
                        st.session_state.train_data, st.session_state.test_data = prepare_train_test_data(df)
                        
                        st.success(f"✅ Successfully fetched {len(df)} candles")
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
    
    with col1:
        if st.session_state.historical_data is not None:
            st.subheader("Historical Data Preview")
            
            # Display data table
            st.dataframe(st.session_state.historical_data.head())
            
            # Display data visualization
            st.subheader("Price Chart")
            fig = visualize_data(st.session_state.historical_data)
            st.pyplot(fig)
            
            st.success(f"✅ Data is ready for model training")
            if st.button("Proceed to Model Training"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.info("Please fetch historical data to proceed.")

# Step 2: Model Training
elif st.session_state.step == 2:
    st.header("Step 2: Model Training")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Training Parameters")
        models_to_train = st.multiselect(
            "Models to Train", 
            ["SAC", "PPO", "A2C"], 
            default=MODELS_TO_TRAIN
        )
        timesteps = st.number_input(
            "Training Timesteps", 
            min_value=1000, 
            max_value=100000, 
            value=TRAINING_TIMESTEPS, 
            step=1000
        )
        
        # Start training button
        if st.button("Start Training"):
            if st.session_state.train_data is None or st.session_state.test_data is None:
                st.error("⚠️ Please collect historical data first!")
            else:
                # Start training in a separate thread
                training_thread = threading.Thread(
                    target=train_models_thread,
                    args=(st.session_state.train_data, st.session_state.test_data, models_to_train, timesteps)
                )
                training_thread.daemon = True
                training_thread.start()
                st.info("🚀 Training started! This may take several minutes...")
    
    with col1:
        if st.session_state.is_training:
            st.subheader("Training in Progress")
            st.warning("⚠️ Please wait while the models are being trained. This may take several minutes.")
            
            # Add a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                # Update the progress bar
                progress_bar.progress(i + 1)
                time.sleep(0.1)
                
            st.info("Training is still in progress. Please wait...")
            
        elif st.session_state.training_complete:
            st.subheader("Training Complete")
            
            if st.session_state.best_model_name:
                st.success(f"✅ Best model: {st.session_state.best_model_name}")
                
                # Display comparison visualization
                if st.session_state.comparison_results:
                    st.subheader("Model Comparison")
                    fig = visualize_model_comparison(st.session_state.comparison_results)
                    st.pyplot(fig)
                
                if st.button("Proceed to Model Evaluation"):
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.error("❌ Training failed or no models were selected.")
        else:
            st.info("Click 'Start Training' to begin the model training process.")

# Step 3: Model Evaluation
elif st.session_state.step == 3:
    st.header("Step 3: Model Evaluation")
    
    if not st.session_state.best_model or not st.session_state.evaluation_results:
        st.warning("⚠️ No trained model available. Please complete the training step first.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Performance Metrics for {st.session_state.best_model_name} Model")
            
            # Create metrics display
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("Sharpe Ratio", f"{st.session_state.evaluation_results['mean_sharpe']:.2f}")
            
            with metrics_cols[1]:
                return_pct = (st.session_state.evaluation_results['mean_portfolio'] / INITIAL_BALANCE - 1) * 100
                st.metric("Return", f"{return_pct:.2f}%")
            
            with metrics_cols[2]:
                st.metric("Profit Factor", f"{st.session_state.evaluation_results['mean_profit_factor']:.2f}")
            
            with metrics_cols[3]:
                st.metric("Avg Trades", f"{st.session_state.evaluation_results['mean_trades']:.0f}")
            
            # Create visualization for model performance
            st.subheader("Portfolio Performance")
            
            # Create a line chart for portfolio values
            all_portfolios = st.session_state.evaluation_results.get('all_portfolios', [])
            if all_portfolios:
                # Create a Plotly chart
                fig = go.Figure()
                
                # Add a line for each episode
                for i, portfolio in enumerate(all_portfolios):
                    fig.add_trace(go.Scatter(
                        y=portfolio,
                        mode='lines',
                        name=f'Episode {i+1}',
                        line=dict(width=2)
                    ))
                
                # Add initial balance reference line
                fig.add_hline(
                    y=INITIAL_BALANCE, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Initial Balance"
                )
                
                # Update layout
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Time Steps',
                    yaxis_title='Portfolio Value ($)',
                    legend_title='Episodes',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            st.subheader("Evaluation Summary")
            
            # Display additional metrics
            st.info(f"""
            **Model Type:** {st.session_state.best_model_name}
            
            **Key Metrics:**
            - Mean Reward: {st.session_state.evaluation_results['mean_reward']:.2f}
            - Mean Portfolio: ${st.session_state.evaluation_results['mean_portfolio']:.2f}
            - Mean Sharpe: {st.session_state.evaluation_results['mean_sharpe']:.2f}
            - Mean Profit Factor: {st.session_state.evaluation_results['mean_profit_factor']:.2f}
            - Mean Trades: {st.session_state.evaluation_results['mean_trades']:.0f}
            """)
            
            if st.button("Proceed to Live Trading"):
                st.session_state.step = 4
                st.rerun()

# Step 4: Live Trading
elif st.session_state.step == 4:
    st.header("Step 4: Live Trading")
    
    if not st.session_state.best_model:
        st.warning("⚠️ No trained model available. Please complete the training and evaluation steps first.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Trading Controls")
            
            # Display trading status if available
            if st.session_state.trader:
                trader_status = st.session_state.trading_status
                
                # Current status
                status_color = "🟢" if trader_status.get('is_active', False) else "🔴"
                st.write(f"{status_color} Trading Status: {'Active' if trader_status.get('is_active', False) else 'Inactive'}")
                
                # Position info
                if trader_status.get('position', 0) > 0:
                    st.write(f"📊 Current Position: {trader_status.get('position', 0):.6f} BTC")
                    st.write(f"💲 Entry Price: ${trader_status.get('entry_price', 0):.2f}")
                else:
                    st.write("📊 Position: No open position")
                
                # Portfolio info
                st.write(f"💰 Portfolio Value: ${trader_status.get('portfolio_value', 0):.2f}")
                st.write(f"🔄 Trades Executed: {trader_status.get('trade_count', 0)}")
                
                # Risk metrics
                risk_metrics = trader_status.get('risk_metrics', {})
                if risk_metrics:
                    st.write(f"📈 Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
                    st.write(f"📉 Max Drawdown: {risk_metrics.get('max_drawdown', 0)*100:.2f}%")
            
            # Start/Stop trading buttons
            if st.session_state.trader and st.session_state.trader.is_trading_active:
                if st.button("Stop Trading"):
                    st.session_state.trader.stop_trading()
                    st.success("Trading stopped!")
                    st.rerun()
            else:
                if st.button("Start Trading"):
                    if not check_api_credentials():
                        st.error("❌ API credentials not found. Please set them before trading.")
                    else:
                        try:
                            # Initialize live trader if needed
                            if not st.session_state.trader:
                                st.session_state.trader = LiveTrader(st.session_state.best_model)
                            
                            # Start trading
                            st.session_state.trader.start_trading()
                            
                            # Start status update thread
                            status_thread = threading.Thread(target=update_trading_status)
                            status_thread.daemon = True
                            status_thread.start()
                            
                            st.success("Trading started!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error starting trading: {str(e)}")
        
        with col1:
            st.subheader("Trading Dashboard")
            
            # Set up placeholders for live updates
            price_chart = st.empty()
            portfolio_chart = st.empty()
            trade_history = st.empty()
            
            # If trading is active, show live charts
            if st.session_state.trader and st.session_state.trader.is_trading_active:
                # Get trading data
                trader_status = st.session_state.trading_status
                portfolio_history = st.session_state.trader.portfolio_history if st.session_state.trader else []
                trades = st.session_state.trader.trades if st.session_state.trader else []
                
                # Create charts
                if portfolio_history:
                    # Portfolio value chart
                    fig_portfolio = go.Figure()
                    fig_portfolio.add_trace(go.Scatter(
                        y=portfolio_history,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Add initial balance reference line
                    fig_portfolio.add_hline(
                        y=INITIAL_BALANCE, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Initial Balance"
                    )
                    
                    # Update layout
                    fig_portfolio.update_layout(
                        title='Portfolio Value',
                        xaxis_title='Time',
                        yaxis_title='Value ($)',
                        template='plotly_dark',
                        height=300
                    )
                    
                    # Show chart
                    portfolio_chart.plotly_chart(fig_portfolio, use_container_width=True)
                
                # Show trade history table
                if trades:
                    trade_df = pd.DataFrame(trades)
                    trade_history.dataframe(trade_df)
                else:
                    trade_history.info("No trades executed yet.")
            else:
                price_chart.info("Start trading to see real-time price data.")
                portfolio_chart.info("Start trading to see portfolio value updates.")
                trade_history.info("Start trading to see trade history.")

# Footer
st.markdown("---")
st.markdown("*Developed with ❤️ by RL Trader | Powered by Stable-Baselines3 and Binance API*")
