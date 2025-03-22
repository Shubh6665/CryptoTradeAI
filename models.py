import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from trading_env import CryptoTradingEnv
from config import TRAINING_TIMESTEPS, LEARNING_RATE, MODEL_SAVE_PATH
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_training_env(train_df):
    """
    Create a training environment
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        Vectorized gym environment
    """
    def _init():
        env = CryptoTradingEnv(train_df)
        return env
        
    env = DummyVecEnv([_init])
    return env

def create_testing_env(test_df):
    """
    Create a testing environment
    
    Args:
        test_df: Testing DataFrame
        
    Returns:
        Trading environment
    """
    env = CryptoTradingEnv(test_df)
    return env

def train_models(train_df, test_df, models=["SAC", "A2C", "PPO"], timesteps=TRAINING_TIMESTEPS):
    """
    Train multiple RL models
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        models: List of model types to train
        timesteps: Number of training timesteps
        
    Returns:
        Dictionary of trained models
    """
    # Create environments
    train_env = create_training_env(train_df)
    test_env = create_testing_env(test_df)
    
    trained_models = {}
    
    for model_type in models:
        logger.info(f"🚀 Training {model_type} model...")
        
        # Create model directory
        model_dir = os.path.join(MODEL_SAVE_PATH, model_type.lower())
        os.makedirs(model_dir, exist_ok=True)
        
        # Create callback for saving best model
        save_path = os.path.join(model_dir, "best_model")
        callback = EvalCallback(
            test_env,
            best_model_save_path=save_path,
            log_path=model_dir,
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        try:
            # Initialize and train the model
            if model_type == "SAC":
                model = SAC(
                    "MlpPolicy", 
                    train_env, 
                    learning_rate=LEARNING_RATE,
                    verbose=1,
                    tensorboard_log=os.path.join(model_dir, "tensorboard")
                )
            elif model_type == "PPO":
                model = PPO(
                    "MlpPolicy", 
                    train_env, 
                    learning_rate=LEARNING_RATE,
                    verbose=1,
                    tensorboard_log=os.path.join(model_dir, "tensorboard")
                )
            elif model_type == "A2C":
                model = A2C(
                    "MlpPolicy", 
                    train_env, 
                    learning_rate=LEARNING_RATE,
                    verbose=1,
                    tensorboard_log=os.path.join(model_dir, "tensorboard")
                )
            else:
                logger.error(f"❌ Unknown model type: {model_type}")
                continue
            
            # Train the model
            model.learn(total_timesteps=timesteps, callback=callback)
            
            # Save the final model
            final_model_path = os.path.join(model_dir, "final_model")
            model.save(final_model_path)
            logger.info(f"✅ {model_type} model trained and saved to {final_model_path}")
            
            # Store the model
            trained_models[model_type] = model
        
        except Exception as e:
            logger.error(f"❌ Error training {model_type} model: {str(e)}")
    
    return trained_models

def load_model(model_type, path=None):
    """
    Load a trained model
    
    Args:
        model_type: Type of model to load (SAC, PPO, A2C)
        path: Path to the model file, if None uses default
        
    Returns:
        Loaded model
    """
    try:
        if path is None:
            # Try loading best model, fall back to final model
            model_dir = os.path.join(MODEL_SAVE_PATH, model_type.lower())
            best_model_path = os.path.join(model_dir, "best_model", "best_model.zip")
            final_model_path = os.path.join(model_dir, "final_model.zip")
            
            if os.path.exists(best_model_path):
                path = best_model_path
            elif os.path.exists(final_model_path):
                path = final_model_path
            else:
                logger.error(f"❌ No saved model found for {model_type}")
                return None
        
        # Load the appropriate model type
        if model_type == "SAC":
            model = SAC.load(path)
        elif model_type == "PPO":
            model = PPO.load(path)
        elif model_type == "A2C":
            model = A2C.load(path)
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            return None
        
        logger.info(f"✅ Successfully loaded {model_type} model from {path}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return None

def evaluate_model(model, test_env, n_eval_episodes=10):
    """
    Evaluate a model on a test environment
    
    Args:
        model: Trained model
        test_env: Test environment
        n_eval_episodes: Number of episodes to run
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"🔍 Evaluating model on {n_eval_episodes} episodes...")
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    portfolio_values = []
    trades_executed = []
    sharpe_ratios = []
    profit_factors = []
    
    for episode in range(n_eval_episodes):
        state = test_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
        
        # Extract metrics from the environment
        returns = test_env.returns
        trade_profits = test_env.trade_profits
        
        # Calculate performance metrics
        if returns:
            returns_std = np.std(returns) if len(returns) > 1 else 1
            sharpe_ratio = np.mean(returns) / returns_std if returns_std != 0 else 0
            sharpe_ratios.append(sharpe_ratio)
        
        if trade_profits:
            profitable_trades = [p for p in trade_profits if p > 0]
            losing_trades = [p for p in trade_profits if p <= 0]
            
            if losing_trades:
                profit_factor = sum(profitable_trades) / abs(sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
            else:
                profit_factor = float('inf') if profitable_trades else 0
                
            profit_factors.append(profit_factor)
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        portfolio_values.append(test_env.net_worth)
        trades_executed.append(len(test_env.trades))
        
        logger.info(f"Episode {episode+1}/{n_eval_episodes} | Reward: {episode_reward:.2f} | Final Portfolio: ${test_env.net_worth:.2f}")
    
    # Compute mean metrics
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_portfolio = np.mean(portfolio_values)
    mean_trades = np.mean(trades_executed)
    mean_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
    mean_profit_factor = np.mean(profit_factors) if profit_factors else 0
    
    # Print evaluation summary
    logger.info(f"✅ Evaluation complete")
    logger.info(f"📊 Mean reward: {mean_reward:.2f}")
    logger.info(f"💰 Mean final portfolio: ${mean_portfolio:.2f}")
    logger.info(f"📈 Mean Sharpe ratio: {mean_sharpe:.2f}")
    logger.info(f"📊 Mean profit factor: {mean_profit_factor:.2f}")
    logger.info(f"🔄 Mean trades executed: {mean_trades:.2f}")
    
    # Return evaluation metrics
    return {
        "mean_reward": mean_reward,
        "mean_portfolio": mean_portfolio,
        "mean_sharpe": mean_sharpe,
        "mean_profit_factor": mean_profit_factor,
        "mean_trades": mean_trades,
        "all_rewards": episode_rewards,
        "all_portfolios": portfolio_values,
        "all_sharpes": sharpe_ratios,
        "all_profit_factors": profit_factors
    }

def compare_models(models, test_env, n_eval_episodes=10):
    """
    Compare multiple models on the same test environment
    
    Args:
        models: Dictionary of trained models
        test_env: Test environment
        n_eval_episodes: Number of episodes to run per model
        
    Returns:
        Dictionary of evaluation results, best model
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"🔍 Evaluating {model_name} model...")
        
        # Evaluate the model
        eval_metrics = evaluate_model(model, test_env, n_eval_episodes)
        
        # Store results
        results[model_name] = eval_metrics
    
    # Determine best model based on Sharpe ratio
    if results:
        best_model_name = max(results, key=lambda k: results[k]["mean_sharpe"])
        best_model = models[best_model_name]
        
        logger.info(f"🏆 Best model: {best_model_name} with Sharpe ratio: {results[best_model_name]['mean_sharpe']:.2f}")
        
        return results, best_model_name, models[best_model_name]
    else:
        logger.warning("⚠️ No models to compare")
        return results, None, None

def visualize_model_comparison(comparison_results):
    """
    Visualize the comparison between different models
    
    Args:
        comparison_results: Results from compare_models function
    """
    if not comparison_results:
        logger.warning("⚠️ No results to visualize")
        return
    
    # Extract metrics for each model
    models = list(comparison_results.keys())
    sharpe_ratios = [comparison_results[m]["mean_sharpe"] for m in models]
    profit_factors = [comparison_results[m]["mean_profit_factor"] for m in models]
    final_portfolios = [comparison_results[m]["mean_portfolio"] for m in models]
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot Sharpe ratio
    axs[0].bar(models, sharpe_ratios, color='blue')
    axs[0].set_title('Mean Sharpe Ratio by Model')
    axs[0].set_ylabel('Sharpe Ratio')
    axs[0].grid(True, alpha=0.3)
    
    # Plot profit factor
    axs[1].bar(models, profit_factors, color='green')
    axs[1].set_title('Mean Profit Factor by Model')
    axs[1].set_ylabel('Profit Factor')
    axs[1].grid(True, alpha=0.3)
    
    # Plot final portfolio value
    axs[2].bar(models, final_portfolios, color='purple')
    axs[2].set_title('Mean Final Portfolio Value by Model')
    axs[2].set_ylabel('Portfolio Value ($)')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(MODEL_SAVE_PATH, "model_comparison.png")
    plt.savefig(fig_path)
    logger.info(f"✅ Model comparison visualization saved to {fig_path}")
    
    return fig
