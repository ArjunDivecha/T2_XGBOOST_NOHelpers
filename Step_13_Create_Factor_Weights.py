"""
Portfolio Optimization with External Expected Returns

INPUT FILES:
1. T2_Optimizer.xlsx
   - Monthly returns data for historical covariance estimation
   - Format: Excel workbook with single sheet
   - Index: Dates in datetime format
   - Columns: Factor strategy returns
   - Values: Returns in decimal or percentage format
   - Note: 'Monthly Return_CS' column will be removed if present
2. Factor_Alpha.xlsx (Main input for standalone mode)
   - Expected returns (annualized alphas) for each factor
   - Format: Excel workbook with expected returns for each factor
   - Index: Dates in datetime format (must align with historical data dates)
   - Columns: Factor names matching those in T2_Optimizer.xlsx
   - Values: Expected returns/alphas (annualized, not monthly)

OUTPUT FILES:
1. output/S13_rolling_window_weights.xlsx
   - Hybrid window optimization results based on external expected returns
   - Format: Excel workbook with single sheet
   - Index: Dates (monthly)
   - Columns: Factor names
   - Values: Portfolio weights (0-1)

2. output/S13_strategy_statistics.xlsx
   - Performance statistics using historical returns for evaluation
   - Format: Excel workbook with multiple sheets
   - Sheet 1: Summary statistics - Returns, volatility, Sharpe ratio, turnover, etc.
   - Sheet 2: Monthly Returns - Time series of monthly returns

3. output/S13_turnover_analysis.pdf
   - Monthly portfolio turnover visualization
   - Format: PDF
   - Shows turnover over time

4. output/S13_strategy_performance.pdf
   - Cumulative performance visualization
   - Format: PDF
   - Shows cumulative returns based on historical data

This module implements portfolio optimization using externally provided expected returns
while estimating covariance from historical data. When run as a standalone program, it:

1. Reads expected returns (alphas) from Factor_Alpha.xlsx
2. Reads historical returns from T2_Optimizer.xlsx for covariance estimation
3. Uses a hybrid window approach for covariance:
   - First 60 months: Uses all available data (expanding window)
   - After 60 months: Uses exactly 60 months of data (rolling window)
4. Optimizes portfolio weights for each date where expected returns are provided
5. Evaluates performance using historical returns from T2_Optimizer.xlsx

Key Parameters:
- Window length: 5 years (60 months) for rolling window portion
- Risk aversion (λ): 50.0
- HHI penalty: 0.1 - penalizes portfolio concentration
- Constraints: Long-only, fully invested

Dependencies:
- numpy
- pandas
- scipy.optimize
- matplotlib

Version: 2.0
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# ================= USER-EDITABLE RISK AVERSION PARAMETER =====================
# Set the risk aversion parameter for all portfolio optimizations here:
LAMBDA_PARAM = 200.0  # Higher values reduce risk-taking (default: 50.0)
# ============================================================================

class PortfolioOptimizer:
    """
    Portfolio optimization class.

    Attributes:
        returns (pd.DataFrame): Asset returns data.
        lambda_param (float): Risk aversion parameter.
        hhi_penalty (float): Penalty coefficient for HHI concentration.
        n_assets (int): Number of assets in the portfolio.
    """

    def __init__(self, returns_df: pd.DataFrame, hhi_penalty: float = 5.0):
        """Initialize with risk aversion parameter from LAMBDA_PARAM."""
        self.returns = returns_df
        self.lambda_param = LAMBDA_PARAM
        self.hhi_penalty = hhi_penalty
        self.n_assets = len(returns_df.columns)
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame = None) -> tuple:
        """
        Calculate portfolio metrics for given weights and optional returns data.

        Args:
            weights (np.ndarray): Portfolio weights.
            returns (pd.DataFrame, optional): Returns data. Defaults to None.

        Returns:
            tuple: Average return and volatility.
        """
        if returns is None:
            returns = self.returns
            
        portfolio_returns = np.sum(returns * weights, axis=1)
        avg_return = (1 + np.mean(portfolio_returns))**12 - 1
        volatility = np.std(portfolio_returns) * np.sqrt(12)
        
        # Redefine avg_return as 8 * avg_return / volatility
        avg_return = 8 * avg_return 
        
        return avg_return, volatility
    
    def objective_function(self, weights: np.ndarray) -> float:
        """
        Calculate negative utility (including HHI penalty).

        Args:
            weights (np.ndarray): Portfolio weights.

        Returns:
            float: Negative utility (including HHI penalty).
        """
        avg_return, volatility = self.calculate_portfolio_metrics(weights)
        utility = avg_return - self.lambda_param * (volatility ** 2)
        # HHI = sum of squared weights
        hhi = np.sum(weights**2)
        # Penalize utility by HHI * penalty coefficient
        penalized_utility = utility - self.hhi_penalty * hhi
        return -penalized_utility  # Return negative penalized utility for minimization
    
    def optimize_weights(self, returns: pd.DataFrame = None) -> np.ndarray:
        """
        Optimize weights for given returns data.

        Args:
            returns (pd.DataFrame, optional): Returns data. Defaults to None.

        Returns:
            np.ndarray: Optimized weights.
        """
        if returns is not None:
            original_returns = self.returns
            self.returns = returns
            
        initial_weights = np.ones(self.n_assets) / self.n_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1.0) for _ in range(self.n_assets))  # Allow weights up to 100% in a single asset
        
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if returns is not None:
            self.returns = original_returns
            
        return result.x

def apply_exponential_weights(returns_df: pd.DataFrame, decay_factor: float = 0.94) -> pd.DataFrame:
    """
    Apply exponential weights to returns data.
    
    Args:
        returns_df (pd.DataFrame): Historical returns data.
        decay_factor (float): Decay factor for exponential weighting (0 < decay_factor < 1).
                              Higher values give more weight to recent observations.
                              Default is 0.94 (approximately 12-month half-life).
    
    Returns:
        pd.DataFrame: Returns data with exponential weights applied.
    """
    # Get the number of observations
    n_obs = len(returns_df)
    
    # Create exponential weights (newest to oldest)
    # Formula: w_i = decay_factor^i / sum(decay_factor^i)
    weights = np.array([decay_factor ** i for i in range(n_obs)])
    
    # Reverse weights so most recent observations get highest weight
    weights = weights[::-1]
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Apply weights to each row of returns
    weighted_returns = returns_df.copy()
    for i, (_, row) in enumerate(returns_df.iterrows()):
        weighted_returns.iloc[i] = row * weights[i]
    
    return weighted_returns, weights

def calculate_exponential_forecast(returns_df: pd.DataFrame, decay_factor: float = 0.94) -> pd.Series:
    """
    Calculate exponentially weighted forecast returns.
    
    Args:
        returns_df (pd.DataFrame): Historical returns data.
        decay_factor (float): Decay factor for exponential weighting.
        
    Returns:
        pd.Series: Exponentially weighted forecast returns.
    """
    # Get the number of observations
    n_obs = len(returns_df)
    
    # Create exponential weights (newest to oldest)
    weights = np.array([decay_factor ** i for i in range(n_obs)])
    
    # Reverse weights so most recent observations get highest weight
    weights = weights[::-1]
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Calculate weighted average for each column
    weighted_means = pd.Series(index=returns_df.columns, dtype=float)
    
    for col in returns_df.columns:
        # Check for NaN values in this column
        if returns_df[col].isna().any():
            # Use only non-NaN values for this column
            valid_data = returns_df[col].dropna()
            if len(valid_data) > 0:
                # Recalculate weights for the valid data length
                valid_n = len(valid_data)
                valid_weights = np.array([decay_factor ** i for i in range(valid_n)])
                valid_weights = valid_weights[::-1]
                valid_weights = valid_weights / valid_weights.sum()
                
                # Calculate weighted average using only valid data
                weighted_means[col] = np.sum(valid_data.values * valid_weights)
            else:
                weighted_means[col] = np.nan
        else:
            # Calculate the weighted average for this column
            weighted_means[col] = np.sum(returns_df[col].values * weights)
    
    # Annualize the weighted means
    annualized_means = (1 + weighted_means) ** 12 - 1
    
    return annualized_means

def calculate_portfolio_statistics(returns: pd.Series) -> dict:
    """
    Calculate comprehensive portfolio statistics.

    Args:
        returns (pd.Series): Portfolio returns.

    Returns:
        dict: Portfolio statistics.
    """
    # Annualize metrics
    ann_return = (1 + returns.mean())**12 - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Monthly statistics
    positive_months = (returns > 0).mean()
    
    return {
        'Annualized Return (%)': ann_return * 100,
        'Annualized Volatility (%)': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Maximum Drawdown (%)': max_drawdown * 100,
        'Positive Months (%)': positive_months * 100,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }

def calculate_turnover(weights_df: pd.DataFrame) -> tuple:
    """
    Calculate average monthly turnover and turnover series.

    Args:
        weights_df (pd.DataFrame): Portfolio weights over time.

    Returns:
        tuple: Average monthly turnover and monthly turnover series.
    """
    # Calculate absolute weight changes for each month
    weight_changes = weights_df.diff().abs()
    
    # Sum across assets to get total turnover for each month
    monthly_turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 as each trade affects two positions
    
    # Calculate average monthly turnover
    avg_monthly_turnover = monthly_turnover.mean()
    
    return avg_monthly_turnover, monthly_turnover

def optimize_portfolio_with_external_returns(
    expected_returns_file: str = 'Factor_Alpha.xlsx',
    expected_returns_df: pd.DataFrame = None,
    hhi_penalty: float = 0.1,
    initial_window: int = 60
) -> tuple:
    """
    Optimizes portfolio weights using externally provided expected returns and hybrid methodology
    for covariance estimation.
    
    This function reads historical returns from T2_Optimizer.xlsx for covariance estimation
    but uses externally provided expected returns for determining optimal portfolio weights.
    This separation allows using forward-looking or custom expected return forecasts
    while maintaining a robust covariance structure from historical data.
    
    Args:
        expected_returns_file (str, optional): Path to Excel file with expected returns/alphas.
                                               Defaults to 'Factor_Alpha.xlsx'.
        expected_returns_df (pd.DataFrame, optional): DataFrame containing expected returns.
                                                    Alternative to providing file path.
        hhi_penalty (float): Herfindahl-Hirschman Index penalty coefficient to control
                             portfolio concentration. Defaults to 0.1.
        initial_window (int): Number of months for initial window before switching
                              to rolling window approach. Defaults to 60 (5 years).
    
    Returns:
        tuple: (weights_df, portfolio_returns, strategy_stats)
            - weights_df: DataFrame of optimized weights
            - portfolio_returns: Series of portfolio returns
            - strategy_stats: Dictionary of strategy statistics
    
    Notes:
        - Expected returns should be annualized (not monthly)
        - Historical returns will be read from T2_Optimizer.xlsx for covariance and performance
        - If both expected_returns_file and expected_returns_df are provided, the DataFrame takes precedence
        - The function will optimize for all dates in the expected returns dataset that have
          corresponding historical data
        - Performance evaluation uses historical returns, not the expected returns
    """
    print("Starting portfolio optimization with external expected returns...")
    
    # Load historical data for covariance estimation
    hist_returns = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    hist_returns.index = pd.to_datetime(hist_returns.index)
    
    # Convert historical returns to decimals if needed
    if hist_returns.abs().mean().mean() > 1:
        hist_returns = hist_returns / 100
    
    # Remove Monthly Return_CS if present
    if 'Monthly Return_CS' in hist_returns.columns:
        hist_returns = hist_returns.drop(columns=['Monthly Return_CS'])
    
    # Get expected returns either from file or DataFrame
    if expected_returns_df is None and expected_returns_file is not None:
        # Load expected returns from file
        expected_returns_df = pd.read_excel(expected_returns_file, index_col=0)
    elif expected_returns_df is None and expected_returns_file is None:
        raise ValueError("Either expected_returns_file or expected_returns_df must be provided")
    
    # Ensure index is datetime
    expected_returns_df.index = pd.to_datetime(expected_returns_df.index)
    
    # Verify that expected returns columns match historical returns columns
    missing_cols = set(hist_returns.columns) - set(expected_returns_df.columns)
    if missing_cols:
        print(f"Warning: The following columns from historical returns are missing in expected returns: {missing_cols}")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(hist_returns, hhi_penalty)
    
    # Prepare dates - we will optimize for dates that exist in both datasets
    hist_dates = hist_returns.index
    
    # Initialize storage for weights
    hybrid_weights = pd.DataFrame(0, index=expected_returns_df.index, columns=hist_returns.columns)
    
    print("\nRunning hybrid window optimizations...")
    # Run optimizations for each date where we have expected returns
    for current_date in expected_returns_df.index:
        if current_date not in hist_dates:
            print(f"Warning: {current_date.strftime('%Y-%m-%d')} not found in historical data. Skipping.")
            continue
            
        current_idx = hist_dates.get_loc(current_date)
        
        # Skip if we don't have enough historical data for covariance
        if current_idx < 2:
            print(f"Warning: Not enough historical data before {current_date.strftime('%Y-%m-%d')}. Skipping.")
            continue
        
        # Get historical data for covariance estimation using hybrid window approach
        if current_idx <= initial_window:
            # Use expanding window for the first initial_window months
            hist_window = hist_returns.iloc[:current_idx]
        else:
            # Switch to rolling window after initial_window months
            start_idx = current_idx - initial_window
            hist_window = hist_returns.iloc[start_idx:current_idx]
            
        # Custom objective function to use external expected returns
        def custom_objective(weights):
            # Calculate volatility using historical window
            portfolio_returns = np.sum(hist_window * weights, axis=1)
            volatility = np.std(portfolio_returns) * np.sqrt(12)
            
            # Get the expected returns for the current date
            expected_return = np.sum(expected_returns_df.loc[current_date] * weights)
            
            # Use the same scaling as in the original objective function
            expected_return = 8 * expected_return
            
            # Calculate utility with HHI penalty
            utility = expected_return - LAMBDA_PARAM * (volatility ** 2)
            hhi = np.sum(weights**2)
            penalized_utility = utility - hhi_penalty * hhi
            
            return -penalized_utility  # Negative for minimization
        
        # Store the original objective function
        original_objective = optimizer.objective_function
        
        # Set the custom objective function
        optimizer.objective_function = custom_objective
        
        # Optimize weights
        try:
            optimal_weights = optimizer.optimize_weights()
            hybrid_weights.loc[current_date] = optimal_weights
        except Exception as e:
            print(f"Error optimizing for {current_date.strftime('%Y-%m-%d')}: {e}")
            # Continue with the next date
        finally:
            # Restore the original objective function
            optimizer.objective_function = original_objective
        
        if (current_idx + 1) % 12 == 0:
            print(f"Processed up to {current_date.strftime('%Y-%m')}")
    
    # Calculate portfolio returns using the optimized weights
    portfolio_returns = pd.Series(index=hist_returns.index[1:], dtype=float)
    
    for i, date in enumerate(hist_returns.index[1:]):
        prev_date = hist_returns.index[i]
        # Find the closest weight date that's before or equal to prev_date
        available_weight_dates = hybrid_weights.index[hybrid_weights.index <= prev_date]
        if len(available_weight_dates) > 0:
            weight_date = available_weight_dates[-1]
            weights = hybrid_weights.loc[weight_date]
            portfolio_returns[date] = np.sum(weights * hist_returns.loc[date])
    
    # Remove dates where we couldn't calculate returns
    portfolio_returns = portfolio_returns.dropna()
    
    if len(portfolio_returns) == 0:
        print("Warning: No portfolio returns could be calculated. Check date alignment.")
        return hybrid_weights, pd.Series(), {}
    
    # Calculate statistics
    strategy_stats = calculate_portfolio_statistics(portfolio_returns)
    
    # Calculate turnover
    avg_turnover, turnover_series = calculate_turnover(hybrid_weights.loc[hybrid_weights.sum(axis=1) > 0])
    strategy_stats['Average Monthly Turnover (%)'] = avg_turnover * 100
    
    print("\nSaving results...")
    
    # Save weights
    hybrid_weights.to_excel('output/S13_rolling_window_weights.xlsx')
    
    # Save strategy statistics
    with pd.ExcelWriter('output/S13_strategy_statistics.xlsx', engine='openpyxl') as writer:
        # Create stats DataFrame
        stats_df = pd.DataFrame({
            'Hybrid Window': strategy_stats
        })
        stats_df.to_excel(writer, sheet_name='Summary Statistics')
        
        # Save monthly returns
        monthly_returns_df = pd.DataFrame({
            'Hybrid Window': portfolio_returns * 100  # Convert to percentage
        })
        monthly_returns_df.to_excel(writer, sheet_name='Monthly Returns')
    
    # Create turnover plot
    if len(turnover_series) > 0:
        plt.figure(figsize=(15, 8))
        plt.plot(turnover_series.index, turnover_series * 100, label='Hybrid Window', linewidth=2)
        plt.title('Monthly Portfolio Turnover', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Turnover (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='upper left')
        plt.figtext(0.02, 0.02, 
                    f'Avg. Monthly Turnover: {avg_turnover*100:.1f}%', 
                    fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig('output/S13_turnover_analysis.pdf')
    
    # Create performance plot
    cum_returns = (1 + portfolio_returns).cumprod()
    plt.figure(figsize=(15, 8))
    plt.plot(cum_returns, label='Hybrid Window', linewidth=2)
    plt.title('Cumulative Performance', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y-1)))
    plt.tight_layout()
    plt.savefig('output/S13_strategy_performance.pdf')
    
    print("\nOptimization complete. Files saved:")
    print("- output/S13_rolling_window_weights.xlsx: Portfolio weights")
    print("- output/S13_strategy_statistics.xlsx: Performance statistics")
    print("- output/S13_strategy_performance.pdf: Cumulative performance chart")
    print("- output/S13_turnover_analysis.pdf: Turnover analysis")
    
    return hybrid_weights, portfolio_returns, strategy_stats

def run_rolling_optimization():
    """
    Run portfolio optimization analysis comparing expanding window and hybrid window approaches.
    The hybrid approach uses expanding window until 60 months, then switches to rolling window.
    """
    # Load data
    print("Loading data...")
    returns = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    returns.index = pd.to_datetime(returns.index)
    
    # Convert to decimals if needed
    if returns.abs().mean().mean() > 1:
        returns = returns / 100
    
    # Remove Monthly Return_CS
    if 'Monthly Return_CS' in returns.columns:
        returns = returns.drop(columns=['Monthly Return_CS'])
    
    # Parameters
    HHI_PENALTY = 0.1  # HHI penalty coefficient (no penalty)
    INITIAL_WINDOW = 60  # 5 years of monthly data
    EMA_DECAY = 0.98  # Increased decay factor for longer tail (was 0.94)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns, HHI_PENALTY)
    
    # Prepare dates for rolling windows
    dates = returns.index
    
    # For rolling window, we can calculate one additional month at the end
    # Create a date for one month after the last date in the dataset
    next_month_date = dates[-1] + pd.DateOffset(months=1)
    
    # Initialize storage for weights and returns - start from month 2 to allow for returns calculation
    expanding_weights = pd.DataFrame(index=dates[1:], columns=returns.columns)
    hybrid_weights = pd.DataFrame(index=list(dates[1:]) + [next_month_date], columns=returns.columns)
    
    # Initialize DataFrame to hold hybrid-window factor forecasts
    hybrid_forecasts = pd.DataFrame(index=hybrid_weights.index, columns=returns.columns, dtype=float)
    # Initialize DataFrame to hold exponentially-weighted expanding window forecasts
    exp_forecasts = pd.DataFrame(index=dates[1:], columns=returns.columns, dtype=float)
    
    print("\nRunning optimizations...")
    # Run optimizations for each date
    for i, current_date in enumerate(dates[1:], 1):
        # Expanding window with exponential weighting - always use all available data
        expanding_data = returns.loc[:current_date]
        # Apply exponential weighting to the expanding window data
        weighted_expanding_data, _ = apply_exponential_weights(expanding_data, EMA_DECAY)
        expanding_weights.loc[current_date] = optimizer.optimize_weights(weighted_expanding_data)
        
        # Calculate exponentially-weighted forecasted returns
        exp_forecasts.loc[current_date] = calculate_exponential_forecast(expanding_data, EMA_DECAY)
        
        # Hybrid approach: expanding window until 60 months, then rolling 60-month window
        if i <= INITIAL_WINDOW:
            # Use expanding window for the first 60 months
            hybrid_data = returns.loc[dates[0]:current_date]
        else:
            # Switch to rolling 60-month window after 60 months
            start_idx = i - INITIAL_WINDOW  # Start INITIAL_WINDOW months back
            hybrid_data = returns.loc[dates[start_idx:i]]  # Use exactly 60 months
        
        # Optimize weights and store
        hybrid_weights.loc[current_date] = optimizer.optimize_weights(hybrid_data)
        
        # Compute factor-level forecasted returns for this hybrid window
        factor_means = hybrid_data.mean(axis=0)
        annualized_means = (1 + factor_means) ** 12 - 1
        hybrid_forecasts.loc[current_date] = annualized_means
        
        if i % 12 == 0:  # Print progress every year
            print(f"Processed up to {current_date.strftime('%Y-%m')}")
    
    # Calculate the extra month for hybrid strategy
    # Use the last 60 months of data for the next month
    start_idx = max(0, len(dates) - INITIAL_WINDOW)
    hybrid_data = returns.loc[dates[start_idx:]]
    hybrid_weights.loc[next_month_date] = optimizer.optimize_weights(hybrid_data)
    
    # Compute factor-level forecasted returns for this last hybrid window
    factor_means = hybrid_data.mean(axis=0)
    annualized_means = (1 + factor_means) ** 12 - 1
    hybrid_forecasts.loc[next_month_date] = annualized_means
    
    print(f"Processed extra month: {next_month_date.strftime('%Y-%m')}")
    
    # Calculate strategy returns - start from month 2 to have weights from month 1
    expanding_returns = pd.Series(index=dates[2:], dtype=float)
    hybrid_returns = pd.Series(index=dates[2:], dtype=float)
    
    for date in dates[2:]:
        prev_date = dates[dates < date][-1]
        # Get returns using previous month's weights
        expanding_returns[date] = np.sum(
            expanding_weights.loc[prev_date] * returns.loc[date]
        )
        hybrid_returns[date] = np.sum(
            hybrid_weights.loc[prev_date] * returns.loc[date]
        )
    
    # Calculate cumulative returns
    cum_expanding = (1 + expanding_returns).cumprod()
    cum_hybrid = (1 + hybrid_returns).cumprod()
    
    # Calculate statistics
    expanding_stats = calculate_portfolio_statistics(expanding_returns)
    hybrid_stats = calculate_portfolio_statistics(hybrid_returns)
    
    # Calculate turnover for both strategies
    avg_expanding_turnover, expanding_turnover = calculate_turnover(expanding_weights)
    avg_hybrid_turnover, hybrid_turnover = calculate_turnover(hybrid_weights.loc[dates[1:]])
    
    print("\nTurnover Statistics:")
    print(f"Expanding Window Average Monthly Turnover: {avg_expanding_turnover*100:.2f}%")
    print(f"Hybrid Window Average Monthly Turnover: {avg_hybrid_turnover*100:.2f}%")
    
    # Report on the transition from expanding to rolling window
    transition_date = dates[INITIAL_WINDOW] if INITIAL_WINDOW < len(dates) else None
    if transition_date:
        print(f"\nTransition from expanding to rolling window occurred at: {transition_date.strftime('%Y-%m')}")
        print(f"- Used expanding window from {dates[0].strftime('%Y-%m')} to {dates[INITIAL_WINDOW-1].strftime('%Y-%m')}")
        print(f"- Used 60-month rolling window from {transition_date.strftime('%Y-%m')} onwards")
    
    # Plot turnover over time
    plt.figure(figsize=(15, 8))
    plt.plot(expanding_turnover.index, expanding_turnover * 100, label='Expanding Window', linewidth=2)
    plt.plot(hybrid_turnover.index, hybrid_turnover * 100, label='Hybrid Window', linewidth=2)
    plt.title('Monthly Portfolio Turnover', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Turnover (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    # Add average turnover annotations
    plt.figtext(0.02, 0.02, 
                f'Avg. Monthly Turnover:\nExpanding Window: {avg_expanding_turnover*100:.1f}%\nHybrid Window: {avg_hybrid_turnover*100:.1f}%', 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('output/S13_turnover_analysis.pdf')
    plt.close()
    
    # Add turnover to statistics
    expanding_stats['Average Monthly Turnover (%)'] = avg_expanding_turnover * 100
    hybrid_stats['Average Monthly Turnover (%)'] = avg_hybrid_turnover * 100
    
    # Save weights
    expanding_weights.to_excel('output/S13_expanding_window_weights.xlsx')
    hybrid_weights.to_excel('output/S13_rolling_window_weights.xlsx')
    
    # --- Write enhanced T2_60_Month.xlsx as requested ---
    # Load the exact contents of T2_Optimizer.xlsx
    targets_df = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    
    # Check if data needs to be converted to decimals
    if targets_df.abs().mean().mean() > 1:
        targets_df_decimal = targets_df / 100
    else:
        targets_df_decimal = targets_df.copy()
    
    # Function to calculate trailing averages (annualized)
    def trailing_annualized(df, window):
        # Calculate trailing window mean
        trailing = df.rolling(window=window, min_periods=window).mean()
        # Annualize: (1 + mean)^12 - 1
        return (1 + trailing) ** 12 - 1
    
    # Compute trailing averages using decimal data, then shift by one month
    feature1 = trailing_annualized(targets_df_decimal, 1).shift(1)
    feature12 = trailing_annualized(targets_df_decimal, 12).shift(1)
    feature36 = trailing_annualized(targets_df_decimal, 36).shift(1)
    feature60 = trailing_annualized(targets_df_decimal, 60).shift(1)
    
    # Add one extra forecast month (next_month_date) for each feature sheet
    last_date = targets_df.index[-1]
    next_month_date = last_date + pd.DateOffset(months=1)
    
    # Calculate the extra forecast row for each feature
    for feat_df, window in zip([feature1, feature12, feature36, feature60], [1, 12, 36, 60]):
        # Get the last 'window' months of data
        last_window_data = targets_df_decimal.iloc[-window:]
        # Calculate mean of those months
        mean_monthly = last_window_data.mean()
        # Annualize the mean
        annualized = (1 + mean_monthly) ** 12 - 1
        # Add to the feature DataFrame
        feat_df.loc[next_month_date] = annualized
    
    # Write to Excel as specified
    with pd.ExcelWriter('output/S13_60_Month.xlsx', engine='openpyxl') as writer:
        targets_df.to_excel(writer, sheet_name='Targets')
        feature60.to_excel(writer, sheet_name='Feature60')
        feature1.to_excel(writer, sheet_name='Feature1')
        feature12.to_excel(writer, sheet_name='Feature12')
        feature36.to_excel(writer, sheet_name='Feature36')
    
    # Create a DataFrame with monthly returns for both strategies
    monthly_returns_df = pd.DataFrame({
        'Expanding Window': expanding_returns * 100,  # Convert to percentage
        'Hybrid Window': hybrid_returns * 100  # Convert to percentage
    })
    
    # Save statistics and monthly returns to the same Excel file but different sheets
    with pd.ExcelWriter('output/S13_strategy_statistics.xlsx', engine='openpyxl') as writer:
        # Save summary statistics to the first sheet
        stats_df = pd.DataFrame({
            'Expanding Window': expanding_stats,
            'Hybrid Window': hybrid_stats
        })
        stats_df.to_excel(writer, sheet_name='Summary Statistics')
        
        # Save monthly returns to a separate sheet
        monthly_returns_df.to_excel(writer, sheet_name='Monthly Returns')
    
    print("\nStrategy Statistics:")
    print(stats_df)
    print("\nMonthly returns saved to output/S13_strategy_statistics.xlsx in sheet 'Monthly Returns'")
    
    # Print a comparison of magnitudes for a sample date
    sample_date = dates[INITIAL_WINDOW + 12] if INITIAL_WINDOW + 12 < len(dates) else dates[-1]  # Choose a date after transition
    print("\nMagnitude Comparison (Sample Date: {})".format(sample_date.strftime('%Y-%m')))
    print("Hybrid Window Forecast Mean: {:.4f}".format(hybrid_forecasts.loc[sample_date].mean()))
    print("Exponential Forecast Mean: {:.4f}".format(exp_forecasts.loc[sample_date].mean()))
    print("\nHybrid Window Forecast Min/Max: {:.4f} / {:.4f}".format(
        hybrid_forecasts.loc[sample_date].min(), 
        hybrid_forecasts.loc[sample_date].max()
    ))
    print("Exponential Forecast Min/Max: {:.4f} / {:.4f}".format(
        exp_forecasts.loc[sample_date].min(), 
        exp_forecasts.loc[sample_date].max()
    ))
    
    # Create performance plot
    plt.figure(figsize=(15, 8))
    plt.plot(cum_expanding, label='Expanding Window', linewidth=2)
    plt.plot(cum_hybrid, label='Hybrid Window', linewidth=2)
    plt.title('Cumulative Performance Comparison', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y-1)))
    
    # Enhance grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    # Add performance annotations
    ann_ret_exp = expanding_stats['Annualized Return (%)']
    ann_ret_hyb = hybrid_stats['Annualized Return (%)']
    plt.figtext(0.02, 0.02, 
                f'Expanding Window: {ann_ret_exp:.1f}% p.a.\nHybrid Window: {ann_ret_hyb:.1f}% p.a.', 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add vertical line at transition date if it exists
    if transition_date:
        plt.axvline(x=transition_date, color='red', linestyle='--', alpha=0.5)
        plt.figtext(0.5, 0.01, 
                    f'Transition to Rolling Window: {transition_date.strftime("%Y-%m")}', 
                    fontsize=10, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    # Save only in PDF format
    plt.savefig('output/S13_strategy_performance.pdf')
    plt.close()
    
    return expanding_returns, hybrid_returns, expanding_weights, hybrid_weights

if __name__ == "__main__":
    print("\nStarting Portfolio Optimization with External Expected Returns")
    print("=" * 70)
    print("This program will:")
    print("1. Read expected returns (alphas) from Factor_Alpha.xlsx")
    print("2. Read historical returns from T2_Optimizer.xlsx for covariance estimation")
    print("3. Optimize portfolio weights using the hybrid window approach")
    print("4. Evaluate performance and generate output files")
    print("=" * 70)
    
    # Check if input files exist
    if not os.path.exists('Factor_Alpha.xlsx'):
        print("\nERROR: Factor_Alpha.xlsx not found!")
        print("Please ensure Factor_Alpha.xlsx exists in the current directory.")
        print("This file should contain expected returns/alphas (annualized) for each factor.")
        exit(1)
        
    if not os.path.exists('T2_Optimizer.xlsx'):
        print("\nERROR: T2_Optimizer.xlsx not found!")
        print("Please ensure T2_Optimizer.xlsx exists in the current directory.")
        print("This file should contain historical returns for covariance estimation.")
        exit(1)
    
    # Run the optimization with external expected returns
    try:
        weights, returns, stats = optimize_portfolio_with_external_returns(
            expected_returns_file='Factor_Alpha.xlsx',
            hhi_penalty=0.1,
            initial_window=60
        )
        
        print("\nOptimization completed successfully!")
        print("\nSummary Statistics:")
        stats_formatted = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in stats.items()}
        for key, value in stats_formatted.items():
            print(f"- {key}: {value}")
            
    except Exception as e:
        print(f"\nERROR during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your input files and try again.")
    
    print("\nTo run the original dual-strategy optimization instead, use:")
    print("expanding_returns, hybrid_returns, expanding_weights, hybrid_weights = run_rolling_optimization()")
