'''
Step_10A_Tune_XGBoost_Hyperparameters.py

This script performs hyperparameter tuning for XGBoost models by randomly selecting
25 windows from the entire timescale rather than using the first 15-20 windows.
It tests various parameter combinations and identifies the optimal configuration based
on validation performance across these windows.

----------------------------------------------------------------------------------------------------
INPUT FILES:
- S4_Window_Schedule.xlsx
  - Path: ./output/S4_Window_Schedule.xlsx
  - Description: Contains the schedule of all rolling windows.
  - Format: Excel (.xlsx) with window information.

- S8_Feature_Sets.pkl
  - Path: ./output/S8_Feature_Sets.pkl
  - Description: Feature sets for each factor and window.
  - Format: Pickle file with nested dictionary structure.

- S9_XGBoost_Config.json
  - Path: ./output/S9_XGBoost_Config.json
  - Description: Base XGBoost configuration.
  - Format: JSON file with parameter dictionary.

OUTPUT FILES:
- S10A_XGBoost_Tuning_Results.xlsx
  - Path: ./output/S10A_XGBoost_Tuning_Results.xlsx
  - Description: Results of hyperparameter tuning experiments.
  - Format: Excel with parameter combinations and performance metrics.

- S10A_XGBoost_Optimal_Params.json
  - Path: ./output/S10A_XGBoost_Optimal_Params.json
  - Description: Optimal XGBoost parameters after tuning.
  - Format: JSON file with parameter dictionary.

- S10A_Tuning_Windows.pkl
  - Path: ./output/S10A_Tuning_Windows.pkl
  - Description: List of windows used for tuning (for reference).
  - Format: Pickle file with list of window IDs.

- S10A_Tuning_Visualization.pdf
  - Path: ./output/S10A_Tuning_Visualization.pdf
  - Description: Visualizations of tuning results.
  - Format: PDF with performance charts.

Version: 1.0
Last Updated: 2025-05-15
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import pickle
import os
import sys
import time
import random
import h5py
from pathlib import Path
from itertools import product
from datetime import datetime

# Section: Define Constants
RANDOM_SEED = 42
NUM_TUNING_WINDOWS = 25
NUM_FACTORS_FOR_TUNING = 20  # Using a subset of factors for faster tuning

# Ensure deterministic behavior
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Section: Ensure Output Directory Exists
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Section: Utility Functions
def load_data():
    """
    Load the window schedule, feature sets, and base XGBoost configuration.
    
    Returns:
    --------
    tuple
        Window schedule, feature sets, and base XGBoost configuration.
    """
    try:
        # Load window schedule
        window_schedule_path = os.path.join(output_dir, "S4_Window_Schedule.xlsx")
        window_schedule = pd.read_excel(window_schedule_path)
        print(f"Loaded window schedule from {window_schedule_path}")
        
        # Load feature sets from H5 file
        feature_sets_path = os.path.join(output_dir, "S8_Feature_Sets.h5")
        feature_sets = {}
        
        with h5py.File(feature_sets_path, 'r') as h5f:
            # Access the feature_sets group
            fs_group = h5f['feature_sets']
            
            # Process each window
            for window_key in fs_group.keys():
                # Extract window ID (e.g., 'window_1' -> 1)
                window_id = int(window_key.split('_')[1])
                window_dict = {}
                
                # Get training and validation data
                training_group = fs_group[window_key]['training']
                validation_group = fs_group[window_key]['validation']
                
                # Process each factor (column)
                all_factors = set(list(training_group.keys()) + list(validation_group.keys()))
                
                for factor_name in all_factors:
                    # Skip if factor not present in both training and validation
                    if factor_name not in training_group or factor_name not in validation_group:
                        continue
                    
                    # Check if the factor is a group with X and y datasets
                    if 'X' in training_group[factor_name] and 'y' in training_group[factor_name] and \
                       'X' in validation_group[factor_name] and 'y' in validation_group[factor_name]:
                        # Handle the nested structure where X is a group containing 'data' subgroup
                        if 'data' in training_group[factor_name]['X'] and 'data' in validation_group[factor_name]['X']:
                            # Extract data for this factor
                            X_train = np.array(training_group[factor_name]['X']['data'], dtype=np.float64)
                            y_train = np.array(training_group[factor_name]['y'], dtype=np.float64)
                            X_val = np.array(validation_group[factor_name]['X']['data'], dtype=np.float64)
                            y_val = np.array(validation_group[factor_name]['y'], dtype=np.float64)
                            
                            # Store in the dictionary
                            window_dict[factor_name] = {
                                'X_train': X_train,
                                'y_train': y_train,
                                'X_val': X_val,
                                'y_val': y_val
                            }
                
                # Add this window to the feature sets dictionary
                feature_sets[window_id] = window_dict
        
        print(f"Loaded feature sets from {feature_sets_path}")
        print(f"  - Processed {len(feature_sets)} windows")
        print(f"  - Example window ID: {list(feature_sets.keys())[0]}")
        
        example_window = list(feature_sets.keys())[0]
        print(f"  - Example factors in window: {list(feature_sets[example_window].keys())[:5]}...")
        
        # Print an example feature shape for debugging
        if feature_sets and example_window in feature_sets:
            example_factor = list(feature_sets[example_window].keys())[0] if feature_sets[example_window] else None
            if example_factor:
                print(f"  - Example feature shapes for {example_factor}:")
                print(f"    X_train: {feature_sets[example_window][example_factor]['X_train'].shape}")
                print(f"    y_train: {feature_sets[example_window][example_factor]['y_train'].shape}")
        
        # Load base XGBoost configuration
        xgb_config_path = os.path.join(output_dir, "S9_XGBoost_Config.json")
        with open(xgb_config_path, 'r') as f:
            base_xgb_params = json.load(f)
        print(f"Loaded base XGBoost configuration from {xgb_config_path}")
        
        return window_schedule, feature_sets, base_xgb_params
    
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def select_tuning_windows(window_schedule, num_windows=NUM_TUNING_WINDOWS):
    """
    Randomly select windows from the entire timescale for hyperparameter tuning.
    
    Parameters:
    -----------
    window_schedule : pandas.DataFrame
        DataFrame containing all windows.
    num_windows : int
        Number of windows to select for tuning.
        
    Returns:
    --------
    list
        List of selected window IDs.
    """
    # Get all window IDs
    all_window_ids = window_schedule['Window_ID'].tolist()
    
    # Randomly select a subset of windows for tuning
    selected_windows = sorted(random.sample(all_window_ids, min(num_windows, len(all_window_ids))))
    
    # Print selected windows information
    print(f"Randomly selected {len(selected_windows)} windows for tuning:")
    selected_windows_info = window_schedule[window_schedule['Window_ID'].isin(selected_windows)]
    for _, window in selected_windows_info.iterrows():
        print(f"  Window {window['Window_ID']}: Training {window['Training_Start_Date'].strftime('%Y-%m')} to {window['Training_End_Date'].strftime('%Y-%m')}, "
              f"Validation {window['Validation_Start_Date'].strftime('%Y-%m')} to {window['Validation_End_Date'].strftime('%Y-%m')}")
    
    return selected_windows

def select_factors_for_tuning(feature_sets, num_factors=NUM_FACTORS_FOR_TUNING):
    """
    Randomly select a subset of factors for tuning to reduce computational load.
    
    Parameters:
    -----------
    feature_sets : dict
        Dictionary of feature sets by factor and window.
    num_factors : int
        Number of factors to use for tuning.
        
    Returns:
    --------
    list
        List of selected factor IDs.
    """
    # Get all unique factor IDs from the feature sets
    all_factors = list(set([factor_id for window_id in feature_sets for factor_id in feature_sets[window_id]]))
    
    # Randomly select a subset of factors for tuning
    selected_factors = sorted(random.sample(all_factors, min(num_factors, len(all_factors))))
    
    print(f"Selected {len(selected_factors)} factors for tuning: {selected_factors[:5]}... (showing first 5)")
    
    return selected_factors

def define_parameter_grid():
    """
    Define the grid of hyperparameters to test.
    
    Returns:
    --------
    dict
        Dictionary with parameter names and values to test.
    """
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.02],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5]
    }
    
    return param_grid

def train_and_evaluate(X_train, y_train, X_val, y_val, params, early_stopping_rounds=10):
    """
    Train an XGBoost model with the given parameters and evaluate on validation data.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training target values.
    X_val : numpy.ndarray
        Validation features.
    y_val : numpy.ndarray
        Validation target values.
    params : dict
        XGBoost parameters.
    early_stopping_rounds : int
        Number of rounds for early stopping.
        
    Returns:
    --------
    dict
        Evaluation metrics.
    """
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train the model with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,  # Maximum number of rounds
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    # Make predictions
    val_pred = model.predict(dval)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, val_pred)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    
    return {
        'model': model,
        'best_iteration': model.best_iteration,
        'best_score': model.best_score,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }

def evaluate_parameter_combination(param_combo, base_params, feature_sets, selected_windows, selected_factors):
    """
    Evaluate a specific parameter combination across all selected windows and factors.
    
    Parameters:
    -----------
    param_combo : dict
        Parameter combination to evaluate.
    base_params : dict
        Base XGBoost parameters to update.
    feature_sets : dict
        Feature sets for all windows and factors.
    selected_windows : list
        List of window IDs to use for evaluation.
    selected_factors : list
        List of factor IDs to use for evaluation.
        
    Returns:
    --------
    dict
        Average performance metrics across all windows and factors.
    """
    # Update base parameters with the combo to test
    test_params = base_params.copy()
    test_params.update(param_combo)
    
    # Initialize metrics
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'best_iteration': []
    }
    
    # Iterate through selected windows and factors
    for window_id in selected_windows:
        for factor_id in selected_factors:
            # Skip if data not available for this window-factor combination
            if window_id not in feature_sets or factor_id not in feature_sets[window_id]:
                continue
            
            # Get data for this window-factor combination
            window_data = feature_sets[window_id][factor_id]
            
            X_train = window_data['X_train']
            y_train = window_data['y_train']
            X_val = window_data['X_val']
            y_val = window_data['y_val']
            
            # Train and evaluate
            eval_results = train_and_evaluate(X_train, y_train, X_val, y_val, test_params)
            
            # Collect metrics
            for metric in metrics:
                if metric in eval_results:
                    metrics[metric].append(eval_results[metric])
    
    # Calculate averages
    avg_metrics = {
        'param_combo': param_combo,
        'avg_mse': np.mean(metrics['mse']),
        'avg_rmse': np.mean(metrics['rmse']),
        'avg_mae': np.mean(metrics['mae']),
        'avg_r2': np.mean(metrics['r2']),
        'avg_best_iteration': np.mean(metrics['best_iteration']),
        'std_mse': np.std(metrics['mse']),
        'std_rmse': np.std(metrics['rmse']),
        'std_mae': np.std(metrics['mae']),
        'std_r2': np.std(metrics['r2']),
        'samples': len(metrics['mse'])
    }
    
    return avg_metrics

def perform_grid_search(base_params, param_grid, feature_sets, selected_windows, selected_factors):
    """
    Perform a grid search over all parameter combinations.
    
    Parameters:
    -----------
    base_params : dict
        Base XGBoost parameters to update.
    param_grid : dict
        Parameter grid with values to test.
    feature_sets : dict
        Feature sets for all windows and factors.
    selected_windows : list
        List of window IDs to use for evaluation.
    selected_factors : list
        List of factor IDs to use for evaluation.
        
    Returns:
    --------
    list
        List of results for all parameter combinations.
    """
    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Initialize results
    results = []
    
    # Iterate through all parameter combinations
    total_combos = len(param_combinations)
    print(f"Starting grid search with {total_combos} parameter combinations")
    
    for i, combo_values in enumerate(param_combinations):
        combo = dict(zip(param_keys, combo_values))
        print(f"Evaluating combination {i+1}/{total_combos}: {combo}")
        
        # Evaluate this combination
        start_time = time.time()
        avg_metrics = evaluate_parameter_combination(combo, base_params, feature_sets, selected_windows, selected_factors)
        elapsed_time = time.time() - start_time
        
        # Add elapsed time
        avg_metrics['elapsed_time'] = elapsed_time
        
        # Add to results
        results.append(avg_metrics)
        
        # Print interim results
        print(f"  Results: RMSE={avg_metrics['avg_rmse']:.6f}, R2={avg_metrics['avg_r2']:.6f}, Time={elapsed_time:.2f}s")
    
    return results

def find_best_parameters(tuning_results):
    """
    Find the best parameter combination based on validation RMSE.
    
    Parameters:
    -----------
    tuning_results : list
        List of results for all parameter combinations.
        
    Returns:
    --------
    dict
        Best parameter combination.
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(tuning_results)
    
    # Sort by average RMSE (ascending)
    results_df = results_df.sort_values('avg_rmse')
    
    # Get the best parameter combination
    best_params = results_df.iloc[0]['param_combo']
    
    return best_params

def visualize_tuning_results(tuning_results, output_file):
    """
    Create visualizations of the tuning results.
    
    Parameters:
    -----------
    tuning_results : list
        List of results for all parameter combinations.
    output_file : str
        Path to save the visualizations.
        
    Returns:
    --------
    None
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(tuning_results)
    
    # Create parameter columns for easier visualization
    for param in tuning_results[0]['param_combo']:
        results_df[param] = results_df['param_combo'].apply(lambda x: x[param])
    
    # Sort by RMSE
    results_df = results_df.sort_values('avg_rmse')
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.5, 'XGBoost Hyperparameter Tuning Results',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()
        
        # Parameter importance plot (impact on RMSE)
        plt.figure(figsize=(11, 8.5))
        param_impacts = {}
        for param in results_df['param_combo'][0].keys():
            param_values = results_df[param].unique()
            impacts = []
            for val in param_values:
                impacts.append({
                    'value': val,
                    'avg_rmse': results_df[results_df[param] == val]['avg_rmse'].mean()
                })
            param_impacts[param] = impacts
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (param, impacts) in enumerate(param_impacts.items()):
            if i < len(axes):
                impact_df = pd.DataFrame(impacts)
                axes[i].bar(impact_df['value'].astype(str), impact_df['avg_rmse'])
                axes[i].set_title(f'Impact of {param} on RMSE')
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Average RMSE')
                
                # Add value labels
                for j, v in enumerate(impact_df['avg_rmse']):
                    axes[i].text(j, v + 0.0005, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Pairwise parameter interaction plots
        params = list(results_df['param_combo'][0].keys())
        n_params = len(params)
        
        for i in range(n_params):
            for j in range(i+1, n_params):
                param1 = params[i]
                param2 = params[j]
                
                pivot_table = results_df.pivot_table(
                    values='avg_rmse',
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Average RMSE'})
                plt.title(f'Interaction between {param1} and {param2}')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        # Top 10 parameter combinations
        plt.figure(figsize=(12, 8))
        top10 = results_df.head(10).copy()
        param_cols = list(top10['param_combo'][0].keys())
        
        # Create a combined parameter string for each combination
        top10['param_string'] = top10.apply(
            lambda row: '\n'.join([f"{param}: {row[param]}" for param in param_cols]),
            axis=1
        )
        
        plt.barh(top10['param_string'], top10['avg_rmse'])
        plt.xlabel('Average RMSE')
        plt.ylabel('Parameter Combination')
        plt.title('Top 10 Parameter Combinations')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Distribution of metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE distribution
        sns.histplot(results_df['avg_rmse'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Average RMSE')
        axes[0, 0].axvline(results_df['avg_rmse'].min(), color='r', linestyle='--', 
                          label=f'Min: {results_df["avg_rmse"].min():.4f}')
        axes[0, 0].legend()
        
        # R2 distribution
        sns.histplot(results_df['avg_r2'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Average R²')
        axes[0, 1].axvline(results_df['avg_r2'].max(), color='r', linestyle='--',
                         label=f'Max: {results_df["avg_r2"].max():.4f}')
        axes[0, 1].legend()
        
        # MAE distribution
        sns.histplot(results_df['avg_mae'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Average MAE')
        axes[1, 0].axvline(results_df['avg_mae'].min(), color='r', linestyle='--',
                         label=f'Min: {results_df["avg_mae"].min():.4f}')
        axes[1, 0].legend()
        
        # Best iteration distribution
        sns.histplot(results_df['avg_best_iteration'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Average Best Iteration')
        axes[1, 1].axvline(results_df['avg_best_iteration'].median(), color='r', linestyle='--',
                         label=f'Median: {results_df["avg_best_iteration"].median():.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Summary page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        best_row = results_df.iloc[0]
        best_params_str = '\n'.join([f"{k}: {v}" for k, v in best_row['param_combo'].items()])
        
        plt.text(0.5, 0.95, 'Summary of Best Parameters',
                horizontalalignment='center', verticalalignment='top',
                fontsize=16, fontweight='bold')
        
        summary_text = (
            f"Best Parameter Combination:\n{best_params_str}\n\n"
            f"Performance Metrics:\n"
            f"- Average RMSE: {best_row['avg_rmse']:.6f}\n"
            f"- Average MSE: {best_row['avg_mse']:.6f}\n"
            f"- Average MAE: {best_row['avg_mae']:.6f}\n"
            f"- Average R²: {best_row['avg_r2']:.6f}\n"
            f"- Average Best Iteration: {best_row['avg_best_iteration']:.1f}\n"
            f"- Standard Deviation RMSE: {best_row['std_rmse']:.6f}\n\n"
            f"Tuning Statistics:\n"
            f"- Number of windows used: {len(selected_windows)}\n"
            f"- Number of factors used: {len(selected_factors)}\n"
            f"- Total parameter combinations tested: {len(results_df)}\n"
            f"- Total samples (window-factor combinations): {best_row['samples']}\n"
        )
        
        plt.text(0.5, 0.85, summary_text,
                horizontalalignment='center', verticalalignment='top',
                fontsize=12)
        
        pdf.savefig()
        plt.close()

# Section: Main Execution
def main():
    print("=== Step 10A: Tune XGBoost Hyperparameters ===")
    
    # Step 10A.1: Load data
    print("\n--- 10A.1 Loading Data ---")
    window_schedule, feature_sets, base_xgb_params = load_data()
    
    # Step 10A.2: Select windows for tuning
    print("\n--- 10A.2 Selecting Windows for Tuning ---")
    selected_windows = select_tuning_windows(window_schedule, NUM_TUNING_WINDOWS)
    
    # Step 10A.3: Select factors for tuning
    print("\n--- 10A.3 Selecting Factors for Tuning ---")
    selected_factors = select_factors_for_tuning(feature_sets, NUM_FACTORS_FOR_TUNING)
    
    # Step 10A.4: Define parameter grid
    print("\n--- 10A.4 Defining Parameter Grid ---")
    param_grid = define_parameter_grid()
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"Parameter grid defined with {total_combinations} combinations")
    
    # Print out the parameter grid
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Step 10A.5: Perform grid search
    print("\n--- 10A.5 Performing Grid Search ---")
    start_time = time.time()
    tuning_results = perform_grid_search(base_xgb_params, param_grid, feature_sets, selected_windows, selected_factors)
    elapsed_time = time.time() - start_time
    print(f"Grid search completed in {elapsed_time:.2f} seconds")
    
    # Step 10A.6: Find best parameters
    print("\n--- 10A.6 Finding Best Parameters ---")
    best_params = find_best_parameters(tuning_results)
    optimal_params = base_xgb_params.copy()
    optimal_params.update(best_params)
    
    print("Best parameter combination:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Step 10A.7: Save results
    print("\n--- 10A.7 Saving Results ---")
    
    # Save tuning results
    results_df = pd.DataFrame(tuning_results)
    results_path = os.path.join(output_dir, "S10A_XGBoost_Tuning_Results.xlsx")
    
    # Clean up the param_combo column for Excel
    results_df['param_combo_str'] = results_df['param_combo'].apply(lambda x: str(x))
    results_df = results_df.drop(columns=['param_combo'])
    results_df.to_excel(results_path, index=False)
    print(f"Saved tuning results to {results_path}")
    
    # Save optimal parameters
    optimal_params_path = os.path.join(output_dir, "S10A_XGBoost_Optimal_Params.json")
    with open(optimal_params_path, 'w') as f:
        json.dump(optimal_params, f, indent=4)
    print(f"Saved optimal parameters to {optimal_params_path}")
    
    # Save tuning windows
    tuning_windows_path = os.path.join(output_dir, "S10A_Tuning_Windows.pkl")
    with open(tuning_windows_path, 'wb') as f:
        pickle.dump({
            'window_ids': selected_windows,
            'factor_ids': selected_factors
        }, f)
    print(f"Saved tuning windows information to {tuning_windows_path}")
    
    # Generate visualizations
    print("\n--- 10A.8 Generating Visualizations ---")
    viz_path = os.path.join(output_dir, "S10A_Tuning_Visualization.pdf")
    visualize_tuning_results(tuning_results, viz_path)
    print(f"Saved tuning visualizations to {viz_path}")
    
    # Step 10A.9: Summary
    print("\n--- 10A.9 Summary ---")
    results_df = pd.DataFrame(tuning_results)
    best_idx = results_df['avg_rmse'].idxmin()
    best_result = results_df.iloc[best_idx]
    
    print("Hyperparameter Tuning Summary:")
    print(f"  Random seed used: {RANDOM_SEED}")
    print(f"  Windows used for tuning: {len(selected_windows)} windows")
    print(f"  Factors used for tuning: {len(selected_factors)} factors")
    print(f"  Parameter combinations tested: {len(tuning_results)}")
    print(f"  Best combination performance:")
    print(f"    - RMSE: {best_result['avg_rmse']:.6f}")
    print(f"    - R²: {best_result['avg_r2']:.6f}")
    print(f"    - MAE: {best_result['avg_mae']:.6f}")
    
    print("\nStep 10A completed successfully!")

if __name__ == "__main__":
    main()
