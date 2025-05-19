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
from joblib import Parallel, delayed
import multiprocessing

# Section: Define Constants
RANDOM_SEED = 42
NUM_TUNING_WINDOWS = 25      # Using 25 windows for tuning
NUM_FACTORS_FOR_TUNING = 10  # Using a subset of factors for faster tuning
VERBOSE = True               # Print detailed progress information

# Determine number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count() - 1  # Leave one core free for system processes

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
    Select a fixed middle window or randomly select windows for hyperparameter tuning.
    
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
    if num_windows == 1:
        # For single window, pick the middle window for a balanced representation
        all_window_ids = window_schedule['Window_ID'].tolist()
        middle_window = [all_window_ids[len(all_window_ids) // 2]]
        
        print(f"Selected window {middle_window[0]} for tuning:")
        selected_window_info = window_schedule[window_schedule['Window_ID'] == middle_window[0]]
        for _, window in selected_window_info.iterrows():
            print(f"  Window {window['Window_ID']}: Training {window['Training_Start_Date'].strftime('%Y-%m')} to {window['Training_End_Date'].strftime('%Y-%m')}, "
                  f"Validation {window['Validation_Start_Date'].strftime('%Y-%m')} to {window['Validation_End_Date'].strftime('%Y-%m')}")
        
        return middle_window
    else:
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
    Only considers 1-month factors (excludes 3m, 12m, and 60m suffixes).
    
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
    
    # Filter to exclude factors with 3m, 12m, and 60m suffixes
    filtered_factors = [f for f in all_factors if not (f.endswith('_3m') or f.endswith('_12m') or f.endswith('_60m'))]
    
    # If we have a filtered list, use it; otherwise fall back to all factors
    if filtered_factors:
        print(f"Filtered out {len(all_factors) - len(filtered_factors)} factors with _3m, _12m, or _60m suffixes.")
        all_factors = filtered_factors
    else:
        print("Warning: No factors remain after filtering out time period suffixes. Using all factors.")
    
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
    # Use the full parameter grid as originally specified
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.02],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5]
    }
    
    return param_grid

def train_and_evaluate_with_cv(X_train, y_train, params, num_folds=3, verbose=VERBOSE):
    """
    Train an XGBoost model using built-in cross-validation.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training target values.
    params : dict
        XGBoost parameters.
    num_folds : int
        Number of folds for cross-validation.
    verbose : bool
        Whether to print detailed progress information.
        
    Returns:
    --------
    dict
        Evaluation metrics.
    """
    if verbose:
        print(f"    Running {num_folds}-fold cross-validation with data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Create DMatrix object once for all CV runs
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Run cross-validation (much more efficient than manual splitting)
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        nfold=num_folds,
        early_stopping_rounds=10,
        metrics=['rmse', 'mae'],
        as_pandas=True,
        seed=RANDOM_SEED,
        verbose_eval=verbose
    )
    
    # Calculate additional metrics (CV already gives us RMSE and MAE)
    best_iteration = len(cv_results)
    best_rmse = cv_results['test-rmse-mean'].iloc[-1]
    best_mae = cv_results['test-mae-mean'].iloc[-1]
    
    # For R² we need to calculate it ourselves
    model = xgb.train(params, dtrain, num_boost_round=best_iteration)
    preds = model.predict(dtrain)
    r2 = r2_score(y_train, preds)
    
    if verbose:
        print(f"    CV results: best_iteration={best_iteration}, RMSE={best_rmse:.6f}, MAE={best_mae:.6f}, R2={r2:.6f}")
    
    return {
        'best_iteration': best_iteration,
        'best_score': -best_rmse,  # Negative because XGBoost maximizes score
        'mse': best_rmse**2,
        'rmse': best_rmse,
        'mae': best_mae,
        'r2': r2
    }

def train_and_evaluate(X_train, y_train, X_val, y_val, params, early_stopping_rounds=10, verbose=VERBOSE):
    """
    Train an XGBoost model with the given parameters and evaluate on validation data.
    This function is kept for backward compatibility but now uses XGBoost's built-in cross-validation.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training target values.
    X_val : numpy.ndarray
        Validation features (not used when using built-in CV).
    y_val : numpy.ndarray
        Validation target values (not used when using built-in CV).
    params : dict
        XGBoost parameters.
    early_stopping_rounds : int
        Number of rounds for early stopping.
    verbose : bool
        Whether to print detailed progress information.
        
    Returns:
    --------
    dict
        Evaluation metrics.
    """
    if verbose:
        print(f"    Training with data: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Use the built-in CV function instead
    return train_and_evaluate_with_cv(X_train, y_train, params, num_folds=3, verbose=verbose)

# Global cache for feature sets to avoid repeated disk access
FEATURE_SETS_CACHE = {}

def preload_feature_data(feature_sets_path, selected_windows, selected_factors, verbose=VERBOSE):
    """
    Preload and cache feature data for selected windows and factors using optimized H5 access.
    
    Parameters:
    -----------
    feature_sets_path : str
        Path to the H5 file containing feature sets.
    selected_windows : list
        List of window IDs to preload.
    selected_factors : list
        List of factor IDs to preload.
    verbose : bool
        Whether to print detailed progress information.
        
    Returns:
    --------
    dict
        Cached feature sets data.
    """
    global FEATURE_SETS_CACHE
    
    if verbose:
        print(f"Preloading data for {len(selected_windows)} windows and {len(selected_factors)} factors...")
    
    # Initialize cache structure
    for window_id in selected_windows:
        FEATURE_SETS_CACHE[window_id] = {}
    
    # Prepare the list of datasets to load
    datasets_to_load = []
    window_keys = [f"window_{window_id}" for window_id in selected_windows]
    
    # Load data from H5 file with optimized access pattern
    with h5py.File(feature_sets_path, 'r') as h5f:
        fs_group = h5f['feature_sets']
        
        # First pass: identify all datasets to load
        for window_id, window_key in zip(selected_windows, window_keys):
            if window_key not in fs_group:
                if verbose:
                    print(f"  Warning: Window {window_id} not found in H5 file")
                continue
                
            window_group = fs_group[window_key]
            if 'training' not in window_group or 'validation' not in window_group:
                if verbose:
                    print(f"  Warning: Training or validation data missing for window {window_id}")
                continue
                
            training_group = window_group['training']
            validation_group = window_group['validation']
            
            # Determine common factors present in both training and validation
            common_factors = set(training_group.keys()).intersection(set(validation_group.keys()))
            target_factors = set(selected_factors).intersection(common_factors)
            
            for factor_id in target_factors:
                # Check if the factor has the expected structure
                if 'X' in training_group[factor_id] and 'y' in training_group[factor_id] and \
                   'X' in validation_group[factor_id] and 'y' in validation_group[factor_id]:
                    
                    # Check data availability
                    if 'data' in training_group[factor_id]['X'] and 'data' in validation_group[factor_id]['X']:
                        # Add datasets to the list for optimized loading
                        datasets_to_load.append({
                            'window_id': window_id,
                            'factor_id': factor_id,
                            'X_train_path': f"{window_key}/training/{factor_id}/X/data",
                            'y_train_path': f"{window_key}/training/{factor_id}/y",
                            'X_val_path': f"{window_key}/validation/{factor_id}/X/data",
                            'y_val_path': f"{window_key}/validation/{factor_id}/y"
                        })
        
        # Second pass: load all datasets using group navigation
        for dataset in datasets_to_load:
            window_id = dataset['window_id']
            factor_id = dataset['factor_id']
            window_key = f"window_{window_id}"
            
            # Use group navigation instead of direct path access for compatibility
            X_train = np.array(fs_group[window_key]['training'][factor_id]['X']['data'], dtype=np.float64)
            y_train = np.array(fs_group[window_key]['training'][factor_id]['y'], dtype=np.float64)
            X_val = np.array(fs_group[window_key]['validation'][factor_id]['X']['data'], dtype=np.float64)
            y_val = np.array(fs_group[window_key]['validation'][factor_id]['y'], dtype=np.float64)
            
            # Reshape if needed
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            if X_val.ndim == 1:
                X_val = X_val.reshape(-1, 1)
            
            # Store in cache
            FEATURE_SETS_CACHE[window_id][factor_id] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
    
    if verbose:
        total_cached = sum(len(FEATURE_SETS_CACHE[w]) for w in FEATURE_SETS_CACHE)
        print(f"Data preloading complete. Cached {total_cached} window-factor combinations.")
    
    return FEATURE_SETS_CACHE

def evaluate_parameter_combination(param_combo, base_params, feature_sets_cache, selected_windows, selected_factors, verbose=VERBOSE):
    """
    Evaluate a specific parameter combination across all selected windows and factors.
    
    Parameters:
    -----------
    param_combo : dict
        Parameter combination to evaluate.
    base_params : dict
        Base XGBoost parameters to update.
    feature_sets_cache : dict
        Cached feature sets for all windows and factors.
    selected_windows : list
        List of window IDs to use for evaluation.
    selected_factors : list
        List of factor IDs to use for evaluation.
    verbose : bool
        Whether to print detailed progress information.
        
    Returns:
    --------
    dict
        Average performance metrics across all windows and factors.
    """
    # Update base parameters with the combo to test
    test_params = base_params.copy()
    test_params.update(param_combo)
    
    if verbose:
        print(f"  Testing parameters: {param_combo}")
    
    # Initialize metrics
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'best_iteration': []
    }
    
    # Track performance for early stopping of poor performers
    early_performance_check_count = 0
    early_performance_threshold = min(3, len(selected_windows) * len(selected_factors) // 4)
    accumulated_rmse = 0
    
    # Iterate through selected windows and factors
    for window_id in selected_windows:
        if verbose:
            print(f"  Processing window {window_id}:")
        
        for i, factor_id in enumerate(selected_factors):
            if verbose:
                print(f"    Factor {i+1}/{len(selected_factors)}: {factor_id}")
                
            # Skip if data not available for this window-factor combination
            if window_id not in feature_sets_cache or factor_id not in feature_sets_cache[window_id]:
                if verbose:
                    print(f"      Skipping: data not available")
                continue
            
            # Get data from cache
            window_data = feature_sets_cache[window_id][factor_id]
            
            X_train = window_data['X_train']
            y_train = window_data['y_train']
            
            # Train and evaluate using built-in CV (no need for separate validation set)
            eval_results = train_and_evaluate_with_cv(X_train, y_train, test_params, verbose=verbose)
            
            # Collect metrics
            for metric in metrics:
                if metric in eval_results:
                    metrics[metric].append(eval_results[metric])
            
            # Check if this parameter combination is performing poorly (for early stopping)
            if 'rmse' in eval_results:
                early_performance_check_count += 1
                accumulated_rmse += eval_results['rmse']
                
                # If we've checked enough combinations and performance is poor, stop evaluation
                if early_performance_check_count >= early_performance_threshold:
                    # Calculate average RMSE so far
                    avg_rmse_so_far = accumulated_rmse / early_performance_check_count
                    
                    # If it's significantly worse than a reasonable threshold, stop evaluation
                    # This threshold could be improved with historical data
                    if avg_rmse_so_far > 2.0:  # This threshold is an example
                        if verbose:
                            print(f"    Early stopping: Poor performance detected (RMSE={avg_rmse_so_far:.4f})")
                        
                        # Return early with current metrics
                        break
    
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
    
    if verbose:
        print(f"  RESULTS: Avg RMSE={avg_metrics['avg_rmse']:.6f}, Avg R2={avg_metrics['avg_r2']:.6f}, Samples={avg_metrics['samples']}")
    
    return avg_metrics

def process_parameter_combination(combo_values, param_keys, base_params, feature_sets_cache, selected_windows, selected_factors, combo_index, total_combos):
    """
    Process a single parameter combination for parallel execution.
    
    Parameters:
    -----------
    combo_values : tuple
        Values for the parameters.
    param_keys : list
        Parameter names.
    base_params : dict
        Base XGBoost parameters to update.
    feature_sets : dict
        Feature sets for all windows and factors.
    selected_windows : list
        List of window IDs for evaluation.
    selected_factors : list
        List of factor IDs for evaluation.
    combo_index : int
        Index of the current combination.
    total_combos : int
        Total number of combinations.
        
    Returns:
    --------
    dict
        Metrics for this parameter combination.
    """
    combo = dict(zip(param_keys, combo_values))
    if VERBOSE:
        print(f"Evaluating combination {combo_index+1}/{total_combos}: {combo}")
    
    # Evaluate this combination
    start_time = time.time()
    avg_metrics = evaluate_parameter_combination(combo, base_params, feature_sets_cache, selected_windows, selected_factors)
    elapsed_time = time.time() - start_time
    
    # Add elapsed time and parameter values
    avg_metrics['elapsed_time'] = elapsed_time
    
    # Print interim results
    if VERBOSE:
        print(f"  Results: RMSE={avg_metrics['avg_rmse']:.6f}, R2={avg_metrics['avg_r2']:.6f}, Time={elapsed_time:.2f}s")
    
    return avg_metrics

def perform_grid_search(base_params, param_grid, feature_sets_cache, selected_windows, selected_factors):
    """
    Perform a grid search over all parameter combinations using parallel processing.
    
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
    
    total_combos = len(param_combinations)
    print(f"Starting grid search with {total_combos} parameter combinations using {NUM_CORES} CPU cores in parallel")
    
    # Use joblib for parallel processing
    results = Parallel(n_jobs=NUM_CORES, verbose=10)(
        delayed(process_parameter_combination)(
            combo_values, param_keys, base_params, feature_sets_cache, 
            selected_windows, selected_factors, i, total_combos
        ) for i, combo_values in enumerate(param_combinations)
    )
    
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

def main():
    print("=== Step 10A: Tune XGBoost Hyperparameters ===")
    
    # Step 10A.1: Load data
    print("\n--- 10A.1 Loading Data ---")
    window_schedule, _, base_xgb_params = load_data()
    
    # Step 10A.2: Select windows for tuning
    print("\n--- 10A.2 Selecting Windows for Tuning ---")
    selected_windows = select_tuning_windows(window_schedule, NUM_TUNING_WINDOWS)
    
    # Step 10A.3: Select factors for tuning
    print("\n--- 10A.3 Selecting Factors for Tuning ---")
    # We'll extract factor names from the H5 file during preloading
    feature_sets_path = os.path.join(output_dir, "S8_Feature_Sets.h5")
    with h5py.File(feature_sets_path, 'r') as h5f:
        fs_group = h5f['feature_sets']
        first_window_key = list(fs_group.keys())[0]
        available_factors = list(fs_group[first_window_key]['training'].keys())
        selected_factors = sorted(random.sample(available_factors, min(NUM_FACTORS_FOR_TUNING, len(available_factors))))
    
    print(f"Selected {len(selected_factors)} factors for tuning: {selected_factors[:5]}... (showing first 5)")
    
    # Step 10A.3.5: Preload and cache feature data
    print("\n--- 10A.3.5 Preloading Feature Data ---")
    feature_sets_cache = preload_feature_data(feature_sets_path, selected_windows, selected_factors)
    
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
    tuning_results = perform_grid_search(base_xgb_params, param_grid, feature_sets_cache, selected_windows, selected_factors)
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
    
    # Skip visualizations
    print("\n--- 10A.8 Skipping Visualizations ---")
    print("Visualization disabled per user request")
    
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
