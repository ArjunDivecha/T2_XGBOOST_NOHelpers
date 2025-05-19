#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10B: Tune Linear Models

This script tunes hyperparameters for four types of linear regression models:
1. Ordinary Least Squares (OLS)
2. Ridge Regression (with alpha regularization)
3. LASSO Regression (with alpha regularization)
4. Non-Negative Least Squares (NNLS)

INPUT FILES:
- S8_Feature_Sets.h5 (output from Step 8)
  - Contains 4-dimensional feature sets for each factor and window
  - Features: 1m, 3m, 12m, 60m moving averages of the factor itself

OUTPUT FILES:
- S10B_Linear_Models_Tuning_Results.xlsx
  - Detailed results of parameter tuning for all linear model types
  - Includes performance metrics for each parameter combination

- S10B_Linear_Models_Optimal_Params.json
  - JSON file containing optimal parameters for each linear model type

- S10B_Linear_Models_Tuning_Visualization.pdf
  - PDF with visualizations of tuning results and performance comparisons

Version: 3.0
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import sys
import h5py
import json
import time
import random
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from itertools import product
from scipy.optimize import nnls

# Import scikit-learn modules
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

# Constants
RANDOM_SEED = 42
NUM_TUNING_WINDOWS = 10  # Number of windows to use for tuning
NUM_FACTORS_FOR_TUNING = 15  # Number of factors to use for tuning

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output directory if it doesn't exist
output_dir = os.path.join("output")
os.makedirs(output_dir, exist_ok=True)

def load_feature_sets(file_path):
    """
    Load feature sets from HDF5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to the feature sets HDF5 file.
    
    Returns:
    --------
    dict
        Dictionary of feature sets for each window and factor.
    """
    print(f"Loading feature sets from {file_path}...")
    
    feature_sets = {}
    
    with h5py.File(file_path, 'r') as hf:
        # Verify file structure
        if 'feature_sets' not in hf:
            raise ValueError("Invalid HDF5 file structure: 'feature_sets' group not found")
        
        feature_sets_group = hf['feature_sets']
        
        # Iterate through window groups
        for window_name in feature_sets_group:
            window_id = int(window_name.split('_')[1])
            window_group = feature_sets_group[window_name]
            
            feature_sets[window_id] = {}
            
            # Iterate through splits
            for split_name in ['training', 'validation', 'prediction']:
                if split_name in window_group:
                    split_group = window_group[split_name]
                    feature_sets[window_id][split_name] = {}
                    
                    # Iterate through factors
                    for factor_key in split_group:
                        factor_group = split_group[factor_key]
                        
                        # Factor data
                        factor_data = {}
                        
                        # Get X data
                        if 'X' in factor_group:
                            X_group = factor_group['X']
                            
                            X_data = X_group['data'][:]
                            X_columns = [col.decode('utf-8') for col in X_group['columns'][:]]
                            
                            # Create DataFrame
                            factor_data['X'] = pd.DataFrame(X_data, columns=X_columns)
                        
                        # Get y data
                        if 'y' in factor_group:
                            factor_data['y'] = factor_group['y'][:]
                        
                        # Restore original factor name (replace underscores if needed)
                        original_factor = factor_key.replace('_', '/')
                        
                        # Store factor data
                        feature_sets[window_id][split_name][original_factor] = factor_data
    
    num_windows = len(feature_sets)
    
    if num_windows > 0:
        sample_window = next(iter(feature_sets.values()))
        if 'training' in sample_window:
            num_factors = len(sample_window['training'])
            print(f"Loaded feature sets for {num_windows} windows and {num_factors} factors per window")
        else:
            print(f"Loaded feature sets for {num_windows} windows but no training data found")
    else:
        print("No windows found in feature sets")
    
    return feature_sets

def load_window_schedule(file_path):
    """
    Load window schedule from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the window schedule Excel file.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing window schedule.
    """
    print(f"Loading window schedule from {file_path}...")
    
    window_schedule = pd.read_excel(file_path)
    
    print(f"Loaded window schedule with {len(window_schedule)} windows")
    
    return window_schedule

def select_tuning_windows(window_schedule, num_windows=NUM_TUNING_WINDOWS):
    """
    Randomly select windows for hyperparameter tuning.
    
    Parameters:
    -----------
    window_schedule : pandas.DataFrame
        DataFrame containing window schedule.
    num_windows : int
        Number of windows to select.
    
    Returns:
    --------
    list
        List of selected window IDs.
    """
    print(f"Selecting {num_windows} windows for tuning...")
    
    # Get all window IDs
    window_ids = window_schedule['Window_ID'].tolist()
    
    # Randomly select windows
    selected_windows = sorted(random.sample(window_ids, min(num_windows, len(window_ids))))
    
    # Print selected windows
    print(f"Selected windows: {selected_windows}")
    
    return selected_windows

def select_factors_for_tuning(feature_sets, selected_windows, num_factors=NUM_FACTORS_FOR_TUNING):
    """
    Randomly select factors for hyperparameter tuning.
    
    Parameters:
    -----------
    feature_sets : dict
        Dictionary of feature sets for each window and factor.
    selected_windows : list
        List of selected window IDs.
    num_factors : int
        Number of factors to select.
    
    Returns:
    --------
    list
        List of selected factor IDs.
    """
    print(f"Selecting {num_factors} factors for tuning...")
    
    # Get all factors from the first window
    all_factors = set()
    for window_id in selected_windows:
        if window_id in feature_sets and 'training' in feature_sets[window_id]:
            all_factors.update(feature_sets[window_id]['training'].keys())
    
    # Randomly select factors
    selected_factors = sorted(random.sample(list(all_factors), min(num_factors, len(all_factors))))
    
    # Print selected factors
    if len(selected_factors) > 0:
        print(f"Selected factors (showing first 5): {selected_factors[:5]}")
    
    return selected_factors

def define_parameter_grids():
    """
    Define parameter grids for each model type.
    
    Returns:
    --------
    dict
        Dictionary of parameter grids for each model type.
    """
    param_grids = {
        'OLS': {
            'fit_intercept': [True, False]
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'fit_intercept': [True, False]
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'max_iter': [10000]
        },
        'NNLS': {
            # NNLS has no hyperparameters to tune in scipy implementation
            'dummy_param': [1]  # Dummy parameter for consistent interface
        }
    }
    
    # Calculate total number of parameter combinations
    total_combinations = sum(len(list(ParameterGrid(grid))) for grid in param_grids.values())
    
    print(f"Defined parameter grids with {total_combinations} total combinations")
    
    return param_grids

class NNLSRegressor:
    """
    Non-Negative Least Squares regressor class with a scikit-learn compatible interface.
    This is a wrapper around scipy.optimize.nnls.
    """
    def __init__(self, dummy_param=1):
        # NNLS doesn't have parameters, but we include dummy_param for consistency
        self.dummy_param = dummy_param
        self.coef_ = None
        self.intercept_ = 0.0  # NNLS doesn't have an intercept
    
    def fit(self, X, y):
        """
        Fit NNLS model.
        
        Parameters:
        -----------
        X : array-like
            Training features.
        y : array-like
            Target values.
        
        Returns:
        --------
        self
            Fitted model.
        """
        # Solve the NNLS problem
        self.coef_, _ = nnls(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the NNLS model.
        
        Parameters:
        -----------
        X : array-like
            Input features.
        
        Returns:
        --------
        array
            Predictions.
        """
        return X.dot(self.coef_)
    
    def get_params(self, deep=True):
        """
        Get parameters for this model.
        
        Parameters:
        -----------
        deep : bool
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns:
        --------
        dict
            Parameter names mapped to their values.
        """
        return {'dummy_param': self.dummy_param}
    
    def set_params(self, **parameters):
        """
        Set the parameters of this model.
        
        Parameters:
        -----------
        **parameters : dict
            Parameters.
        
        Returns:
        --------
        self
            Model with updated parameters.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, params):
    """
    Train and evaluate a model with specific parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('OLS', 'Ridge', 'Lasso', or 'NNLS').
    X_train : pandas.DataFrame
        Training features.
    y_train : numpy.ndarray
        Training target values.
    X_val : pandas.DataFrame
        Validation features.
    y_val : numpy.ndarray
        Validation target values.
    params : dict
        Model parameters.
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics.
    """
    try:
        # Create model
        if model_type == 'OLS':
            model = LinearRegression(**params)
        elif model_type == 'Ridge':
            model = Ridge(**params)
        elif model_type == 'Lasso':
            model = Lasso(**params)
        elif model_type == 'NNLS':
            model = NNLSRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Return metrics
        return {
            'model_type': model_type,
            'params': params,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'success': True
        }
    except Exception as e:
        # Return failure with error message
        return {
            'model_type': model_type,
            'params': params,
            'error': str(e),
            'success': False
        }

def evaluate_parameter_combination(args):
    """
    Evaluate a parameter combination across multiple windows and factors.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing (model_type, params, feature_sets, selected_windows, selected_factors).
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics aggregated across windows and factors.
    """
    model_type, params, feature_sets, selected_windows, selected_factors = args
    
    # Results for this parameter combination
    results = []
    
    # Evaluate on each window and factor
    for window_id in selected_windows:
        for factor_id in selected_factors:
            # Check if data exists for this window and factor
            if window_id not in feature_sets:
                continue
                
            if 'training' not in feature_sets[window_id] or factor_id not in feature_sets[window_id]['training']:
                continue
                
            if 'validation' not in feature_sets[window_id] or factor_id not in feature_sets[window_id]['validation']:
                continue
            
            # Get training and validation data
            train_data = feature_sets[window_id]['training'][factor_id]
            val_data = feature_sets[window_id]['validation'][factor_id]
            
            if 'X' not in train_data or 'y' not in train_data or 'X' not in val_data or 'y' not in val_data:
                continue
            
            X_train = train_data['X']
            y_train = train_data['y']
            X_val = val_data['X']
            y_val = val_data['y']
            
            # Skip if any of the data is empty
            if X_train.empty or X_val.empty or len(y_train) == 0 or len(y_val) == 0:
                continue
            
            # Train and evaluate model
            result = train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, params)
            
            if result['success']:
                # Add window and factor information
                result['window_id'] = window_id
                result['factor_id'] = factor_id
                
                # Add to results
                results.append(result)
    
    # Calculate aggregated metrics
    if results:
        # Calculate mean metrics
        aggregated_metrics = {
            'model_type': model_type,
            'params': params,
            'train_rmse_mean': np.mean([r['train_rmse'] for r in results]),
            'train_mae_mean': np.mean([r['train_mae'] for r in results]),
            'train_r2_mean': np.mean([r['train_r2'] for r in results]),
            'val_rmse_mean': np.mean([r['val_rmse'] for r in results]),
            'val_mae_mean': np.mean([r['val_mae'] for r in results]),
            'val_r2_mean': np.mean([r['val_r2'] for r in results]),
            'num_samples': len(results),
            'success': True,
            'individual_results': results
        }
        
        return aggregated_metrics
    else:
        # Return failure
        return {
            'model_type': model_type,
            'params': params,
            'success': False,
            'error': 'No successful evaluations'
        }

def run_grid_search_parallel(param_grids, feature_sets, selected_windows, selected_factors):
    """
    Run grid search in parallel for all model types and parameters.
    
    Parameters:
    -----------
    param_grids : dict
        Dictionary of parameter grids for each model type.
    feature_sets : dict
        Dictionary of feature sets for each window and factor.
    selected_windows : list
        List of selected window IDs.
    selected_factors : list
        List of selected factor IDs.
    
    Returns:
    --------
    dict
        Dictionary of tuning results for each model type.
    """
    print("Running grid search in parallel...")
    
    # Create list of parameter combinations to evaluate
    tasks = []
    
    for model_type, param_grid in param_grids.items():
        parameter_combinations = list(ParameterGrid(param_grid))
        
        for params in parameter_combinations:
            # Create task
            task = (model_type, params, feature_sets, selected_windows, selected_factors)
            tasks.append(task)
    
    # Run tasks in parallel
    print(f"Running {len(tasks)} parameter combinations on {len(selected_windows)} windows and {len(selected_factors)} factors")
    
    # Determine number of processes
    num_processes = min(mp.cpu_count(), len(tasks))
    print(f"Using {num_processes} parallel processes")
    
    # Execute grid search
    results = []
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(evaluate_parameter_combination, tasks), total=len(tasks)))
    
    # Organize results by model type
    tuning_results = {}
    
    for model_type in param_grids:
        tuning_results[model_type] = [r for r in results if r['model_type'] == model_type and r['success']]
    
    return tuning_results

def find_best_parameters(tuning_results):
    """
    Find the best parameters for each model type.
    
    Parameters:
    -----------
    tuning_results : dict
        Dictionary of tuning results for each model type.
    
    Returns:
    --------
    dict
        Dictionary of best parameters for each model type.
    """
    best_params = {}
    
    for model_type, results in tuning_results.items():
        if not results:
            continue
        
        # Sort by validation RMSE (ascending)
        sorted_results = sorted(results, key=lambda x: x['val_rmse_mean'])
        
        # Get best parameters
        best_result = sorted_results[0]
        
        best_params[model_type] = {
            'params': best_result['params'],
            'val_rmse': best_result['val_rmse_mean'],
            'val_mae': best_result['val_mae_mean'],
            'val_r2': best_result['val_r2_mean'],
            'samples': best_result['num_samples']
        }
    
    return best_params

def save_tuning_results(tuning_results, output_file):
    """
    Save tuning results to Excel.
    
    Parameters:
    -----------
    tuning_results : dict
        Dictionary of tuning results for each model type.
    output_file : str
        Path to output Excel file.
    """
    print(f"Saving tuning results to {output_file}...")
    
    # Create a writer for the Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Create summary sheet
        summary_rows = []
        
        for model_type, results in tuning_results.items():
            if not results:
                continue
            
            # Convert to DataFrame
            results_df = pd.DataFrame([
                {
                    'model_type': r['model_type'],
                    'val_rmse': r['val_rmse_mean'],
                    'val_mae': r['val_mae_mean'],
                    'val_r2': r['val_r2_mean'],
                    'train_rmse': r['train_rmse_mean'],
                    'train_mae': r['train_mae_mean'],
                    'train_r2': r['train_r2_mean'],
                    'num_samples': r['num_samples'],
                    **{f'param_{k}': v for k, v in r['params'].items()}
                }
                for r in results
            ])
            
            # Sort by validation RMSE
            results_df = results_df.sort_values('val_rmse')
            
            # Save to sheet
            sheet_name = model_type[:31]  # Excel sheet name limit
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Add best result to summary
            if not results_df.empty:
                best_row = results_df.iloc[0]
                summary_row = {
                    'model_type': model_type,
                    'val_rmse': best_row['val_rmse'],
                    'val_mae': best_row['val_mae'],
                    'val_r2': best_row['val_r2'],
                    'train_rmse': best_row['train_rmse'],
                    'train_mae': best_row['train_mae'],
                    'train_r2': best_row['train_r2'],
                    'num_samples': best_row['num_samples']
                }
                
                # Add parameter columns
                param_cols = [col for col in best_row.index if col.startswith('param_')]
                for col in param_cols:
                    param_name = col.replace('param_', '')
                    summary_row[param_name] = best_row[col]
                
                summary_rows.append(summary_row)
        
        # Save summary
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

def save_best_parameters(best_params, output_file):
    """
    Save best parameters to JSON.
    
    Parameters:
    -----------
    best_params : dict
        Dictionary of best parameters for each model type.
    output_file : str
        Path to output JSON file.
    """
    print(f"Saving best parameters to {output_file}...")
    
    # Convert best parameters to JSON-serializable format
    json_params = {}
    
    for model_type, params in best_params.items():
        json_params[model_type] = {
            'params': params['params'],
            'metrics': {
                'val_rmse': float(params['val_rmse']),
                'val_mae': float(params['val_mae']),
                'val_r2': float(params['val_r2']),
                'samples': int(params['samples'])
            }
        }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(json_params, f, indent=4)

def create_visualizations(tuning_results, best_params, output_file):
    """
    Create visualizations of tuning results.
    
    Parameters:
    -----------
    tuning_results : dict
        Dictionary of tuning results for each model type.
    best_params : dict
        Dictionary of best parameters for each model type.
    output_file : str
        Path to output PDF file.
    """
    print(f"Creating visualizations in {output_file}...")
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Linear Models Hyperparameter Tuning Results',
                fontsize=20, ha='center')
        plt.text(0.5, 0.45, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=14, ha='center')
        plt.text(0.5, 0.4, f'Models tuned: {", ".join(tuning_results.keys())}',
                fontsize=14, ha='center')
        pdf.savefig()
        plt.close()
        
        # Model comparison - Validation metrics
        plt.figure(figsize=(10, 6))
        
        # Create bar data
        models = list(best_params.keys())
        val_rmse = [best_params[m]['val_rmse'] for m in models]
        
        # Create bar chart
        plt.bar(models, val_rmse)
        plt.title('Validation RMSE by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Validation RMSE')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(val_rmse):
            plt.text(i, v + 0.0001, f'{v:.6f}', ha='center')
        
        pdf.savefig()
        plt.close()
        
        # Model comparison - R² values
        plt.figure(figsize=(10, 6))
        
        # Create bar data
        models = list(best_params.keys())
        val_r2 = [best_params[m]['val_r2'] for m in models]
        
        # Create bar chart
        plt.bar(models, val_r2)
        plt.title('Validation R² by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Validation R²')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(val_r2):
            plt.text(i, v + 0.01, f'{v:.6f}', ha='center')
        
        pdf.savefig()
        plt.close()
        
        # Parameter importance for each model type
        for model_type, results in tuning_results.items():
            if not results:
                continue
            
            # Skip NNLS since it has no tunable parameters
            if model_type == 'NNLS':
                continue
                
            # Convert results to DataFrame
            results_df = pd.DataFrame([
                {
                    'val_rmse': r['val_rmse_mean'],
                    'val_r2': r['val_r2_mean'],
                    **r['params']
                }
                for r in results
            ])
            
            # Get all parameters
            params = list(best_params[model_type]['params'].keys())
            
            # Create parameter importance plots
            if params and all(param in results_df.columns for param in params):
                # Create figure with subplots for each parameter
                n_params = len(params)
                n_cols = min(2, n_params)
                n_rows = (n_params + n_cols - 1) // n_cols
                
                plt.figure(figsize=(12, 4 * n_rows))
                plt.suptitle(f'Parameter Importance for {model_type}', fontsize=16)
                
                for i, param in enumerate(params):
                    if param not in results_df.columns:
                        continue
                    
                    # Skip parameters with only one value
                    if results_df[param].nunique() <= 1:
                        continue
                    
                    # Calculate parameter importance
                    param_importance = []
                    param_values = sorted(results_df[param].unique())
                    
                    for val in param_values:
                        val_results = results_df[results_df[param] == val]
                        param_importance.append((val, val_results['val_rmse'].mean()))
                    
                    # Create subplot
                    plt.subplot(n_rows, n_cols, i + 1)
                    
                    # Create bar chart
                    vals = [str(p[0]) for p in param_importance]
                    rmse = [p[1] for p in param_importance]
                    
                    plt.bar(vals, rmse)
                    plt.title(f'{param} vs. Validation RMSE')
                    plt.xlabel(param)
                    plt.ylabel('Validation RMSE')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig()
                plt.close()
            
            # Distribution of validation metrics
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(results_df['val_rmse'], kde=True)
            plt.title(f'{model_type} - Validation RMSE Distribution')
            plt.xlabel('Validation RMSE')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            sns.histplot(results_df['val_r2'], kde=True)
            plt.title(f'{model_type} - Validation R² Distribution')
            plt.xlabel('Validation R²')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Summary of best parameters
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Best Parameters for Each Model Type',
                fontsize=16, fontweight='bold', ha='center')
        
        y_pos = 0.9
        line_height = 0.03
        
        for model_type, params in best_params.items():
            plt.text(0.1, y_pos, f'{model_type}:', fontsize=14, fontweight='bold')
            y_pos -= line_height
            
            plt.text(0.15, y_pos, f'Parameters:', fontsize=12)
            y_pos -= line_height
            
            for param_name, param_value in params['params'].items():
                plt.text(0.2, y_pos, f'{param_name}: {param_value}', fontsize=12)
                y_pos -= line_height
            
            plt.text(0.15, y_pos, f'Metrics:', fontsize=12)
            y_pos -= line_height
            
            plt.text(0.2, y_pos, f'Validation RMSE: {params["val_rmse"]:.6f}', fontsize=12)
            y_pos -= line_height
            
            plt.text(0.2, y_pos, f'Validation R²: {params["val_r2"]:.6f}', fontsize=12)
            y_pos -= line_height
            
            plt.text(0.2, y_pos, f'Sample size: {params["samples"]}', fontsize=12)
            y_pos -= line_height * 2
        
        pdf.savefig()
        plt.close()

def main():
    """
    Main function.
    """
    print("\n" + "="*80)
    print("STEP 10B: TUNE LINEAR MODELS")
    print("="*80)
    
    start_time = time.time()
    
    # File paths
    feature_sets_file = os.path.join("output", "S8_Feature_Sets.h5")
    window_schedule_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    
    # Output files
    tuning_results_file = os.path.join("output", "S10B_Linear_Models_Tuning_Results.xlsx")
    best_params_file = os.path.join("output", "S10B_Linear_Models_Optimal_Params.json")
    visualization_file = os.path.join("output", "S10B_Linear_Models_Tuning_Visualization.pdf")
    
    # Load data
    feature_sets = load_feature_sets(feature_sets_file)
    window_schedule = load_window_schedule(window_schedule_file)
    
    # Select windows and factors for tuning
    selected_windows = select_tuning_windows(window_schedule, NUM_TUNING_WINDOWS)
    selected_factors = select_factors_for_tuning(feature_sets, selected_windows, NUM_FACTORS_FOR_TUNING)
    
    # Define parameter grids
    param_grids = define_parameter_grids()
    
    # Run grid search
    tuning_results = run_grid_search_parallel(param_grids, feature_sets, selected_windows, selected_factors)
    
    # Find best parameters
    best_params = find_best_parameters(tuning_results)
    
    # Print best parameters
    print("\nBest parameters for each model type:")
    for model_type, params in best_params.items():
        print(f"\n{model_type}:")
        print(f"  Parameters: {params['params']}")
        print(f"  Validation RMSE: {params['val_rmse']:.6f}")
        print(f"  Validation R²: {params['val_r2']:.6f}")
    
    # Save results
    save_tuning_results(tuning_results, tuning_results_file)
    save_best_parameters(best_params, best_params_file)
    create_visualizations(tuning_results, best_params, visualization_file)
    
    # Print completion message
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nStep 10B completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Results saved to:")
    print(f"  - {tuning_results_file}")
    print(f"  - {best_params_file}")
    print(f"  - {visualization_file}")
    print("="*80)

if __name__ == "__main__":
    main()