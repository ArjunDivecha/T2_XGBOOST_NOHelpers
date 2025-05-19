#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 11B: Train Linear Models

This script trains four types of linear regression models for all factors and windows:
1. Ordinary Least Squares (OLS)
2. Ridge Regression (with alpha regularization)
3. LASSO Regression (with alpha regularization)
4. Non-Negative Least Squares (NNLS)

INPUT FILES:
- S8_Feature_Sets.h5 (output from Step 8)
  - Contains 4-dimensional feature sets for each factor and window
  - Features: 1m, 3m, 12m, 60m moving averages of the factor itself

- S4_Window_Schedule.xlsx (output from Step 4)
  - Contains window definitions and dates

- S10B_Linear_Models_Optimal_Params.json (output from Step 10B)
  - Contains optimal parameters for each linear model type

OUTPUT FILES:
- S11B_Linear_Models.h5
  - HDF5 file containing trained linear models for all factors and windows
  - Organized by window, factor, and model type

- S11B_Linear_Models_Training_Summary.xlsx
  - Excel file with training performance metrics for all models
  - Includes validation metrics and feature importance

- S11B_Linear_Models_Training_Visualization.pdf
  - PDF with visualizations of training results and performance metrics

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
import warnings
import time
import random
import json
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import re
from scipy.optimize import nnls

# Import scikit-learn modules for linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_SEED = 42
NUM_PROCESSES = min(mp.cpu_count(), 16)  # Use up to 16 processes for parallel training
VERBOSE = True  # Print detailed progress information

# Ensure deterministic behavior
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

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

def load_data():
    """
    Load the window schedule, feature sets, and optimal model parameters.
    
    Returns:
    --------
    tuple
        (window_schedule DataFrame, feature_sets dictionary, optimal_params dictionary)
    """
    print("Loading data...")
    
    # Load window schedule
    window_schedule_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    if not os.path.exists(window_schedule_file):
        raise FileNotFoundError(f"Window schedule file not found: {window_schedule_file}")
    
    window_schedule = pd.read_excel(window_schedule_file)
    print(f"Loaded window schedule with {len(window_schedule)} windows")
    
    # Load feature sets
    feature_sets_file = os.path.join("output", "S8_Feature_Sets.h5")
    if not os.path.exists(feature_sets_file):
        raise FileNotFoundError(f"Feature sets file not found: {feature_sets_file}")
    
    # Initialize dictionary to store feature sets
    feature_sets = {}
    
    with h5py.File(feature_sets_file, 'r') as hf:
        # Check if the expected structure exists
        if 'feature_sets' not in hf:
            raise ValueError("Invalid feature sets file format: 'feature_sets' group not found")
        
        feature_sets_group = hf['feature_sets']
        
        # Iterate through window groups
        for window_name in feature_sets_group:
            # Extract window ID from group name (e.g., "window_1" -> 1)
            window_id = int(window_name.split('_')[1])
            window_group = feature_sets_group[window_name]
            
            feature_sets[window_id] = {}
            
            # Load training, validation, and prediction data for this window
            for split_name in ['training', 'validation', 'prediction']:
                if split_name in window_group:
                    split_group = window_group[split_name]
                    feature_sets[window_id][split_name] = {}
                    
                    # Iterate through factors in this split
                    for factor_key in split_group:
                        factor_group = split_group[factor_key]
                        factor_data = {}
                        
                        # Load X data if available
                        if 'X' in factor_group:
                            X_group = factor_group['X']
                            
                            # Get data and column names
                            X_data = X_group['data'][:]
                            X_columns = [col.decode('utf-8') for col in X_group['columns'][:]]
                            
                            # Create DataFrame
                            factor_data['X'] = pd.DataFrame(X_data, columns=X_columns)
                            factor_data['X_columns'] = X_columns
                        
                        # Load y data if available
                        if 'y' in factor_group:
                            factor_data['y'] = factor_group['y'][:]
                        
                        # Convert factor_key back to original name if needed (replacing '_' with '/')
                        original_factor = factor_key.replace('_', '/')
                        
                        # Store data for this factor
                        feature_sets[window_id][split_name][original_factor] = factor_data
    
    num_windows = len(feature_sets)
    num_factors = len(list(feature_sets.values())[0]['training']) if num_windows > 0 else 0
    
    print(f"Loaded feature sets for {num_windows} windows and approximately {num_factors} factors per window")
    
    # Load optimal parameters
    optimal_params_file = os.path.join("output", "S10B_Linear_Models_Optimal_Params.json")
    if not os.path.exists(optimal_params_file):
        raise FileNotFoundError(f"Optimal parameters file not found: {optimal_params_file}")
    
    with open(optimal_params_file, 'r') as f:
        optimal_params = json.load(f)
    
    print(f"Loaded optimal parameters for {len(optimal_params)} model types")
    
    return window_schedule, feature_sets, optimal_params

def get_model_params(model_type, optimal_params):
    """
    Get the optimal parameters for a specific model type.
    
    Parameters:
    -----------
    model_type : str
        Type of linear model ('OLS', 'Ridge', 'Lasso', or 'NNLS').
    optimal_params : dict
        Dictionary containing optimal parameters for all model types.
        
    Returns:
    --------
    dict
        Dictionary of model parameters.
    """
    if model_type not in optimal_params:
        print(f"Warning: No optimal parameters found for {model_type}. Using default parameters.")
        if model_type == 'OLS':
            return {'fit_intercept': True}
        elif model_type == 'Ridge':
            return {'alpha': 1.0, 'fit_intercept': True}
        elif model_type == 'Lasso':
            return {'alpha': 0.1, 'fit_intercept': True, 'max_iter': 10000}
        elif model_type == 'NNLS':
            return {'dummy_param': 1}
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    
    return optimal_params[model_type]['params']

def get_factor_list(feature_sets):
    """
    Get a list of all factors in the feature sets.
    
    Parameters:
    -----------
    feature_sets : dict
        Dictionary of feature sets by factor and window.
        
    Returns:
    --------
    list
        List of all factor IDs.
    """
    # Get the first window
    first_window_id = min(feature_sets.keys())
    
    # Get all factors in the training set
    all_factors = []
    if 'training' in feature_sets[first_window_id]:
        all_factors = list(feature_sets[first_window_id]['training'].keys())
    
    # Print a sample of factor names
    if VERBOSE:
        print("Sample factor names:")
        for factor in sorted(all_factors)[:5]:  # Just show the first 5
            print(f"  - {factor}")
    
    print(f"Found {len(all_factors)} factors")
    
    return sorted(all_factors)

def create_model(model_type, params):
    """
    Create a model instance with the provided parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of linear model ('OLS', 'Ridge', 'Lasso', or 'NNLS').
    params : dict
        Model parameters.
        
    Returns:
    --------
    object
        Initialized model object.
    """
    if model_type == 'OLS':
        return LinearRegression(**params)
    elif model_type == 'Ridge':
        return Ridge(**params)
    elif model_type == 'Lasso':
        return Lasso(**params)
    elif model_type == 'NNLS':
        return NNLSRegressor(**params)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def train_model(model_type, X_train, y_train, params):
    """
    Train a model on the given data.
    
    Parameters:
    -----------
    model_type : str
        Type of linear model ('OLS', 'Ridge', 'Lasso', or 'NNLS').
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training targets.
    params : dict
        Model parameters.
        
    Returns:
    --------
    object
        Trained model object.
    """
    # Create and train model
    model = create_model(model_type, params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model_type, model, X, y):
    """
    Evaluate a trained model on the given data.
    
    Parameters:
    -----------
    model_type : str
        Type of linear model ('OLS', 'Ridge', 'Lasso', or 'NNLS').
    model : object
        Trained model object.
    X : pd.DataFrame
        Features.
    y : np.ndarray
        Targets.
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics.
    """
    # Generate predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y - y_pred) / (np.abs(y) + 1e-8))) * 100  # Add small epsilon to avoid division by zero
    
    # Calculate directional accuracy (up/down movement)
    actual_direction = np.sign(y)
    predicted_direction = np.sign(y_pred)
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'y_pred': y_pred
    }

def get_feature_importance(model_type, model, feature_names):
    """
    Get feature importance from a trained model.
    
    Parameters:
    -----------
    model_type : str
        Type of linear model ('OLS', 'Ridge', 'Lasso', or 'NNLS').
    model : object
        Trained model object.
    feature_names : list
        List of feature names.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature names and importance scores.
    """
    # Get coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
    else:
        raise ValueError(f"Model of type {model_type} does not have coefficients")
    
    # Create DataFrame with feature names and coefficients
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # Sort by absolute coefficient value (descending)
    importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    # Drop the abs_coefficient column (used only for sorting)
    importance_df = importance_df.drop('abs_coefficient', axis=1)
    
    return importance_df

def train_and_evaluate_models_for_factor(window_id, factor_id, feature_sets, optimal_params):
    """
    Train and evaluate all model types for a specific factor and window.
    
    Parameters:
    -----------
    window_id : int
        Window ID.
    factor_id : str
        Factor ID.
    feature_sets : dict
        Dictionary of feature sets by factor and window.
    optimal_params : dict
        Dictionary of optimal parameters for all model types.
        
    Returns:
    --------
    dict
        Dictionary of trained models, evaluation metrics, and feature importance.
    """
    # Get training and validation data for this factor and window
    try:
        factor_data = feature_sets[window_id]['training'][factor_id]
        X_train = factor_data['X']
        y_train = factor_data['y']
        
        factor_data_val = feature_sets[window_id]['validation'][factor_id]
        X_val = factor_data_val['X']
        y_val = factor_data_val['y']
    except KeyError:
        if VERBOSE:
            print(f"Warning: Missing data for window {window_id}, factor {factor_id}")
        return None
    
    # Define model types
    model_types = ['OLS', 'Ridge', 'Lasso', 'NNLS']
    
    # Initialize results dictionary
    results = {
        'window_id': window_id,
        'factor_id': factor_id,
        'models': {},
        'train_metrics': {},
        'val_metrics': {},
        'feature_importance': {}
    }
    
    # Train and evaluate each model type
    for model_type in model_types:
        # Get model parameters
        params = get_model_params(model_type, optimal_params)
        
        try:
            # Train model
            model = train_model(model_type, X_train, y_train, params)
            
            # Evaluate on training set
            train_metrics = evaluate_model(model_type, model, X_train, y_train)
            
            # Evaluate on validation set
            val_metrics = evaluate_model(model_type, model, X_val, y_val)
            
            # Get feature importance
            feature_importance = get_feature_importance(model_type, model, X_train.columns)
            
            # Store results
            results['models'][model_type] = model
            results['train_metrics'][model_type] = train_metrics
            results['val_metrics'][model_type] = val_metrics
            results['feature_importance'][model_type] = feature_importance
        except Exception as e:
            if VERBOSE:
                print(f"Error training {model_type} model for window {window_id}, factor {factor_id}: {e}")
    
    return results

def train_models_for_window(window_id, factor_ids, feature_sets, optimal_params):
    """
    Train and evaluate all models for all factors in a specific window.
    
    Parameters:
    -----------
    window_id : int
        Window ID.
    factor_ids : list
        List of factor IDs to process.
    feature_sets : dict
        Dictionary of feature sets by factor and window.
    optimal_params : dict
        Dictionary of optimal parameters for all model types.
        
    Returns:
    --------
    dict
        Dictionary of results by factor ID.
    """
    if VERBOSE:
        print(f"Processing window {window_id}...")
    
    # Initialize results dictionary
    window_results = {}
    
    # Process each factor
    for factor_id in factor_ids:
        # Train and evaluate models for this factor
        factor_results = train_and_evaluate_models_for_factor(
            window_id, factor_id, feature_sets, optimal_params
        )
        
        # Store results if available
        if factor_results is not None:
            window_results[factor_id] = factor_results
    
    return {window_id: window_results}

def save_models_to_h5(all_results, output_file):
    """
    Save trained models and related data to an HDF5 file.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of model results by window and factor.
    output_file : str
        Path to output HDF5 file.
    """
    print(f"Saving models to {output_file}...")
    
    with h5py.File(output_file, 'w') as hf:
        # Create group for models
        models_group = hf.create_group('models')
        
        # Add metadata
        hf.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hf.attrs['num_windows'] = len(all_results)
        
        # Iterate through windows
        for window_id, window_results in tqdm(all_results.items(), desc="Saving windows"):
            window_group = models_group.create_group(f'window_{window_id}')
            
            # Iterate through factors
            for factor_id, factor_results in window_results.items():
                # Replace forward slashes with underscores in factor_id for HDF5 compatibility
                factor_key = factor_id.replace('/', '_')
                factor_group = window_group.create_group(factor_key)
                
                # Store metadata about the factor
                factor_group.attrs['original_factor_id'] = factor_id
                
                # Create group for each model type
                models_subgroup = factor_group.create_group('models')
                metrics_subgroup = factor_group.create_group('metrics')
                
                # Iterate through model types
                for model_type, model in factor_results['models'].items():
                    model_group = models_subgroup.create_group(model_type)
                    
                    # Store model parameters and coefficients
                    if model_type in ('OLS', 'Ridge', 'Lasso', 'NNLS'):
                        # All these models have coefficients
                        model_group.create_dataset('coef', data=model.coef_)
                        
                        # Store intercept if available (NNLS doesn't have one)
                        if hasattr(model, 'intercept_'):
                            model_group.create_dataset('intercept', data=np.array([model.intercept_]))
                        else:
                            model_group.create_dataset('intercept', data=np.array([0.0]))
                    
                    # Store model parameters
                    param_group = model_group.create_group('params')
                    for param_name, param_value in model.get_params().items():
                        if isinstance(param_value, (int, float, bool, str)):
                            param_group.attrs[param_name] = param_value
                    
                    # Store feature names
                    if 'feature_importance' in factor_results and model_type in factor_results['feature_importance']:
                        feature_names = factor_results['feature_importance'][model_type]['feature'].values
                        feature_names_bytes = [name.encode('utf-8') for name in feature_names]
                        model_group.create_dataset('feature_names', data=feature_names_bytes)
                    
                    # Store metrics
                    metric_group = metrics_subgroup.create_group(model_type)
                    
                    # Store training metrics
                    train_metrics_group = metric_group.create_group('train')
                    for metric_name, metric_value in factor_results['train_metrics'][model_type].items():
                        if metric_name != 'y_pred':  # Skip predicted values to save space
                            train_metrics_group.attrs[metric_name] = metric_value
                    
                    # Store validation metrics
                    val_metrics_group = metric_group.create_group('val')
                    for metric_name, metric_value in factor_results['val_metrics'][model_type].items():
                        if metric_name != 'y_pred':  # Skip predicted values to save space
                            val_metrics_group.attrs[metric_name] = metric_value

def create_training_summary(all_results, window_schedule):
    """
    Create a summary of training results for analysis and visualization.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of model results by window and factor.
    window_schedule : pd.DataFrame
        DataFrame containing the window schedule.
        
    Returns:
    --------
    dict
        Dictionary of summary DataFrames.
    """
    print("Creating training summary...")
    
    # Initialize lists to store results
    rows = []
    
    # Model types
    model_types = ['OLS', 'Ridge', 'Lasso', 'NNLS']
    
    # Iterate through windows
    for window_id, window_results in all_results.items():
        # Get window information
        window_info = window_schedule.loc[window_schedule['Window_ID'] == window_id].iloc[0]
        
        # Iterate through factors
        for factor_id, factor_results in window_results.items():
            # Iterate through model types
            for model_type in model_types:
                if model_type not in factor_results['train_metrics'] or model_type not in factor_results['val_metrics']:
                    continue
                
                # Get metrics
                train_metrics = factor_results['train_metrics'][model_type]
                val_metrics = factor_results['val_metrics'][model_type]
                
                # Get feature importance
                if model_type in factor_results['feature_importance']:
                    feature_importance = factor_results['feature_importance'][model_type]
                    
                    # Find top features
                    top_features = feature_importance.iloc[:3]['feature'].tolist() if not feature_importance.empty else []
                    top_features_str = ', '.join(top_features)
                else:
                    top_features_str = ''
                
                # Add row
                row = {
                    'window_id': window_id,
                    'pred_month': window_info['Prediction_Index'] if 'Prediction_Index' in window_info else None,
                    'pred_date': window_info['Prediction_Date'] if 'Prediction_Date' in window_info else None,
                    'training_start': window_info['Training_Start_Date'] if 'Training_Start_Date' in window_info else None,
                    'training_end': window_info['Training_End_Date'] if 'Training_End_Date' in window_info else None,
                    'factor_id': factor_id,
                    'model_type': model_type,
                    'top_features': top_features_str
                }
                
                # Add training metrics
                for metric_name, metric_value in train_metrics.items():
                    if metric_name != 'y_pred':  # Skip predicted values
                        row[f'train_{metric_name}'] = metric_value
                
                # Add validation metrics
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != 'y_pred':  # Skip predicted values
                        row[f'val_{metric_name}'] = metric_value
                
                rows.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(rows)
    
    # Create aggregated summaries
    model_summary = summary_df.groupby('model_type').agg({
        'train_rmse': 'mean',
        'train_mae': 'mean',
        'train_r2': 'mean',
        'train_directional_accuracy': 'mean',
        'val_rmse': 'mean',
        'val_mae': 'mean',
        'val_r2': 'mean',
        'val_directional_accuracy': 'mean'
    }).reset_index()
    
    factor_summary = summary_df.groupby(['factor_id', 'model_type']).agg({
        'train_rmse': 'mean',
        'train_r2': 'mean',
        'val_rmse': 'mean',
        'val_r2': 'mean',
        'val_directional_accuracy': 'mean'
    }).reset_index()
    
    window_summary = summary_df.groupby(['window_id', 'pred_date', 'model_type']).agg({
        'train_rmse': 'mean',
        'train_r2': 'mean',
        'val_rmse': 'mean',
        'val_r2': 'mean',
        'val_directional_accuracy': 'mean'
    }).reset_index()
    
    # Create dictionary of summary DataFrames
    summary = {
        'full': summary_df,
        'model': model_summary,
        'factor': factor_summary,
        'window': window_summary
    }
    
    return summary

def save_summary_to_excel(summary, output_file):
    """
    Save training summary to an Excel file.
    
    Parameters:
    -----------
    summary : dict
        Dictionary of summary DataFrames.
    output_file : str
        Path to output Excel file.
    """
    print(f"Saving summary to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save each summary DataFrame to a separate sheet
        summary['full'].to_excel(writer, sheet_name='Full_Results', index=False)
        summary['model'].to_excel(writer, sheet_name='Model_Summary', index=False)
        summary['factor'].to_excel(writer, sheet_name='Factor_Summary', index=False)
        summary['window'].to_excel(writer, sheet_name='Window_Summary', index=False)
        
        # Create a pivot table of model performance by factor
        pivot_df = pd.pivot_table(
            summary['factor'],
            values='val_r2',
            index='factor_id',
            columns='model_type'
        )
        
        # Add a "best model" column
        pivot_df['best_model'] = pivot_df.idxmax(axis=1)
        
        # Save pivot table
        pivot_df.to_excel(writer, sheet_name='Model_by_Factor')

def create_visualizations(summary, output_file):
    """
    Create visualizations of training results and save to PDF.
    
    Parameters:
    -----------
    summary : dict
        Dictionary of summary DataFrames.
    output_file : str
        Path to output PDF file.
    """
    print(f"Creating visualizations and saving to {output_file}...")
    
    with PdfPages(output_file) as pdf:
        # Set style
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training RMSE
        sns.barplot(x='model_type', y='train_rmse', data=summary['model'], ax=axes[0, 0])
        axes[0, 0].set_title('Training RMSE by Model Type')
        axes[0, 0].set_xlabel('Model Type')
        axes[0, 0].set_ylabel('RMSE')
        
        # Validation RMSE
        sns.barplot(x='model_type', y='val_rmse', data=summary['model'], ax=axes[0, 1])
        axes[0, 1].set_title('Validation RMSE by Model Type')
        axes[0, 1].set_xlabel('Model Type')
        axes[0, 1].set_ylabel('RMSE')
        
        # Training R²
        sns.barplot(x='model_type', y='train_r2', data=summary['model'], ax=axes[1, 0])
        axes[1, 0].set_title('Training R² by Model Type')
        axes[1, 0].set_xlabel('Model Type')
        axes[1, 0].set_ylabel('R²')
        
        # Validation R²
        sns.barplot(x='model_type', y='val_r2', data=summary['model'], ax=axes[1, 1])
        axes[1, 1].set_title('Validation R² by Model Type')
        axes[1, 1].set_xlabel('Model Type')
        axes[1, 1].set_ylabel('R²')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 2. Directional Accuracy by Model Type
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y='val_directional_accuracy', data=summary['model'])
        plt.title('Validation Directional Accuracy by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Directional Accuracy (%)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 3. Distribution of Validation R² by Model Type
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='model_type', y='val_r2', data=summary['full'])
        plt.title('Distribution of Validation R² by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Validation R²')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 4. Distribution of Validation RMSE by Model Type
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='model_type', y='val_rmse', data=summary['full'])
        plt.title('Distribution of Validation RMSE by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Validation RMSE')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 5. Model Performance Over Time (Windows)
        plt.figure(figsize=(15, 8))
        
        # Convert pred_date to datetime if it's not already
        time_series_df = summary['window'].copy()
        if 'pred_date' in time_series_df.columns:
            if not pd.api.types.is_datetime64_dtype(time_series_df['pred_date']):
                time_series_df['pred_date'] = pd.to_datetime(time_series_df['pred_date'])
            
            # Plot validation R² over time
            for model_type in time_series_df['model_type'].unique():
                model_data = time_series_df[time_series_df['model_type'] == model_type]
                plt.plot(model_data['pred_date'], model_data['val_r2'], label=model_type)
            
            plt.title('Validation R² Over Time by Model Type')
            plt.xlabel('Prediction Date')
            plt.ylabel('Validation R²')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 6. Best Model Distribution
        # Calculate the count of best model by factor
        best_models = summary['factor'].groupby(['factor_id', 'model_type'])['val_r2'].mean().reset_index()
        best_models = best_models.loc[best_models.groupby('factor_id')['val_r2'].idxmax()]
        best_model_counts = best_models['model_type'].value_counts().reset_index()
        best_model_counts.columns = ['model_type', 'count']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y='count', data=best_model_counts)
        plt.title('Number of Factors Where Each Model Type Performs Best')
        plt.xlabel('Model Type')
        plt.ylabel('Count')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 7. Correlation between Training and Validation R²
        plt.figure(figsize=(10, 10))
        
        for model_type in summary['full']['model_type'].unique():
            model_data = summary['full'][summary['full']['model_type'] == model_type]
            plt.scatter(model_data['train_r2'], model_data['val_r2'], alpha=0.5, label=model_type)
        
        plt.title('Correlation between Training and Validation R²')
        plt.xlabel('Training R²')
        plt.ylabel('Validation R²')
        plt.legend()
        plt.grid(True)
        # Add diagonal line (y = x)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def main():
    """
    Main execution function.
    """
    print("=== Step 11B: Training Linear Models ===")
    
    start_time = time.time()
    
    # Output files
    models_output_file = os.path.join("output", "S11B_Linear_Models.h5")
    summary_output_file = os.path.join("output", "S11B_Linear_Models_Training_Summary.xlsx")
    visualization_output_file = os.path.join("output", "S11B_Linear_Models_Training_Visualization.pdf")
    
    # Step 11B.1: Load data
    window_schedule, feature_sets, optimal_params = load_data()
    
    # Step 11B.2: Get the list of unique factors to process
    factor_ids = get_factor_list(feature_sets)
    print(f"Processing {len(window_schedule)} windows and {len(factor_ids)} factors...")
    
    # Step 11B.3: Train models for all windows and factors in parallel
    window_ids = sorted(list(feature_sets.keys()))
    
    # Setup parallel processing
    print(f"Using {NUM_PROCESSES} parallel processes for training...")
    
    # Process all windows (sequentially or in parallel depending on config)
    if NUM_PROCESSES > 1:
        # Parallel processing
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            window_batches = [(window_id, factor_ids, feature_sets, optimal_params) for window_id in window_ids]
            all_results_list = list(tqdm(pool.starmap(train_models_for_window, window_batches), total=len(window_ids)))
    else:
        # Sequential processing
        all_results_list = []
        for window_id in tqdm(window_ids):
            window_result = train_models_for_window(window_id, factor_ids, feature_sets, optimal_params)
            all_results_list.append(window_result)
    
    # Combine results from all windows
    all_results = {}
    for window_result in all_results_list:
        all_results.update(window_result)
    
    # Step 11B.4: Save models to HDF5 file
    save_models_to_h5(all_results, models_output_file)
    
    # Step 11B.5: Create summary of training results
    summary = create_training_summary(all_results, window_schedule)
    
    # Step 11B.6: Save summary to Excel
    save_summary_to_excel(summary, summary_output_file)
    
    # Step 11B.7: Create and save visualizations
    create_visualizations(summary, visualization_output_file)
    
    # Print completion summary
    elapsed_time = time.time() - start_time
    total_models = len(window_ids) * len(factor_ids) * 4  # 4 model types per factor
    
    print(f"\nStep 11B completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Trained approximately {total_models} models")
    print(f"Models saved to: {models_output_file}")
    print(f"Summary saved to: {summary_output_file}")
    print(f"Visualizations saved to: {visualization_output_file}")

if __name__ == "__main__":
    main()