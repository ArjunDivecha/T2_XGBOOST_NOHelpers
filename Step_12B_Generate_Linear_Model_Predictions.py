#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 12B: Generate Linear Model Predictions

This script generates predictions for all factors using the trained linear models
from Step 11B. It implements four types of linear models:
1. Ordinary Least Squares (OLS)
2. Ridge Regression (with alpha regularization)
3. LASSO Regression (with alpha regularization)
4. Non-Negative Least Squares (NNLS)

INPUT FILES:
- S11B_Linear_Models.h5 (output from Step 11B)
  - HDF5 file containing trained linear models for all factors and windows
  - Organized by window, factor, and model type

- S4_Window_Schedule.xlsx (output from Step 4)
  - Excel file with window schedule information

- S8_Feature_Sets.h5 (output from Step 8)
  - HDF5 file with feature sets
  - Contains 4-dimensional feature sets for prediction

OUTPUT FILES:
- S12B_Linear_Model_Predictions.h5
  - HDF5 file containing all predictions for all factors, windows, and model types

- S12B_Linear_Model_Predictions_Matrix.xlsx
  - Excel file with predictions organized in a matrix format (dates as rows, factors as columns)
  - One sheet per model type

- S12B_Prediction_Visualization.pdf
  - PDF with visualizations of predictions, performance analysis, etc.

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
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scipy.optimize import nnls

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_SEED = 42
VERBOSE = True  # Print detailed progress information

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Create output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Custom NNLSRegressor class for prediction
class NNLSRegressor:
    """
    Non-Negative Least Squares regressor class with a scikit-learn compatible interface.
    """
    def __init__(self, coef, intercept=0.0):
        self.coef_ = coef
        self.intercept_ = intercept
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.coef_) + self.intercept_

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
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Window schedule file not found: {file_path}")
    
    window_schedule = pd.read_excel(file_path)
    print(f"Loaded window schedule with {len(window_schedule)} windows")
    
    return window_schedule

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
        Dictionary of feature sets for prediction.
    """
    print(f"Loading feature sets from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature sets file not found: {file_path}")
    
    # Initialize dictionary to store feature sets
    feature_sets = {}
    
    with h5py.File(file_path, 'r') as hf:
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
            
            # We only need prediction data for this step
            if 'prediction' in window_group:
                prediction_group = window_group['prediction']
                feature_sets[window_id]['prediction'] = {}
                
                # Iterate through factors in prediction split
                for factor_key in prediction_group:
                    factor_group = prediction_group[factor_key]
                    factor_data = {}
                    
                    # Load X data for prediction if available
                    if 'X' in factor_group:
                        X_group = factor_group['X']
                        
                        # Get data and column names
                        X_data = X_group['data'][:]
                        X_columns = [col.decode('utf-8') for col in X_group['columns'][:]]
                        
                        # Create DataFrame
                        factor_data['X'] = pd.DataFrame(X_data, columns=X_columns)
                        factor_data['X_columns'] = X_columns
                    
                    # Convert factor_key back to original name if needed (replacing '_' with '/')
                    original_factor = factor_key.replace('_', '/')
                    
                    # Store data for this factor
                    feature_sets[window_id]['prediction'][original_factor] = factor_data
    
    num_windows = len(feature_sets)
    num_factors = sum(len(window.get('prediction', {})) for window in feature_sets.values()) // max(1, num_windows)
    
    print(f"Loaded feature sets for {num_windows} windows and approximately {num_factors} factors per window")
    
    return feature_sets

def load_trained_models(file_path):
    """
    Load trained models from HDF5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to the trained models HDF5 file.
    
    Returns:
    --------
    dict
        Dictionary of trained models for prediction.
    """
    print(f"Loading trained models from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trained models file not found: {file_path}")
    
    # Initialize dictionary to store trained models
    models = {}
    
    with h5py.File(file_path, 'r') as hf:
        # Check if the expected structure exists
        if 'models' not in hf:
            raise ValueError("Invalid models file format: 'models' group not found")
        
        models_group = hf['models']
        
        # Iterate through window groups
        for window_name in models_group:
            # Extract window ID from group name (e.g., "window_1" -> 1)
            window_id = int(window_name.split('_')[1])
            window_group = models_group[window_name]
            
            models[window_id] = {}
            
            # Iterate through factors
            for factor_key in window_group:
                factor_group = window_group[factor_key]
                
                # Get original factor name
                original_factor = factor_key
                if 'original_factor_id' in factor_group.attrs:
                    original_factor = factor_group.attrs['original_factor_id']
                
                models[window_id][original_factor] = {}
                
                # Load models for this factor
                if 'models' in factor_group:
                    models_subgroup = factor_group['models']
                    
                    # Iterate through model types
                    for model_type in models_subgroup:
                        model_group = models_subgroup[model_type]
                        
                        # Extract model coefficients and intercept
                        if 'coef' in model_group:
                            coef = model_group['coef'][:]
                            
                            # Get intercept if available
                            intercept = 0.0
                            if 'intercept' in model_group:
                                intercept = model_group['intercept'][0]
                            
                            # Create model object for prediction
                            if model_type == 'NNLS':
                                model = NNLSRegressor(coef, intercept)
                            else:
                                # For OLS, Ridge, and Lasso, we only need the coefficients for prediction
                                model = NNLSRegressor(coef, intercept)  # Use the same simple class
                            
                            # Store model
                            models[window_id][original_factor][model_type] = {
                                'model': model,
                                'coef': coef,
                                'intercept': intercept
                            }
                            
                            # Store feature names if available
                            if 'feature_names' in model_group:
                                feature_names = [name.decode('utf-8') for name in model_group['feature_names'][:]]
                                models[window_id][original_factor][model_type]['feature_names'] = feature_names
    
    num_windows = len(models)
    num_factors = sum(len(window) for window in models.values()) // max(1, num_windows)
    num_models = sum(sum(len(factor) for factor in window.values()) for window in models.values())
    
    print(f"Loaded {num_models} trained models for {num_windows} windows and {num_factors} factors")
    
    return models

def generate_predictions(window_id, models, feature_sets, window_schedule):
    """
    Generate predictions for a specific window.
    
    Parameters:
    -----------
    window_id : int
        Window ID.
    models : dict
        Dictionary of trained models.
    feature_sets : dict
        Dictionary of feature sets.
    window_schedule : pandas.DataFrame
        DataFrame containing window schedule.
    
    Returns:
    --------
    list
        List of prediction dictionaries.
    """
    if VERBOSE:
        print(f"Generating predictions for window {window_id}...")
    
    # Get prediction date from window schedule
    window_info = window_schedule[window_schedule['Window_ID'] == window_id]
    if window_info.empty:
        print(f"Warning: Window {window_id} not found in schedule")
        return []
    
    prediction_date = window_info.iloc[0]['Prediction_Date']
    
    # Get models and feature sets for this window
    window_models = models.get(window_id, {})
    window_features = feature_sets.get(window_id, {}).get('prediction', {})
    
    # Storage for predictions
    predictions = []
    
    # Model types
    model_types = ['OLS', 'Ridge', 'Lasso', 'NNLS']
    
    # Iterate through factors
    for factor_id in window_models:
        # Get factor models
        factor_models = window_models[factor_id]
        
        # Get factor features
        if factor_id not in window_features:
            if VERBOSE:
                print(f"Warning: Features not found for window {window_id}, factor {factor_id}")
            continue
        
        factor_features = window_features[factor_id]
        
        # Check if we have X data for prediction
        if 'X' not in factor_features:
            if VERBOSE:
                print(f"Warning: X data not found for window {window_id}, factor {factor_id}")
            continue
        
        X_pred = factor_features['X']
        
        # Skip if X is empty
        if X_pred.empty:
            if VERBOSE:
                print(f"Warning: Empty X data for window {window_id}, factor {factor_id}")
            continue
        
        # Generate predictions for each model type
        for model_type in model_types:
            if model_type not in factor_models:
                continue
            
            try:
                # Get model
                model_info = factor_models[model_type]
                model = model_info['model']
                
                # Make prediction
                prediction = model.predict(X_pred)[0]  # Get first (and only) value
                
                # Add to predictions
                predictions.append({
                    'window_id': window_id,
                    'factor_id': factor_id,
                    'model_type': model_type,
                    'prediction_date': prediction_date,
                    'prediction': prediction
                })
                
            except Exception as e:
                if VERBOSE:
                    print(f"Error predicting for window {window_id}, factor {factor_id}, model {model_type}: {e}")
    
    return predictions

def process_all_windows(models, feature_sets, window_schedule):
    """
    Process all windows and generate predictions.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    feature_sets : dict
        Dictionary of feature sets.
    window_schedule : pandas.DataFrame
        DataFrame containing window schedule.
    
    Returns:
    --------
    list
        List of all prediction dictionaries.
    """
    print("Generating predictions for all windows...")
    
    # Get window IDs to process
    window_ids = sorted(models.keys())
    
    # Storage for all predictions
    all_predictions = []
    
    # Process each window
    for window_id in tqdm(window_ids):
        window_predictions = generate_predictions(window_id, models, feature_sets, window_schedule)
        all_predictions.extend(window_predictions)
    
    print(f"Generated {len(all_predictions)} predictions in total")
    
    return all_predictions

def save_predictions_to_h5(predictions, output_file):
    """
    Save predictions to HDF5 file.
    
    Parameters:
    -----------
    predictions : list
        List of prediction dictionaries.
    output_file : str
        Path to output HDF5 file.
    """
    print(f"Saving predictions to {output_file}...")
    
    # Convert predictions to a structured format for HDF5
    structured_predictions = {}
    
    # First, organize by model type
    for model_type in ['OLS', 'Ridge', 'Lasso', 'NNLS']:
        structured_predictions[model_type] = {}
        
        # Get predictions for this model type
        model_predictions = [p for p in predictions if p['model_type'] == model_type]
        
        # Organize by window
        for prediction in model_predictions:
            window_id = prediction['window_id']
            factor_id = prediction['factor_id']
            prediction_date = prediction['prediction_date']
            prediction_value = prediction['prediction']
            
            # Create window entry if not exists
            if window_id not in structured_predictions[model_type]:
                structured_predictions[model_type][window_id] = {
                    'date': prediction_date,
                    'factors': {}
                }
            
            # Add factor prediction
            structured_predictions[model_type][window_id]['factors'][factor_id] = prediction_value
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as hf:
        # Add metadata
        hf.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create group for each model type
        for model_type, model_data in structured_predictions.items():
            model_group = hf.create_group(model_type)
            
            # Create group for each window
            for window_id, window_data in model_data.items():
                window_group = model_group.create_group(f'window_{window_id}')
                
                # Store date
                window_group.attrs['date'] = str(window_data['date'])
                
                # Store factor predictions
                factors_group = window_group.create_group('factors')
                for factor_id, prediction_value in window_data['factors'].items():
                    # HDF5 doesn't allow '/' in dataset names, so we need to replace it
                    safe_factor_id = factor_id.replace('/', '_')
                    factors_group.create_dataset(safe_factor_id, data=np.array([prediction_value]))
                    
                    # Store original factor ID for reference
                    factors_group[safe_factor_id].attrs['original_factor_id'] = factor_id

def create_prediction_matrix(predictions):
    """
    Create prediction matrices for each model type.
    
    Parameters:
    -----------
    predictions : list
        List of prediction dictionaries.
    
    Returns:
    --------
    dict
        Dictionary of prediction matrices by model type.
    """
    print("Creating prediction matrices...")
    
    # Create DataFrames for each model type
    matrix_by_model = {}
    
    # Model types
    model_types = ['OLS', 'Ridge', 'Lasso', 'NNLS']
    
    for model_type in model_types:
        # Get predictions for this model type
        model_preds = [p for p in predictions if p['model_type'] == model_type]
        
        if not model_preds:
            print(f"No predictions for {model_type}")
            continue
        
        # Create DataFrame
        df = pd.DataFrame(model_preds)
        
        # Pivot to create matrix (dates as rows, factors as columns)
        matrix = df.pivot(index='prediction_date', columns='factor_id', values='prediction')
        
        # Sort by date
        matrix = matrix.sort_index()
        
        # Store
        matrix_by_model[model_type] = matrix
    
    return matrix_by_model

def save_prediction_matrix(prediction_matrices, output_file):
    """
    Save prediction matrices to Excel file.
    
    Parameters:
    -----------
    prediction_matrices : dict
        Dictionary of prediction matrices by model type.
    output_file : str
        Path to output Excel file.
    """
    print(f"Saving prediction matrices to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save each model's predictions to a separate sheet
        for model_type, matrix in prediction_matrices.items():
            # Reset index to make prediction_date a column
            df = matrix.reset_index()
            
            # Save to Excel
            df.to_excel(writer, sheet_name=model_type, index=False)
        
        # Create a summary sheet
        summary_rows = []
        for model_type, matrix in prediction_matrices.items():
            # Get basic statistics
            summary_rows.append({
                'model_type': model_type,
                'num_dates': matrix.shape[0],
                'num_factors': matrix.shape[1],
                'mean_prediction': matrix.values.mean(),
                'median_prediction': np.median(matrix.values),
                'min_prediction': matrix.values.min(),
                'max_prediction': matrix.values.max()
            })
        
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

def create_visualizations(prediction_matrices, output_file):
    """
    Create visualizations of predictions.
    
    Parameters:
    -----------
    prediction_matrices : dict
        Dictionary of prediction matrices by model type.
    output_file : str
        Path to output PDF file.
    """
    print(f"Creating visualizations and saving to {output_file}...")
    
    with PdfPages(output_file) as pdf:
        # Set style
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Linear Model Predictions',
                fontsize=24, ha='center')
        plt.text(0.5, 0.45, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=14, ha='center')
        plt.text(0.5, 0.4, f'Model types: {", ".join(prediction_matrices.keys())}',
                fontsize=14, ha='center')
        pdf.savefig()
        plt.close()
        
        # 1. Overall prediction distribution by model type
        plt.figure(figsize=(12, 8))
        
        # Extract all predictions by model type
        all_predictions = {}
        for model_type, matrix in prediction_matrices.items():
            all_predictions[model_type] = matrix.values.flatten()
        
        # Create violin plot
        plt.violinplot([all_predictions[model] for model in prediction_matrices.keys()],
                      showmeans=True, showmedians=True)
        
        # Set x-axis
        plt.xticks(range(1, len(prediction_matrices) + 1), prediction_matrices.keys())
        
        # Labels
        plt.title('Prediction Distribution by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Prediction Value')
        plt.grid(True, alpha=0.3)
        
        pdf.savefig()
        plt.close()
        
        # 2. Time series of predictions for selected factors
        for model_type, matrix in prediction_matrices.items():
            plt.figure(figsize=(15, 10))
            
            # Select a few factors (up to 10)
            if matrix.shape[1] > 10:
                selected_factors = list(matrix.columns[:10])  # Take first 10
            else:
                selected_factors = list(matrix.columns)
            
            # Plot time series for each selected factor
            for factor in selected_factors:
                plt.plot(matrix.index, matrix[factor], label=factor)
            
            # Labels
            plt.title(f'{model_type} Predictions Over Time')
            plt.xlabel('Date')
            plt.ylabel('Prediction Value')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            pdf.savefig()
            plt.close()
        
        # 3. Heatmap of correlation between model predictions
        if len(prediction_matrices) > 1:
            plt.figure(figsize=(12, 10))
            
            # Merge all model predictions into one DataFrame
            all_dfs = []
            for model_type, matrix in prediction_matrices.items():
                # Flatten matrix into a Series
                flattened = matrix.values.flatten()
                all_dfs.append(pd.Series(flattened, name=model_type))
            
            # Create correlation DataFrame
            corr_df = pd.DataFrame(all_dfs).T.corr()
            
            # Create heatmap
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5)
            
            # Labels
            plt.title('Correlation Between Model Predictions')
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()
        
        # 4. Boxplot of predictions by model type
        plt.figure(figsize=(12, 8))
        
        # Extract all predictions by model type
        all_predictions = []
        model_labels = []
        for model_type, matrix in prediction_matrices.items():
            all_predictions.append(matrix.values.flatten())
            model_labels.append(model_type)
        
        # Create boxplot
        plt.boxplot(all_predictions, labels=model_labels, showfliers=False)
        
        # Labels
        plt.title('Prediction Distribution by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Prediction Value')
        plt.grid(True, alpha=0.3)
        
        pdf.savefig()
        plt.close()
        
        # 5. Factor-specific visualizations (for a few selected factors)
        if any(matrix.shape[1] > 0 for matrix in prediction_matrices.values()):
            # Get a common set of factors across all models
            common_factors = set.intersection(*[set(matrix.columns) for matrix in prediction_matrices.values()])
            
            # If no common factors, take from the first model
            if not common_factors and len(prediction_matrices) > 0:
                first_model = list(prediction_matrices.keys())[0]
                common_factors = set(prediction_matrices[first_model].columns)
            
            # Select a few factors (up to 5)
            if len(common_factors) > 5:
                selected_factors = list(common_factors)[:5]  # Take first 5
            else:
                selected_factors = list(common_factors)
            
            # Create comparison plots for each selected factor
            for factor in selected_factors:
                plt.figure(figsize=(15, 10))
                
                # Plot time series for each model
                for model_type, matrix in prediction_matrices.items():
                    if factor in matrix.columns:
                        plt.plot(matrix.index, matrix[factor], label=model_type)
                
                # Labels
                plt.title(f'Model Comparison for Factor: {factor}')
                plt.xlabel('Date')
                plt.ylabel('Prediction Value')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                pdf.savefig()
                plt.close()

def main():
    """
    Main execution function.
    """
    print("=== Step 12B: Generate Linear Model Predictions ===")
    
    start_time = time.time()
    
    # Input files
    models_file = os.path.join("output", "S11B_Linear_Models.h5")
    feature_sets_file = os.path.join("output", "S8_Feature_Sets.h5")
    window_schedule_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    
    # Output files
    predictions_h5_file = os.path.join("output", "S12B_Linear_Model_Predictions.h5")
    predictions_excel_file = os.path.join("output", "S12B_Linear_Model_Predictions_Matrix.xlsx")
    visualization_file = os.path.join("output", "S12B_Prediction_Visualization.pdf")
    
    # Step 12B.1: Load data
    window_schedule = load_window_schedule(window_schedule_file)
    feature_sets = load_feature_sets(feature_sets_file)
    models = load_trained_models(models_file)
    
    # Step 12B.2: Generate predictions for all windows and models
    predictions = process_all_windows(models, feature_sets, window_schedule)
    
    # Step 12B.3: Save raw predictions to HDF5
    save_predictions_to_h5(predictions, predictions_h5_file)
    
    # Step 12B.4: Create and save prediction matrices
    prediction_matrices = create_prediction_matrix(predictions)
    save_prediction_matrix(prediction_matrices, predictions_excel_file)
    
    # Step 12B.5: Create and save visualizations
    create_visualizations(prediction_matrices, visualization_file)
    
    # Print completion summary
    elapsed_time = time.time() - start_time
    
    print(f"\nStep 12B completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Predictions saved to: {predictions_h5_file}")
    print(f"Prediction matrices saved to: {predictions_excel_file}")
    print(f"Visualizations saved to: {visualization_file}")

if __name__ == "__main__":
    main()