# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S2_T2_Optimizer_with_MA.xlsx
  - Path: ./output/S2_T2_Optimizer_with_MA.xlsx (output from Step 2)
  - Description: Excel file containing factor data with all moving averages.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

- S4_Window_Schedule.xlsx
  - Path: ./output/S4_Window_Schedule.xlsx (output from Step 4)
  - Description: Excel file containing the schedule of all 236 rolling windows.
  - Format: Excel (.xlsx) with window parameters and dates.

- S7_Helper_Features.h5
  - Path: ./output/S7_Helper_Features.h5 (output from Step 7)
  - Description: HDF5 file containing top 10 most correlated helper features for each factor and window.
  - Format: HDF5 with window and factor-specific helper features.

- S2_Column_Mapping.xlsx
  - Path: ./output/S2_Column_Mapping.xlsx (output from Step 2)
  - Description: Excel file containing mappings between original factor names and their MA columns.
  - Format: Excel (.xlsx) with mappings between factors and their MA column names.

OUTPUT FILES:
- S8_Feature_Sets.h5
  - Path: ./output/S8_Feature_Sets.h5
  - Description: HDF5 file containing the 14-dimensional feature sets for each factor and window.
                 Includes own factor MAs (1, 3, 12, 60-month) and helper factors' 60-month MAs.
  - Format: HDF5 with window and factor-specific feature sets.

- S8_Feature_Sets_Sample.xlsx
  - Path: ./output/S8_Feature_Sets_Sample.xlsx
  - Description: Excel file with feature sets for a sample of windows and factors.
                 For easy inspection and verification.
  - Format: Excel (.xlsx) with header and organized by window and factor.

- S8_Feature_Sets_Visualization.pdf
  - Path: ./output/S8_Feature_Sets_Visualization.pdf
  - Description: PDF with visualizations of feature set statistics and distributions.
  - Format: PDF with various charts and analysis.

----------------------------------------------------------------------------------------------------
Purpose:
This script creates the 14-dimensional feature sets for each factor in each window:
- 4 features from the target factor's own MAs (1, 3, 12, 60-month)
- 10 features from the 60-month MAs of the top 10 helper factors

These feature sets will be used to train factor-specific XGBoost models in subsequent steps.

Version: 3.0
Last Updated: Current Date
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
import h5py
import warnings
from datetime import datetime
import random
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import re  # Add this import for regex pattern matching

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Section: Utility Functions
def load_helper_features(helper_file):
    """
    Load helper features from the H5 file.
    
    Parameters:
    -----------
    helper_file : str
        Path to the helper features H5 file.
    
    Returns:
    --------
    dict
        Nested dictionary with window_id -> factor -> list of helper features.
    list
        List of all factor names from metadata.
    """
    print(f"Loading helper features from {helper_file}...")
    
    helper_features = {}
    factor_columns = []
    
    with h5py.File(helper_file, 'r') as hf:
        # Get factor columns from metadata
        if 'metadata' in hf and 'factor_columns' in hf['metadata']:
            factor_columns = [name.decode('utf-8') for name in hf['metadata']['factor_columns'][:]]
            print(f"Loaded {len(factor_columns)} factor names from metadata")
        
        # Get helper features for each window and factor
        if 'helper_features' in hf:
            helpers_group = hf['helper_features']
            
            # Iterate through each window
            for window_name in helpers_group:
                window_id = int(window_name.split('_')[1])  # Extract window ID from "window_X"
                window_group = helpers_group[window_name]
                
                helper_features[window_id] = {}
                
                # Iterate through each factor in this window
                for factor_key in window_group:
                    factor_group = window_group[factor_key]
                    
                    # Get helper names (top 10 most correlated helpers)
                    helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
                    helper_corrs = factor_group['correlations'][:]
                    
                    # Convert HDF5 key back to original factor name if needed
                    original_factor = next((f for f in factor_columns if f.replace('/', '_') == factor_key), factor_key)
                    
                    # Store the helpers with their correlation values
                    helper_features[window_id][original_factor] = list(zip(helper_names, helper_corrs))
    
    return helper_features, factor_columns

def load_column_mapping(mapping_file):
    """
    Load column mapping from Excel file and create lookups for factor to MA columns.
    
    Parameters:
    -----------
    mapping_file : str
        Path to the column mapping Excel file.
    
    Returns:
    --------
    dict
        Dictionary mapping factor names to their MA columns.
    """
    print(f"Loading column mapping from {mapping_file}...")
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Column mapping file not found: {mapping_file}")
    
    mapping_df = pd.read_excel(mapping_file)
    
    # Create a dictionary to map factor names to their MA columns
    factor_to_ma_map = {}
    
    for _, row in mapping_df.iterrows():
        factor_name = row['factor_name']
        original_col = row['original_column']
        
        # Get MA columns
        ma_1m = original_col  # The original column is the 1-month version
        ma_3m = row['column_3m'] if row['column_3m'] != 'N/A' else None
        ma_12m = row['column_12m'] if row['column_12m'] != 'N/A' else None
        ma_60m = row['column_60m'] if row['column_60m'] != 'N/A' else None
        
        # Store mapping for both the factor name and original column name
        factor_to_ma_map[factor_name] = {
            "1m": ma_1m,
            "3m": ma_3m,
            "12m": ma_12m,
            "60m": ma_60m
        }
        
        # Also store the mapping for the original column
        factor_to_ma_map[original_col] = {
            "1m": ma_1m,
            "3m": ma_3m,
            "12m": ma_12m,
            "60m": ma_60m
        }
    
    print(f"Loaded mapping for {len(factor_to_ma_map)} factors")
    return factor_to_ma_map

def get_windows_data(factor_data, window_schedule):
    """
    Precompute data subsets for each window to avoid repeated filtering.
    
    Parameters:
    -----------
    factor_data : pandas.DataFrame
        DataFrame containing the factor data.
    window_schedule : pandas.DataFrame
        DataFrame containing the window schedule.
        
    Returns:
    --------
    dict
        Dictionary mapping window_id to dictionary with 'training', 'validation', and 'prediction' data.
    """
    print("Precomputing data subsets for each window...")
    
    window_data = {}
    
    for _, window in tqdm(window_schedule.iterrows(), total=len(window_schedule)):
        window_id = window['Window_ID']
        
        training_start = window['Training_Start_Date']
        training_end = window['Training_End_Date']
        validation_end = window['Validation_End_Date']
        prediction_date = window['Prediction_Date']
        
        # Get data for each period
        training_data = factor_data[(factor_data['Date'] >= training_start) & 
                                    (factor_data['Date'] <= training_end)]
        
        validation_data = factor_data[(factor_data['Date'] > training_end) & 
                                      (factor_data['Date'] <= validation_end)]
        
        prediction_data = factor_data[factor_data['Date'] == prediction_date]
        
        window_data[window_id] = {
            'training': training_data,
            'validation': validation_data,
            'prediction': prediction_data,
            'training_end': training_end,
            'validation_end': validation_end,
            'prediction_date': prediction_date
        }
    
    return window_data

def precompute_next_month_returns(factor_data, date_column='Date'):
    """
    Precompute next month returns for all factors to avoid repeated calculations.
    
    Parameters:
    -----------
    factor_data : pandas.DataFrame
        DataFrame containing the factor data.
    date_column : str
        Name of the date column.
        
    Returns:
    --------
    dict
        Dictionary mapping factor names to DataFrames with next month returns.
    """
    print("Precomputing next month returns for all factors...")
    
    next_month_returns = {}
    
    # Sort data by date
    factor_data_sorted = factor_data.sort_values(by=date_column)
    
    # Calculate next month return for all factors (except date)
    for column in tqdm(factor_data.columns):
        if column != date_column:
            # Create a copy of the date column and the factor column
            next_month_df = pd.DataFrame({
                date_column: factor_data_sorted[date_column],
                'next_return': factor_data_sorted[column].shift(-1)  # Next month's value
            })
            
            next_month_returns[column] = next_month_df
    
    return next_month_returns

def extract_base_factor_name(factor):
    """
    Extract the base factor name by removing MA suffixes.
    
    Parameters:
    -----------
    factor : str
        Factor name with possible suffixes.
        
    Returns:
    --------
    str
        Base factor name.
    """
    # Common MA suffix patterns
    patterns = [
        r'_[CT]S_[0-9]+m$',  # Matches _CS_3m, _TS_60m, etc.
        r'_[0-9]+m$',        # Matches _3m, _12m, etc.
        r'_[CT]S$'           # Matches _CS, _TS suffixes
    ]
    
    for pattern in patterns:
        if re.search(pattern, factor):
            return re.sub(pattern, '', factor)
    
    return factor

def create_feature_sets_for_window(window_id, window_data, helper_features, column_mapping, 
                                  next_month_returns):
    """
    Create feature sets for a specific window.
    
    Parameters:
    -----------
    window_id : int
        Window ID.
    window_data : dict
        Dictionary with 'training', 'validation', and 'prediction' data.
    helper_features : dict
        Dictionary with helper features for each factor in this window.
    column_mapping : dict
        Dictionary mapping factor names to their MA columns.
    next_month_returns : dict
        Dictionary mapping factor names to DataFrames with next month returns.
        
    Returns:
    --------
    dict
        Dictionary with feature sets for this window.
    """
    print(f"Creating feature sets for window {window_id}...")
    
    # Get data for this window
    training_data = window_data['training']
    validation_data = window_data['validation']
    prediction_data = window_data['prediction'] 
    
    # Dictionary to store feature sets for this window
    window_feature_sets = {
        'training': {},
        'validation': {},
        'prediction': {}
    }
    
    # Build a mapping of columns with _60m suffix for fast lookup
    columns_60m = {}
    for col in training_data.columns:
        if col.endswith('_60m'):
            base_name = col[:-4]  # Remove _60m suffix
            columns_60m[base_name] = col
    
    # Process each factor that has helper features for this window
    window_helper_features = helper_features.get(window_id, {})
    
    for target_factor in tqdm(window_helper_features.keys(), desc=f"Window {window_id}"):
        # Extract base factor name if it has MA suffixes
        base_factor = extract_base_factor_name(target_factor)
        
        # Check if this target factor has MA column mapping
        ma_columns = None
        if target_factor in column_mapping:
            ma_columns = column_mapping[target_factor]
        elif base_factor in column_mapping:
            ma_columns = column_mapping[base_factor]
        else:
            print(f"Warning: No column mapping found for target {target_factor} (base: {base_factor}). Skipping.")
            continue
        
        # Verify all required MA columns exist
        missing_columns = False
        for period, column in ma_columns.items():
            if column is None or column not in training_data.columns:
                print(f"Warning: {period} MA column {column} for {base_factor or target_factor} not found. Skipping.")
                missing_columns = True
                break
        
        if missing_columns:
            continue
        
        # Get helper features for this target factor
        helpers = window_helper_features[target_factor]
        helper_names = [h[0] for h in helpers[:10]]  # Take top 10 helpers
        
        # Check if we have enough helpers
        if len(helper_names) < 10:
            print(f"Warning: Not enough helpers for {target_factor}. Found {len(helper_names)}, need 10. Skipping.")
            continue
        
        try:
            # Create target factor's own MA features (4 dimensions)
            X_train_own = pd.DataFrame({
                f"{target_factor}_1m": training_data[ma_columns["1m"]],
                f"{target_factor}_3m": training_data[ma_columns["3m"]],
                f"{target_factor}_12m": training_data[ma_columns["12m"]],
                f"{target_factor}_60m": training_data[ma_columns["60m"]]
            })
            
            X_val_own = pd.DataFrame({
                f"{target_factor}_1m": validation_data[ma_columns["1m"]],
                f"{target_factor}_3m": validation_data[ma_columns["3m"]],
                f"{target_factor}_12m": validation_data[ma_columns["12m"]],
                f"{target_factor}_60m": validation_data[ma_columns["60m"]]
            })
            
            X_pred_own = pd.DataFrame({
                f"{target_factor}_1m": prediction_data[ma_columns["1m"]],
                f"{target_factor}_3m": prediction_data[ma_columns["3m"]],
                f"{target_factor}_12m": prediction_data[ma_columns["12m"]],
                f"{target_factor}_60m": prediction_data[ma_columns["60m"]]
            })
            
            # Create helper features (10 dimensions from 60m versions of helper factors)
            X_train_helpers = pd.DataFrame()
            X_val_helpers = pd.DataFrame()
            X_pred_helpers = pd.DataFrame()
            
            valid_helpers = 0
            for helper_idx, helper_name in enumerate(helper_names):
                # Extract base helper name if needed
                base_helper = extract_base_factor_name(helper_name)
                
                # Find the 60m version of this helper
                helper_60m_col = None
                
                # Try different ways to find the helper's 60m column
                if helper_name.endswith('_60m') and helper_name in training_data.columns:
                    # Direct match if it already ends with _60m
                    helper_60m_col = helper_name
                elif helper_name in column_mapping and column_mapping[helper_name]['60m'] is not None:
                    # Use column mapping if available
                    helper_60m_col = column_mapping[helper_name]['60m']
                elif base_helper in column_mapping and column_mapping[base_helper]['60m'] is not None:
                    # Use base name column mapping if available
                    helper_60m_col = column_mapping[base_helper]['60m']
                elif base_helper in columns_60m:
                    # Use our 60m columns dictionary
                    helper_60m_col = columns_60m[base_helper]
                
                # Skip if we can't find the 60m column
                if helper_60m_col is None or helper_60m_col not in training_data.columns:
                    continue
                
                # Add to feature sets
                col_name = f"helper_{valid_helpers+1}_{helper_name}"
                X_train_helpers[col_name] = training_data[helper_60m_col]
                X_val_helpers[col_name] = validation_data[helper_60m_col]
                X_pred_helpers[col_name] = prediction_data[helper_60m_col]
                valid_helpers += 1
                
                # If we have 10 valid helpers, we can stop
                if valid_helpers >= 10:
                    break
            
            # Check if we got enough helpers
            if valid_helpers < 10:
                print(f"Warning: Only found {valid_helpers} valid helpers with 60m columns for {target_factor}. Skipping.")
                continue
            
            # Create target return series (next month's value)
            target_column = ma_columns["1m"]  # Use the 1m column for target returns
            if target_column not in next_month_returns:
                print(f"Warning: No next month returns for {target_factor} ({target_column}). Skipping.")
                continue
            
            factor_returns = next_month_returns[target_column]
            
            # Join returns with training and validation data
            training_with_returns = pd.merge(
                training_data[['Date']], 
                factor_returns, 
                left_on='Date', 
                right_on='Date',
                how='left'
            )
            
            validation_with_returns = pd.merge(
                validation_data[['Date']], 
                factor_returns, 
                left_on='Date', 
                right_on='Date',
                how='left'
            )
            
            # Extract target values
            y_train = training_with_returns['next_return']
            y_val = validation_with_returns['next_return']
            
            # Combine own features and helper features
            X_train = pd.concat([X_train_own, X_train_helpers], axis=1)
            X_val = pd.concat([X_val_own, X_val_helpers], axis=1)
            X_pred = pd.concat([X_pred_own, X_pred_helpers], axis=1)
            
            # Remove rows with missing values
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
            pred_mask = ~X_pred.isna().any(axis=1)
            
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            
            X_val_clean = X_val[val_mask]
            y_val_clean = y_val[val_mask]
            
            X_pred_clean = X_pred[pred_mask]
            
            # Store feature sets - use the original target_factor as the key 
            # (not the base factor name) to preserve the original names
            window_feature_sets['training'][target_factor] = {
                'X': X_train_clean,
                'y': y_train_clean,
                'X_columns': X_train_clean.columns.tolist()
            }
            
            window_feature_sets['validation'][target_factor] = {
                'X': X_val_clean,
                'y': y_val_clean,
                'X_columns': X_val_clean.columns.tolist()
            }
            
            window_feature_sets['prediction'][target_factor] = {
                'X': X_pred_clean,
                'X_columns': X_pred_clean.columns.tolist()
            }
        
        except Exception as e:
            print(f"Error creating feature set for {target_factor}: {e}")
            continue
    
    return window_id, window_feature_sets

def save_feature_sets_to_h5(feature_sets, factor_columns, output_file):
    """
    Save feature sets to an HDF5 file.
    
    Parameters:
    -----------
    feature_sets : dict
        Dictionary with feature sets for each window.
    factor_columns : list
        List of factor column names.
    output_file : str
        Path to the output HDF5 file.
    """
    print(f"Saving feature sets to {output_file}...")
    
    with h5py.File(output_file, 'w') as hf:
        # Save metadata
        meta_group = hf.create_group('metadata')
        
        # Save creation timestamp
        meta_group.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta_group.attrs['total_windows'] = len(feature_sets)
        
        # Save factor columns
        factor_columns_bytes = [col.encode('utf-8') for col in factor_columns]
        meta_group.create_dataset('factor_columns', data=factor_columns_bytes)
        
        # Create group for feature sets
        feature_sets_group = hf.create_group('feature_sets')
        
        # Save feature sets for each window
        for window_id, window_sets in feature_sets.items():
            window_group = feature_sets_group.create_group(f'window_{window_id}')
            
            # Save each split (training, validation, prediction)
            for split_name, split_data in window_sets.items():
                split_group = window_group.create_group(split_name)
                
                # Save data for each factor
                for factor, factor_data in split_data.items():
                    # Use a safe version of the factor name (replace / with _)
                    safe_factor = factor.replace('/', '_')
                    factor_group = split_group.create_group(safe_factor)
                    
                    # Save features (X)
                    if 'X' in factor_data and not factor_data['X'].empty:
                        X_group = factor_group.create_group('X')
                        
                        # Save X data
                        X_group.create_dataset('data', data=factor_data['X'].values)
                        
                        # Save column names
                        X_columns_bytes = [col.encode('utf-8') for col in factor_data['X_columns']]
                        X_group.create_dataset('columns', data=X_columns_bytes)
                    
                    # Save target (y) if available
                    if 'y' in factor_data and not factor_data['y'].empty:
                        factor_group.create_dataset('y', data=factor_data['y'].values)

def save_sample_to_excel(feature_sets, output_file, num_windows=3, num_factors=5):
    """
    Save a sample of the feature sets to an Excel file for easy inspection.
    
    Parameters:
    -----------
    feature_sets : dict
        Dictionary with feature sets for each window.
    output_file : str
        Path to the output Excel file.
    num_windows : int
        Number of windows to include in the sample.
    num_factors : int
        Number of factors per window to include in the sample.
    """
    print(f"Saving sample feature sets to {output_file}...")
    
    # Create a new Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Get a sample of windows
        sample_windows = list(feature_sets.keys())
        if len(sample_windows) > num_windows:
            sample_windows = sorted(sample_windows)
            step = max(1, len(sample_windows) // num_windows)
            sample_windows = sample_windows[::step][:num_windows]
        
        # For each window in the sample
        for window_id in sample_windows:
            window_data = feature_sets[window_id]
            
            # For each split (training, validation, prediction)
            for split_name, split_data in window_data.items():
                # Get a sample of factors
                sample_factors = list(split_data.keys())
                if len(sample_factors) > num_factors:
                    random_factors = random.sample(sample_factors, num_factors)
                else:
                    random_factors = sample_factors
                
                # For each factor in the sample
                for factor in random_factors:
                    factor_data = split_data[factor]
                    
                    # Create a sheet name
                    sheet_name = f"W{window_id}_{split_name[:3]}_{factor[:10]}"
                    sheet_name = sheet_name.replace('/', '_')[:31]  # Excel sheet name limit
                    
                    # Create a combined DataFrame with X and y
                    if 'X' in factor_data and not factor_data['X'].empty:
                        combined_df = factor_data['X'].copy()
                        
                        # Add target column if it exists and has values
                        if 'y' in factor_data and isinstance(factor_data['y'], (pd.Series, np.ndarray)) and len(factor_data['y']) > 0:
                            # Make sure we have the right length
                            if len(factor_data['y']) == len(combined_df):
                                combined_df['target'] = factor_data['y'].values
                            else:
                                # If lengths don't match, handle it
                                print(f"Warning: Length mismatch for {sheet_name} - X: {len(combined_df)}, y: {len(factor_data['y'])}")
                                # Add target column but with NaN values
                                combined_df['target'] = np.nan
                        else:
                            # Add empty target column for consistency
                            combined_df['target'] = np.nan
                        
                        # Save to Excel
                        combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

def create_feature_sets_visualization(feature_sets, output_file):
    """
    Create visualizations of the feature sets and save to a PDF file.
    
    Parameters:
    -----------
    feature_sets : dict
        Dictionary with feature sets for each window.
    output_file : str
        Path to the output PDF file.
    """
    print(f"Creating feature sets visualization and saving to {output_file}...")
    
    with PdfPages(output_file) as pdf:
        # Create a title page
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Feature Sets Visualization', 
                fontsize=24, ha='center', va='center')
        plt.text(0.5, 0.4, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=16, ha='center', va='center')
        plt.text(0.5, 0.3, f'Total Windows: {len(feature_sets)}', 
                fontsize=16, ha='center', va='center')
        pdf.savefig()
        plt.close()
        
        # Calculate statistics across all windows
        factors_per_window = []
        features_per_factor = []
        samples_per_factor = []
        corr_values = []
        
        for window_id, window_data in feature_sets.items():
            for split_name, split_data in window_data.items():
                if split_name == 'training':
                    factors_per_window.append(len(split_data))
                    
                    for factor, factor_data in split_data.items():
                        if 'X' in factor_data and not factor_data['X'].empty:
                            features_per_factor.append(factor_data['X'].shape[1])
                            samples_per_factor.append(factor_data['X'].shape[0])
                            
                            # Calculate correlations
                            if 'X' in factor_data and 'y' in factor_data:
                                X = factor_data['X']
                                y = factor_data['y']
                                
                                if not X.empty and not y.empty:
                                    for col in X.columns:
                                        corr = X[col].corr(y)
                                        if not np.isnan(corr):
                                            corr_values.append(corr)
        
        # Create summary statistics plots
        plt.figure(figsize=(10, 6))
        plt.hist(factors_per_window, bins=20, alpha=0.7)
        plt.title('Number of Factors per Window')
        plt.xlabel('Factors')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(features_per_factor, bins=10, alpha=0.7)
        plt.title('Number of Features per Factor')
        plt.xlabel('Features')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(samples_per_factor, bins=20, alpha=0.7)
        plt.title('Number of Samples per Factor')
        plt.xlabel('Samples')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(corr_values, bins=20, alpha=0.7)
        plt.title('Feature-Target Correlation Distribution')
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Create sample feature importance plots (for a few windows/factors)
        sample_windows = list(feature_sets.keys())
        if len(sample_windows) > 3:
            sample_windows = sorted(sample_windows)
            step = max(1, len(sample_windows) // 3)
            sample_windows = sample_windows[::step][:3]
        
        for window_id in sample_windows:
            window_data = feature_sets[window_id]['training']
            
            sample_factors = list(window_data.keys())
            if len(sample_factors) > 3:
                sample_factors = random.sample(sample_factors, 3)
            
            for factor in sample_factors:
                factor_data = window_data[factor]
                
                if 'X' in factor_data and 'y' in factor_data:
                    X = factor_data['X']
                    y = factor_data['y']
                    
                    if not X.empty and not y.empty:
                        # Calculate absolute correlation for each feature
                        correlations = []
                        for col in X.columns:
                            corr = X[col].corr(y)
                            if not np.isnan(corr):
                                correlations.append((col, abs(corr)))
                        
                        if correlations:
                            # Sort by correlation
                            correlations.sort(key=lambda x: x[1], reverse=True)
                            
                            # Plot
                            plt.figure(figsize=(12, 6))
                            
                            # Get data for the plot
                            features = [c[0][:15] + '...' if len(c[0]) > 15 else c[0] for c in correlations]
                            corrs = [c[1] for c in correlations]
                            
                            # Create horizontal bar plot
                            bars = plt.barh(features, corrs, alpha=0.7)
                            plt.title(f'Feature Importance (Absolute Correlation) - Window {window_id}, Factor {factor}')
                            plt.xlabel('Absolute Correlation with Target')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            pdf.savefig()
                            plt.close()

def process_window_batch(window_ids, window_data, helper_features, column_mapping, next_month_returns):
    """
    Process a batch of windows in parallel.
    
    Parameters:
    -----------
    window_ids : list
        List of window IDs to process.
    window_data : dict
        Dictionary with data for each window.
    helper_features : dict
        Dictionary with helper features for each window and factor.
    column_mapping : dict
        Dictionary mapping factor names to their MA columns.
    next_month_returns : dict
        Dictionary mapping factor names to DataFrames with next month returns.
        
    Returns:
    --------
    dict
        Dictionary with feature sets for the processed windows.
    """
    results = {}
    
    for window_id in window_ids:
        if window_id in window_data:
            window_id, window_feature_sets = create_feature_sets_for_window(
                window_id, 
                window_data[window_id], 
                helper_features, 
                column_mapping, 
                next_month_returns
            )
            results[window_id] = window_feature_sets
    
    return results

def main():
    print("\n" + "="*80)
    print("STEP 8: CREATE FEATURE SETS")
    print("="*80)
    
    start_time = time.time()
    
    # Input files
    factor_data_file = os.path.join("output", "S2_T2_Optimizer_with_MA.xlsx")
    window_schedule_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    helper_features_file = os.path.join("output", "S7_Helper_Features.h5")
    column_mapping_file = os.path.join("output", "S2_Column_Mapping.xlsx")
    
    # Output files
    feature_sets_output = os.path.join("output", "S8_Feature_Sets.h5")
    sample_output = os.path.join("output", "S8_Feature_Sets_Sample.xlsx")
    visualization_output = os.path.join("output", "S8_Feature_Sets_Visualization.pdf")
    
    # Check if input files exist
    for file_path in [factor_data_file, window_schedule_file, helper_features_file, column_mapping_file]:
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} not found.")
            return
    
    try:
        # Step 1: Load the data
        print("\nStep 1: Loading input data...")
        factor_data = pd.read_excel(factor_data_file)
        window_schedule = pd.read_excel(window_schedule_file)
        helper_features, factor_columns = load_helper_features(helper_features_file)
        column_mapping = load_column_mapping(column_mapping_file)
        
        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing data...")
        window_data = get_windows_data(factor_data, window_schedule)
        next_month_returns = precompute_next_month_returns(factor_data)
        
        # Step 3: Create feature sets for each window
        print("\nStep 3: Creating feature sets...")
        
        # Determine number of processes based on CPU cores
        num_processes = min(mp.cpu_count(), len(window_data))
        print(f"Using {num_processes} processes for parallel processing")
        
        # Split windows into batches for parallel processing
        window_ids = list(window_data.keys())
        batch_size = max(1, len(window_ids) // num_processes)
        window_batches = [window_ids[i:i+batch_size] for i in range(0, len(window_ids), batch_size)]
        
        # Create feature sets in parallel
        feature_sets = {}
        
        if num_processes > 1:
            process_func = partial(
                process_window_batch,
                window_data=window_data,
                helper_features=helper_features,
                column_mapping=column_mapping,
                next_month_returns=next_month_returns
            )
            
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(process_func, window_batches)
                
                # Combine results
                for result in results:
                    feature_sets.update(result)
        else:
            # Serial processing
            for window_id in tqdm(window_ids):
                window_id, window_feature_sets = create_feature_sets_for_window(
                    window_id,
                    window_data[window_id],
                    helper_features,
                    column_mapping,
                    next_month_returns
                )
                feature_sets[window_id] = window_feature_sets
        
        # Step 4: Save the results
        print("\nStep 4: Saving results...")
        save_feature_sets_to_h5(feature_sets, factor_columns, feature_sets_output)
        save_sample_to_excel(feature_sets, sample_output)
        create_feature_sets_visualization(feature_sets, visualization_output)
        
        # Print summary
        total_factors = sum(len(window_data['training']) for window_data in feature_sets.values())
        print("\nSummary:")
        print(f"Total Windows: {len(feature_sets)}")
        print(f"Total Factors: {total_factors}")
        
        elapsed_time = time.time() - start_time
        print(f"\nStep 8 completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print("="*80)
        
    except Exception as e:
        print(f"Error in Step 8: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 