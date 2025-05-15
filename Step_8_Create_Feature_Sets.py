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

Version: 1.0
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
from tqdm import tqdm
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Section: Utility Functions
def load_helper_features_from_h5(input_file):
    """
    Load helper features from HDF5 file.
    
    Parameters:
    -----------
    input_file : str
        Path to input HDF5 file with helper features.
    
    Returns:
    --------
    dict
        Nested dictionary with window IDs and factors as keys and list of helper features as values.
    list
        List of factor column names.
    """
    helper_features = {}
    factor_columns = None
    
    with h5py.File(input_file, 'r') as hf:
        # Get factor columns from metadata
        factor_columns = [name.decode('utf-8') for name in hf['metadata']['factor_columns'][:]]
        
        # Get helper features for each window and factor
        helpers_group = hf['helper_features']
        
        for window_name in helpers_group:
            window_id = int(window_name.split('_')[1])  # Extract window ID
            window_group = helpers_group[window_name]
            
            helper_features[window_id] = {}
            
            for factor_name in window_group:
                factor_group = window_group[factor_name]
                
                # Get helper names and correlations
                helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
                helper_corrs = factor_group['correlations'][:]
                
                # Map HDF5 group key back to the original factor name using metadata
                original_factor = next((f for f in factor_columns if f.replace('/', '_') == factor_name), factor_name)
                helper_features[window_id][original_factor] = list(zip(helper_names, helper_corrs))
    
    return helper_features, factor_columns

def create_feature_sets(factor_data, helper_features, window_schedule, factor_columns):
    """
    Create 14-dimensional feature sets for each factor and window.
    
    Parameters:
    -----------
    factor_data : pandas.DataFrame
        DataFrame with all factor data including moving averages.
    helper_features : dict
        Nested dictionary with window IDs and factors as keys and list of helper features as values.
    window_schedule : pandas.DataFrame
        DataFrame with window schedule information.
    factor_columns : list
        List of factor column names.
    
    Returns:
    --------
    dict
        Nested dictionary with window IDs, factors, and data splits as keys and feature sets as values.
    """
    feature_sets = {}
    
    # Get all available columns for reference
    available_columns = set(factor_data.columns)
    
    # Loop through each window
    for _, window_row in tqdm(window_schedule.iterrows(), total=len(window_schedule), desc="Processing windows"):
        window_id = window_row['Window_ID']
        
        # Get date ranges for this window
        training_start = window_row['Training_Start_Date']
        training_end = window_row['Training_End_Date']
        validation_end = window_row['Validation_End_Date']
        prediction_date = window_row['Prediction_Date']
        
        # Get data for this window
        training_data = factor_data[(factor_data['Date'] >= training_start) & (factor_data['Date'] <= training_end)]
        validation_data = factor_data[(factor_data['Date'] > training_end) & (factor_data['Date'] <= validation_end)]
        prediction_data = factor_data[factor_data['Date'] == prediction_date]
        
        # Skip if any dataset is empty
        if training_data.empty or validation_data.empty or prediction_data.empty:
            print(f"Warning: Missing data for window {window_id}, skipping...")
            continue
        
        # Initialize feature sets for this window
        feature_sets[window_id] = {'training': {}, 'validation': {}, 'prediction': {}}
        
        # Helper function to find moving average columns for a factor
        def find_ma_columns(factor, base_data):
            """
            Find the moving average columns for a factor in the data.
            
            Parameters:
            -----------
            factor : str
                The name of the factor to find MA columns for.
            base_data : pandas.DataFrame
                The data containing the MA columns.
            
            Returns:
            --------
            dict
                Dictionary with periods as keys and column names as values.
            """
            # First check if there's a column mapping file from Step 2
            column_mapping_file = os.path.join("output", "S2_Column_Mapping.xlsx")
            if os.path.exists(column_mapping_file):
                try:
                    mapping_df = pd.read_excel(column_mapping_file)
                    # Check if this factor is in the mapping
                    factor_row = mapping_df[mapping_df['original_column'] == factor]
                    if not factor_row.empty:
                        # Get columns from the mapping
                        col_3m = factor_row['column_3m'].iloc[0]
                        col_12m = factor_row['column_12m'].iloc[0]
                        col_60m = factor_row['column_60m'].iloc[0]
                        
                        # Verify columns exist in data
                        col_3m = col_3m if col_3m != "N/A" and col_3m in base_data.columns else factor
                        col_12m = col_12m if col_12m != "N/A" and col_12m in base_data.columns else factor
                        col_60m = col_60m if col_60m != "N/A" and col_60m in base_data.columns else factor
                        
                        return {
                            "1m": factor,
                            "3m": col_3m,
                            "12m": col_12m,
                            "60m": col_60m
                        }
                    
                    # Also try with the factor name (might be a base name)
                    factor_row = mapping_df[mapping_df['factor_name'] == factor]
                    if not factor_row.empty:
                        # Get the original column and its MA columns
                        original_col = factor_row['original_column'].iloc[0]
                        col_3m = factor_row['column_3m'].iloc[0]
                        col_12m = factor_row['column_12m'].iloc[0]
                        col_60m = factor_row['column_60m'].iloc[0]
                        
                        # Verify columns exist in data
                        original_col = original_col if original_col in base_data.columns else factor
                        col_3m = col_3m if col_3m != "N/A" and col_3m in base_data.columns else original_col
                        col_12m = col_12m if col_12m != "N/A" and col_12m in base_data.columns else original_col
                        col_60m = col_60m if col_60m != "N/A" and col_60m in base_data.columns else original_col
                        
                        return {
                            "1m": original_col,
                            "3m": col_3m,
                            "12m": col_12m,
                            "60m": col_60m
                        }
                except Exception as e:
                    print(f"Warning: Error reading column mapping file: {e}")
            
            # Try different patterns for moving average columns
            patterns = [
                # Pattern 1: factor_3m (standard pattern, e.g., "Gold_TS_3m")
                {
                    "1m": factor,
                    "3m": f"{factor}_3m",
                    "12m": f"{factor}_12m",
                    "60m": f"{factor}_60m"
                },
                # Pattern 2: factor_TS_3m (where _TS is appended, e.g., "Gold_TS_3m")
                {
                    "1m": factor,
                    "3m": f"{factor}_TS_3m",
                    "12m": f"{factor}_TS_12m",
                    "60m": f"{factor}_TS_60m"
                },
                # Pattern 3: factor_CS_3m (where _CS is appended, e.g., "Gold_CS_3m")
                {
                    "1m": factor,
                    "3m": f"{factor}_CS_3m",
                    "12m": f"{factor}_CS_12m",
                    "60m": f"{factor}_CS_60m"
                }
            ]
            
            # If factor already has a period suffix (e.g., Gold_TS_3m)
            period_match = re.search(r'_(\d+)m$', factor)
            if period_match:
                # Add pattern with additional time period (e.g., Gold_TS_3m_12m)
                period_patterns = {
                    "1m": factor,
                    "3m": f"{factor}_3m",
                    "12m": f"{factor}_12m",
                    "60m": f"{factor}_60m"
                }
                patterns.append(period_patterns)
                
                # Also try replacing period with other periods
                base_name = re.sub(r'_\d+m$', '', factor)
                base_patterns = {
                    "1m": factor,  # Use the factor with period as 1m
                    "3m": f"{base_name}_3m",
                    "12m": f"{base_name}_12m",
                    "60m": f"{base_name}_60m"
                }
                patterns.append(base_patterns)
                
                # If base_name has CS or TS designation
                if "_CS" in base_name:
                    cs_patterns = {
                        "1m": factor,
                        "3m": f"{base_name}_3m",
                        "12m": f"{base_name}_12m",
                        "60m": f"{base_name}_60m"
                    }
                    patterns.append(cs_patterns)
                elif "_TS" in base_name:
                    ts_patterns = {
                        "1m": factor,
                        "3m": f"{base_name}_3m",
                        "12m": f"{base_name}_12m",
                        "60m": f"{base_name}_60m"
                    }
                    patterns.append(ts_patterns)
            
            # Check each pattern
            for pattern in patterns:
                if all(col in base_data.columns for col in pattern.values()):
                    return pattern
            
            # If no pattern fully matches, use whatever is available
            result = {
                "1m": factor,  # Base factor always available (we checked this earlier)
            }
            
            # Try to find MA columns in any format
            for period in ["3m", "12m", "60m"]:
                # Build extensive list of candidates
                ma_col_candidates = [
                    f"{factor}_{period}",      # Standard: Gold_TS_3m
                    f"{factor}_TS_{period}",   # With TS: Gold_TS_TS_3m
                    f"{factor}_CS_{period}"    # With CS: Gold_TS_CS_3m
                ]
                
                # If factor has CS or TS suffix, also try base name
                if "_CS" in factor:
                    base_name = factor.split("_CS")[0]
                    ma_col_candidates.extend([
                        f"{base_name}_CS_{period}",
                        f"{base_name}_{period}"
                    ])
                elif "_TS" in factor:
                    base_name = factor.split("_TS")[0]
                    ma_col_candidates.extend([
                        f"{base_name}_TS_{period}",
                        f"{base_name}_{period}"
                    ])
                
                # If factor already has a period suffix (e.g., Gold_TS_3m)
                if period_match:
                    # Try adding new period after existing period
                    ma_col_candidates.append(f"{factor}_{period}")
                    
                    # Also try replacing period with new period
                    base_name = re.sub(r'_\d+m$', '', factor)
                    ma_col_candidates.extend([
                        f"{base_name}_{period}",
                        f"{base_name}_TS_{period}",
                        f"{base_name}_CS_{period}"
                    ])
                
                # Use the first available column
                found = False
                for col in ma_col_candidates:
                    if col in base_data.columns:
                        result[period] = col
                        found = True
                        break
                
                if not found:
                    # If no MA column found, use the base factor as fallback
                    result[period] = factor
                    print(f"Warning: No {period} MA column found for {factor}, using base factor")
            
            return result
                
        # Process each factor
        valid_factors_count = 0
        
        for factor in factor_columns:
            # Skip if this factor doesn't have helper features for this window
            if window_id not in helper_features or factor not in helper_features[window_id]:
                continue
            
            # Skip if base factor is not in the data
            if factor not in training_data.columns:
                print(f"Warning: Factor {factor} not found in training data, skipping")
                continue
                
            # Get helper features for this factor in this window
            factor_helpers = helper_features[window_id][factor]
            helper_names = [h[0] for h in factor_helpers]
            
            # Find moving average columns for this factor
            factor_ma_cols = find_ma_columns(factor, training_data)
            
            # --- Create training feature set ---
            # 1. Target factor's own MAs (1, 3, 12, 60 months)
            X_train_own = pd.DataFrame({
                f"{factor}_1m": training_data[factor_ma_cols["1m"]],
                f"{factor}_3m": training_data[factor_ma_cols["3m"]],
                f"{factor}_12m": training_data[factor_ma_cols["12m"]],
                f"{factor}_60m": training_data[factor_ma_cols["60m"]]
            })
            
            # 2. Helper factors' 60-month MAs
            X_train_helpers = pd.DataFrame()
            for i, helper in enumerate(helper_names):
                # Find 60-month MA for this helper
                helper_col = find_helper_ma_column(helper, training_data)
                
                if helper_col is not None:
                    X_train_helpers[f"helper_{i+1}_{helper}"] = training_data[helper_col]
                else:
                    # Skip this helper if not found
                    print(f"Warning: Helper {helper} not found in training data")
                    continue
            
            # Skip if no helpers were added
            if X_train_helpers.empty:
                print(f"Warning: No valid helpers found for {factor}, skipping")
                continue
                
            # Combine own features and helper features
            X_train = pd.concat([X_train_own, X_train_helpers], axis=1)
            
            # Target (next month return)
            y_train = training_data[factor].shift(-1)
            
            # Remove last row (no target available)
            X_train = X_train.iloc[:-1]
            y_train = y_train.iloc[:-1]
            
            # Skip if empty after dropping last row
            if X_train.empty or y_train.empty:
                print(f"Warning: Empty training data for {factor}, skipping")
                continue
            
            # --- Create validation feature set (same approach) ---
            # Find MA columns in validation data (might be different than training)
            factor_ma_cols_val = find_ma_columns(factor, validation_data)
            
            X_val_own = pd.DataFrame({
                f"{factor}_1m": validation_data[factor_ma_cols_val["1m"]],
                f"{factor}_3m": validation_data[factor_ma_cols_val["3m"]],
                f"{factor}_12m": validation_data[factor_ma_cols_val["12m"]],
                f"{factor}_60m": validation_data[factor_ma_cols_val["60m"]]
            })
            
            X_val_helpers = pd.DataFrame()
            
            for i, helper in enumerate(helper_names):
                # Find 60-month MA for this helper
                helper_col = find_helper_ma_column(helper, validation_data)
                
                # Use same column name as in training for consistency
                train_col_name = next((col for col in X_train_helpers.columns if helper in col), None)
                
                if helper_col is not None and train_col_name is not None:
                    X_val_helpers[train_col_name] = validation_data[helper_col]
                elif train_col_name is not None:
                    # If helper not found but we have the column in training, fill with zeros
                    X_val_helpers[train_col_name] = 0
            
            # Ensure validation has same columns as training
            for col in X_train_helpers.columns:
                if col not in X_val_helpers:
                    X_val_helpers[col] = 0  # Fill missing columns with zeros
            
            X_val = pd.concat([X_val_own, X_val_helpers], axis=1)
            y_val = validation_data[factor].shift(-1)
            
            # Remove last row (no target available)
            X_val = X_val.iloc[:-1]
            y_val = y_val.iloc[:-1]
            
            # Skip if empty after dropping last row
            if X_val.empty or y_val.empty:
                print(f"Warning: Empty validation data for {factor}, skipping")
                continue
            
            # --- Create prediction feature set (just one row) ---
            # Find MA columns in prediction data
            factor_ma_cols_pred = find_ma_columns(factor, prediction_data)
            
            X_pred_own = pd.DataFrame({
                f"{factor}_1m": prediction_data[factor_ma_cols_pred["1m"]],
                f"{factor}_3m": prediction_data[factor_ma_cols_pred["3m"]],
                f"{factor}_12m": prediction_data[factor_ma_cols_pred["12m"]],
                f"{factor}_60m": prediction_data[factor_ma_cols_pred["60m"]]
            })
            
            X_pred_helpers = pd.DataFrame()
            
            for i, helper in enumerate(helper_names):
                # Find 60-month MA for this helper
                helper_col = find_helper_ma_column(helper, prediction_data)
                
                # Use same column name as in training for consistency
                train_col_name = next((col for col in X_train_helpers.columns if helper in col), None)
                
                if helper_col is not None and train_col_name is not None:
                    X_pred_helpers[train_col_name] = prediction_data[helper_col]
                elif train_col_name is not None:
                    # If helper not found but we have the column in training, fill with zeros
                    X_pred_helpers[train_col_name] = 0
            
            # Ensure prediction has same columns as training
            for col in X_train_helpers.columns:
                if col not in X_pred_helpers:
                    X_pred_helpers[col] = 0  # Fill missing columns with zeros
            
            X_pred = pd.concat([X_pred_own, X_pred_helpers], axis=1)
            
            # Store feature sets
            feature_sets[window_id]['training'][factor] = {'X': X_train, 'y': y_train}
            feature_sets[window_id]['validation'][factor] = {'X': X_val, 'y': y_val}
            feature_sets[window_id]['prediction'][factor] = {'X': X_pred}
            
            valid_factors_count += 1
        
        print(f"Window {window_id}: Processed {valid_factors_count} factors successfully")
    
    return feature_sets

def save_feature_sets_to_h5(feature_sets, factor_columns, output_file):
    """
    Save feature sets to HDF5 file.
    
    Parameters:
    -----------
    feature_sets : dict
        Nested dictionary with window IDs, factors, and data splits as keys and feature sets as values.
    factor_columns : list
        List of factor column names.
    output_file : str
        Path to output HDF5 file.
    """
    with h5py.File(output_file, 'w') as hf:
        # Create a group for metadata
        meta = hf.create_group('metadata')
        meta.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta.attrs['total_windows'] = len(feature_sets)
        meta.attrs['total_factors'] = len(factor_columns)
        
        # Store factor columns
        meta.create_dataset('factor_columns', data=np.array(factor_columns, dtype='S'))
        
        # Create a group for feature sets
        feature_sets_group = hf.create_group('feature_sets')
        
        # Loop through each window
        for window_id, window_data in feature_sets.items():
            window_group = feature_sets_group.create_group(f'window_{window_id}')
            
            # Store training, validation, and prediction data
            for split_name, split_data in window_data.items():
                split_group = window_group.create_group(split_name)
                
                # Loop through each factor
                for factor, factor_data in split_data.items():
                    factor_group = split_group.create_group(factor.replace('/', '_'))
                    
                    # Store X features
                    if 'X' in factor_data:
                        X_group = factor_group.create_group('X')
                        
                        # Store column names
                        X_columns = factor_data['X'].columns.tolist()
                        X_group.create_dataset('columns', data=np.array(X_columns, dtype='S'))
                        
                        # Store data
                        X_group.create_dataset('data', data=factor_data['X'].values)
                    
                    # Store y targets (if available)
                    if 'y' in factor_data:
                        y_data = factor_data['y'].values
                        factor_group.create_dataset('y', data=y_data)
                        
                        # Store y index for reference
                        y_index = factor_data['y'].index.tolist()
                        factor_group.create_dataset('y_index', data=np.array([str(idx) for idx in y_index], dtype='S'))

def save_sample_to_excel(feature_sets, factor_columns, output_file, num_windows=3, num_factors=5):
    """
    Save a sample of feature sets to Excel for easy inspection.
    
    Parameters:
    -----------
    feature_sets : dict
        Nested dictionary with window IDs, factors, and data splits as keys and feature sets as values.
    factor_columns : list
        List of factor column names.
    output_file : str
        Path to output Excel file.
    num_windows : int
        Number of windows to sample (default: 3).
    num_factors : int
        Number of factors to sample for each window (default: 5).
    """
    # Check if there are any feature sets to sample
    if not feature_sets:
        print("Warning: No feature sets available to sample. Creating a minimal sample file.")
        # Create a minimal Excel file with just the summary sheet
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_data = {
                'Metric': ['Total Windows', 'Total Factors', 'Sample Generated Date', 'Note'],
                'Value': [0, len(factor_columns), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                         "No feature sets were available to sample. This could indicate an issue with the data."]
            }
            pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
            
            # Add an empty sample sheet to ensure the Excel file has at least one sheet
            pd.DataFrame().to_excel(writer, index=False, sheet_name='Empty_Sample')
        return
        
    # Calculate the feature dimensions (if available)
    try:
        # Find the first window with training data
        for window_id in feature_sets:
            if feature_sets[window_id]['training']:
                # Find the first factor in training data
                first_factor = next(iter(feature_sets[window_id]['training']))
                feature_dimensions = len(feature_sets[window_id]['training'][first_factor]['X'].columns)
                break
        else:
            feature_dimensions = "Unknown"  # No training data found
    except (StopIteration, KeyError, AttributeError):
        feature_dimensions = "Unknown"  # Error finding feature dimensions
        
    # Sample windows evenly
    all_window_ids = sorted(feature_sets.keys())
    if len(all_window_ids) > num_windows:
        sample_indices = np.linspace(0, len(all_window_ids) - 1, num_windows, dtype=int)
        sample_window_ids = [all_window_ids[i] for i in sample_indices]
    else:
        sample_window_ids = all_window_ids
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Add a summary sheet
        summary_data = {
            'Metric': [
                'Total Windows', 
                'Sampled Windows',
                'Total Factors',
                'Sampled Factors per Window',
                'Feature Dimensions',
                'Sample Generated Date'
            ],
            'Value': [
                len(all_window_ids),
                len(sample_window_ids),
                len(factor_columns),
                num_factors,
                feature_dimensions,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
        
        # Make sure we have at least one sheet to prevent Excel errors
        data_sheet_added = False
        
        # For each sampled window
        for window_id in sample_window_ids:
            window_data = feature_sets[window_id]
            
            # Sample factors
            all_factors = list(window_data['training'].keys())
            if not all_factors:
                # Skip windows with no factors
                continue
                
            if len(all_factors) > num_factors:
                random.seed(int(window_id))
                sample_factors = random.sample(all_factors, num_factors)
            else:
                sample_factors = all_factors
            
            # For each sampled factor
            for factor in sample_factors:
                # Get training data
                X_train = window_data['training'][factor]['X']
                y_train = window_data['training'][factor]['y']
                
                # Create a combined DataFrame for easy viewing
                train_df = X_train.copy()
                train_df['Target'] = y_train
                
                # Save to Excel
                sheet_name = f"W{window_id}_{factor[:10]}_Train"[:31]  # Excel limits sheet names to 31 chars
                train_df.to_excel(writer, index=True, sheet_name=sheet_name)
                data_sheet_added = True
                
                # Get validation data
                X_val = window_data['validation'][factor]['X']
                y_val = window_data['validation'][factor]['y']
                
                val_df = X_val.copy()
                val_df['Target'] = y_val
                
                # Save to Excel
                sheet_name = f"W{window_id}_{factor[:10]}_Val"[:31]
                val_df.to_excel(writer, index=True, sheet_name=sheet_name)
                
                # Get prediction data
                X_pred = window_data['prediction'][factor]['X']
                
                # Save to Excel
                sheet_name = f"W{window_id}_{factor[:10]}_Pred"[:31]
                X_pred.to_excel(writer, index=True, sheet_name=sheet_name)
        
        # If no data sheet was added, add an empty one to prevent Excel errors
        if not data_sheet_added:
            pd.DataFrame({'No Data': ['No factor data was found to sample']}).to_excel(
                writer, index=False, sheet_name='No_Data')
            print("Warning: No factor data found to sample. Created a minimal sample file.")

def create_feature_sets_visualizations(feature_sets, factor_columns, output_file):
    """
    Create visualizations for feature sets.
    
    Parameters:
    -----------
    feature_sets : dict
        Nested dictionary with window IDs, factors, and data splits as keys and feature sets as values.
    factor_columns : list
        List of factor column names.
    output_file : str
        Path to output PDF file.
    """
    # Check if we have data to visualize
    has_data = False
    for window_id in feature_sets:
        if feature_sets[window_id]['training']:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No feature data available for visualization. Creating minimal PDF.")
        # Create a minimal PDF with just a title page
        with PdfPages(output_file) as pdf:
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.text(0.5, 0.5, 'Feature Sets Visualizations\nFactor Return Forecasting Project\n\nNO DATA AVAILABLE',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=20)
            plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12)
            plt.text(0.5, 0.3, f'No feature data was available for visualization.\nThis could indicate an issue with the data or processing.',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='red')
            pdf.savefig()
            plt.close()
        return
    
    # Set plot style
    plt.style.use('ggplot')
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Feature Sets Visualizations\nFactor Return Forecasting Project',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        plt.text(0.5, 0.3, f'Total Windows: {len(feature_sets)}\nTotal Factors: {len(factor_columns)}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        pdf.savefig()
        plt.close()
        
        # Find a valid window and factor for visualization
        valid_window_id = None
        valid_factor = None
        
        for window_id in sorted(feature_sets.keys()):
            if feature_sets[window_id]['training']:
                valid_window_id = window_id
                valid_factor = next(iter(feature_sets[window_id]['training'].keys()))
                break
        
        if valid_window_id is None or valid_factor is None:
            print("Warning: Could not find valid data for visualization.")
            return
            
        # Use the valid window and factor for visualizations
        middle_window_id = valid_window_id
        middle_window = feature_sets[middle_window_id]
        sample_factor = valid_factor
        X_train = middle_window['training'][sample_factor]['X']
        
        # === Analysis 1: Feature Value Distributions ===
        plt.figure(figsize=(12, 8))
        
        # Create box plots for each feature
        X_train.boxplot(vert=False)
        plt.title(f'Feature Value Distributions for Window {middle_window_id}, Factor {sample_factor}')
        plt.xlabel('Value')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 2: Feature Correlations ===
        # Create correlation matrix for the features
        plt.figure(figsize=(10, 8))
        
        corr_matrix = X_train.corr()
        
        im = plt.imshow(corr_matrix, cmap='coolwarm')
        plt.colorbar(im, label='Correlation')
        
        # Add labels
        plt.title(f'Feature Correlations for Window {middle_window_id}, Factor {sample_factor}')
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 3: Target Distribution ===
        # Plot the distribution of target values
        plt.figure(figsize=(10, 6))
        
        y_train = middle_window['training'][sample_factor]['y']
        
        plt.hist(y_train, bins=30, alpha=0.7)
        plt.axvline(x=y_train.mean(), color='red', linestyle='--', 
                   label=f'Mean: {y_train.mean():.3f}')
        plt.axvline(x=y_train.median(), color='green', linestyle='--', 
                   label=f'Median: {y_train.median():.3f}')
        
        plt.title(f'Target Distribution for Window {middle_window_id}, Factor {sample_factor}')
        plt.xlabel('Target Value (Next Month Return)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 4: Feature-Target Correlations ===
        # Analyze correlations between features and targets
        plt.figure(figsize=(12, 6))
        
        # Create DataFrame with features and target
        train_df = X_train.copy()
        train_df['Target'] = y_train
        
        # Calculate correlations with target
        target_corrs = train_df.corr()['Target'].drop('Target').sort_values(ascending=False)
        
        # Plot correlations
        plt.barh(range(len(target_corrs)), target_corrs.values)
        plt.yticks(range(len(target_corrs)), target_corrs.index)
        plt.title(f'Feature-Target Correlations for Window {middle_window_id}, Factor {sample_factor}')
        plt.xlabel('Correlation with Next Month Return')
        plt.axvline(x=0, color='black', linestyle='-')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 5: Feature Value Stability ===
        # Analyze how feature values change over training and validation periods
        plt.figure(figsize=(12, 8))
        
        X_train_mean = X_train.mean()
        X_val_mean = middle_window['validation'][sample_factor]['X'].mean()
        
        # Combine into a DataFrame for plotting
        compare_df = pd.DataFrame({
            'Training': X_train_mean,
            'Validation': X_val_mean
        })
        
        compare_df.plot(kind='bar')
        plt.title(f'Feature Mean Values: Training vs Validation\nWindow {middle_window_id}, Factor {sample_factor}')
        plt.ylabel('Mean Value')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Add a summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Calculate some statistics across windows and factors
        feature_count = 0
        feature_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        
        # Sample a few windows for statistics
        available_windows = [w_id for w_id in feature_sets if feature_sets[w_id]['training']]
        if not available_windows:
            sample_windows = []
        else:
            sample_windows = random.sample(available_windows, min(5, len(available_windows)))
        
        for window_id in sample_windows:
            window = feature_sets[window_id]
            
            # Sample a few factors
            available_factors = list(window['training'].keys())
            if not available_factors:
                continue
                
            sample_factors = random.sample(available_factors, min(5, len(available_factors)))
            
            for factor in sample_factors:
                X = window['training'][factor]['X']
                feature_count += X.shape[1]
                
                feature_stats['min'].append(X.min().min())
                feature_stats['max'].append(X.max().max())
                feature_stats['mean'].append(X.mean().mean())
                feature_stats['std'].append(X.std().mean())
        
        # Calculate averages
        for stat in feature_stats:
            if feature_stats[stat]:
                feature_stats[stat] = np.mean(feature_stats[stat])
            else:
                feature_stats[stat] = "N/A"
        
        # Determine features per factor
        if feature_count > 0 and len(sample_windows) > 0 and len(sample_factors) > 0:
            features_per_factor = int(feature_count / (len(sample_windows) * len(sample_factors)))
        else:
            features_per_factor = "N/A"
            
        plt.text(0.5, 0.5, f'Summary of Feature Set Analysis:\n\n'
                          f'Total Windows: {len(feature_sets)}\n'
                          f'Total Factors: {len(factor_columns)}\n'
                          f'Features per Factor: {features_per_factor}\n\n'
                          f'Feature Value Statistics (Sample):\n'
                          f'  Mean Min Value: {feature_stats["min"]}\n'
                          f'  Mean Max Value: {feature_stats["max"]}\n'
                          f'  Mean Average Value: {feature_stats["mean"]}\n'
                          f'  Mean Std Deviation: {feature_stats["std"]}\n\n'
                          f'Data Split Information:\n'
                          f'  Training: 60 months\n'
                          f'  Validation: 6 months\n'
                          f'  Prediction: 1 month\n',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()

# Section: Main Script Logic
def main():
    print("=== Step 8: Create Feature Sets ===")
    
    # --- Step 8.1: Load Data ---
    print("--- 8.1 Loading Data ---")
    
    # Define file paths
    input_factor_file = os.path.join("output", "S2_T2_Optimizer_with_MA.xlsx")
    input_window_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    input_helper_file = os.path.join("output", "S7_Helper_Features.h5")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output files with S8 prefix in output directory
    output_feature_file = os.path.join(output_dir, "S8_Feature_Sets.h5")
    output_sample_file = os.path.join(output_dir, "S8_Feature_Sets_Sample.xlsx")
    output_viz_file = os.path.join(output_dir, "S8_Feature_Sets_Visualization.pdf")
    
    try:
        # Load factor data with moving averages
        factor_data = pd.read_excel(input_factor_file)
        print(f"Loaded factor data from {input_factor_file}")
        print(f"  - Shape: {factor_data.shape}")
        print(f"  - Date range: {factor_data['Date'].min()} to {factor_data['Date'].max()}")
        
        # Load window schedule
        window_schedule = pd.read_excel(input_window_file)
        print(f"Loaded window schedule from {input_window_file}")
        print(f"  - Total windows: {len(window_schedule)}")
        
        # Load helper features
        helper_features, factor_columns = load_helper_features_from_h5(input_helper_file)
        print(f"Loaded helper features from {input_helper_file}")
        print(f"  - Total windows: {len(helper_features)}")
        print(f"  - Total factors: {len(factor_columns)}")
        
    except Exception as e:
        print(f"Error loading input files: {e}")
        sys.exit(1)
    
    # --- Step 8.2: Create Feature Sets ---
    print("\n--- 8.2 Creating Feature Sets ---")
    
    feature_sets = create_feature_sets(factor_data, helper_features, window_schedule, factor_columns)
    
    # Print info about feature sets
    num_features = 14  # 4 own + 10 helper features
    total_features = len(feature_sets) * len(factor_columns) * num_features
    print(f"Created feature sets for {len(feature_sets)} windows and {len(factor_columns)} factors")
    print(f"Each factor has {num_features} features (4 own + 10 helper features)")
    print(f"Total features created: {total_features:,}")
    
    # --- Step 8.3: Save Feature Sets ---
    print("\n--- 8.3 Saving Feature Sets ---")
    
    # Save to HDF5 file
    save_feature_sets_to_h5(feature_sets, factor_columns, output_feature_file)
    print(f"Saved feature sets to {output_feature_file}")
    
    # Save sample to Excel for easy inspection
    save_sample_to_excel(feature_sets, factor_columns, output_sample_file)
    print(f"Saved feature sets sample to {output_sample_file}")
    
    # --- Step 8.4: Generate Visualizations ---
    print("\n--- 8.4 Generating Visualizations ---")
    
    create_feature_sets_visualizations(feature_sets, factor_columns, output_viz_file)
    print(f"Generated feature sets visualizations in {output_viz_file}")
    
    # --- Step 8.5: Summary ---
    print("\n--- 8.5 Summary ---")
    
    # Calculate some statistics
    num_windows = len(feature_sets)
    factors_per_window = [len(window_data['training']) for window_data in feature_sets.values()]
    
    if factors_per_window:
        avg_factors_per_window = np.mean(factors_per_window)
        print(f"Feature Set Summary:")
        print(f"  - Total windows processed: {num_windows}")
        print(f"  - Average factors per window: {avg_factors_per_window:.1f}")
        print(f"  - Features per factor: {num_features}")
    else:
        print(f"Feature Set Summary:")
        print(f"  - Total windows processed: {num_windows}")
        print(f"  - No factor data found in windows")
    
    print(f"Output files:")
    print(f"  - HDF5 file with all feature sets: {output_feature_file}")
    print(f"  - Excel file with sample feature sets: {output_sample_file}")
    print(f"  - PDF with feature set visualizations: {output_viz_file}")
    
    print("\nStep 8 completed successfully!")
    print("Next step: Configure XGBoost models (Step 9)")

# Helper function to find 60-month MA column for a helper factor
def find_helper_ma_column(helper, base_data):
    """
    Find the 60-month MA column for a helper factor.
    
    Parameters:
    -----------
    helper : str
        The name of the helper factor.
    base_data : pandas.DataFrame
        The data containing the MA columns.
        
    Returns:
    --------
    str
        The name of the MA column if found, or the helper factor name if not found.
    """
    # First check if there's a column mapping file from Step 2
    column_mapping_file = os.path.join("output", "S2_Column_Mapping.xlsx")
    if os.path.exists(column_mapping_file):
        try:
            mapping_df = pd.read_excel(column_mapping_file)
            # Check if this helper is in the mapping
            helper_row = mapping_df[mapping_df['original_column'] == helper]
            if not helper_row.empty:
                # Use the 60m column from the mapping
                ma_column = helper_row['column_60m'].iloc[0]
                if ma_column != "N/A" and ma_column in base_data.columns:
                    return ma_column
            
            # Also try with the base name (without CS/TS/period)
            # This covers cases where helper might be "Gold" but column is "Gold_TS"
            for _, row in mapping_df.iterrows():
                if row['factor_name'] == helper and row['column_60m'] != "N/A":
                    ma_column = row['column_60m']
                    if ma_column in base_data.columns:
                        return ma_column
        except Exception as e:
            print(f"Warning: Error reading column mapping file: {e}")
    
    # If mapping file doesn't exist or helper not found, try different patterns
    ma_candidates = [
        f"{helper}_60m",       # Standard pattern (e.g., "Gold_TS_60m")
        f"{helper}_TS_60m",    # With TS suffix
        f"{helper}_CS_60m"     # With CS suffix
    ]
    
    # If helper has CS or TS suffix, also try base name with suffix
    if "_CS" in helper:
        base_name = helper.split("_CS")[0]
        ma_candidates.extend([
            f"{base_name}_CS_60m",
            f"{base_name}_60m"
        ])
    elif "_TS" in helper:
        base_name = helper.split("_TS")[0]
        ma_candidates.extend([
            f"{base_name}_TS_60m",
            f"{base_name}_60m"
        ])
    
    # If helper already has a period suffix (e.g., Gold_TS_3m)
    period_match = re.search(r'_(\d+)m$', helper)
    if period_match:
        # Try adding 60m after existing period (e.g., Gold_TS_3m_60m)
        ma_candidates.append(f"{helper}_60m")
        
        # Also try replacing period with 60m
        base_name = re.sub(r'_\d+m$', '', helper)
        ma_candidates.extend([
            f"{base_name}_60m",
            f"{base_name}_TS_60m",
            f"{base_name}_CS_60m"
        ])
    
    # Use the first available MA column
    for col in ma_candidates:
        if col in base_data.columns:
            return col
    
    # If no MA column found, use the raw helper if available
    if helper in base_data.columns:
        return helper
    
    # Could not find any suitable column
    return None

if __name__ == "__main__":
    main() 