# File Documentation
'''
----------------------------------------------------------------------------------------------------
MODIFIED FOR MULTIFACTOR REGRESSION (ALL FACTORS PREDICTED SIMULTANEOUSLY)
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S2_T2_Optimizer_with_MA.xlsx
  - Path: ../output/S2_T2_Optimizer_with_MA.xlsx (output from Step 2 in parent directory)
  - Description: Excel file containing factor data with all moving averages.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

- S4_Window_Schedule.xlsx
  - Path: ../output/S4_Window_Schedule.xlsx (output from Step 4 in parent directory)
  - Description: Excel file containing the schedule of all 236 rolling windows.
  - Format: Excel (.xlsx) with window parameters and dates.

- S2_Column_Mapping.xlsx
  - Path: ../output/S2_Column_Mapping.xlsx (output from Step 2 in parent directory)
  - Description: Excel file containing mappings between original factor names and their MA columns.
                 This is crucial for identifying the MA columns for all 106 factors.
  - Format: Excel (.xlsx) with mappings between factors and their MA column names.

OUTPUT FILES:
- S8_1_MF_Feature_Sets.h5
  - Path: ./output/S8_1_MF_Feature_Sets.h5 (local to Multifactor/output/)
  - Description: HDF5 file containing the wide feature matrix (X) and multi-target matrix (Y) for each window.
                 X_train: (months_in_training_period, num_total_features (e.g., 106 factors * 4 MAs = 424 features))
                 Y_train: (months_in_training_period, num_target_factors (e.g., 106 factors))
                 Also includes X_val, Y_val, X_pred_features for consistency if needed.
  - Format: HDF5 with window-specific datasets for X_train, Y_train, X_val, Y_val, etc.

# TODO_MULTIFACTOR: Sample Excel and Visualizations need significant adaptation or removal for this version.

----------------------------------------------------------------------------------------------------
Purpose:
This script creates the feature sets for multi-factor regression for each window:
- Input Features (X): A wide matrix comprising 4 MA variants (1, 3, 12, 60-month) for ALL 106 factors.
  (e.g., 106 factors * 4 MAs = 424 features).
- Target Variables (Y): A matrix of next month's returns for ALL 106 factors.

These feature sets (X_train, Y_train_matrix, X_val, Y_val_matrix) will be used to train
multi-output linear models in subsequent steps.

Version: 1.0 (Multifactor Adaptation)
Last Updated: 2025-05-16
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
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
import re
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure output directory for this script exists
output_dir_mf = Path("output") # Local to Multifactor/
output_dir_mf.mkdir(parents=True, exist_ok=True)

# Define root project output directory to load shared inputs
project_root_output = Path("../output")

# Section: Configuration & Constants
RANDOM_SEED = 42
NUM_PROCESSES = min(mp.cpu_count(), 16) # Adjust as needed, can be memory intensive
VERBOSE = True
MA_VARIANTS = ['_MA1', '_MA3', '_MA12', '_MA60'] # Define the MA variants to use

# Ensure deterministic behavior
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Section: Utility Functions
def load_column_mapping(mapping_file_path_str):
    """
    Load column mapping. For MF, we need a list of all base factor series names as they appear
    in S2_Moving_Averages.h5 (expected to be the 'original_column' from S2_Column_Mapping.xlsx).
    The specific MA column names (e.g., FactorName_MA1) will be constructed for features.
    This function ensures it gets all unique base factor series names.
    """
    if VERBOSE: print(f"Loading column mapping and deriving base factor series names from: {mapping_file_path_str}")
    
    column_mapping_file = Path(mapping_file_path_str)
    if not column_mapping_file.exists():
        raise FileNotFoundError(f"Required column mapping file not found at {column_mapping_file}")
    
    try:
        # Assuming the relevant sheet is the first one if not named 'Column_Mapping'
        # Based on previous tool interaction, the sheet is 'Sheet1'
        df_mapping = pd.read_excel(column_mapping_file, sheet_name='Sheet1') 
    except Exception as e:
        raise ValueError(f"Could not read S2_Column_Mapping.xlsx file at {column_mapping_file}. Error: {e}")

    if 'original_column' not in df_mapping.columns:
        raise ValueError("'original_column' not found in S2_Column_Mapping.xlsx. Cannot derive base factor series names.")

    # These are the names of the series in S2_Moving_Averages.h5 that we will calculate MAs on
    base_series_names = sorted(list(df_mapping['original_column'].unique()))
    
    if not base_series_names:
        raise ValueError("No base factor series names could be extracted from 'original_column' in S2_Column_Mapping.xlsx.")

    # Construct the MA column names that this script will generate for its feature set
    factor_ma_col_mapping = {}
    for series_name in base_series_names:
        factor_ma_col_mapping[series_name] = [f"{series_name}{suffix}" for suffix in MA_VARIANTS]

    if VERBOSE: print(f"Derived MA column mapping for {len(base_series_names)} base factor series.")
    # The second returned item should be the list of base names that will be used to fetch raw series for MA calculation
    return factor_ma_col_mapping, base_series_names

def get_windows_data(factor_data, window_schedule):
    """ Precompute data subsets for each window. """
    if VERBOSE: print("Slicing window data subsets...")
    windows_data = {}
    if not pd.api.types.is_datetime64_any_dtype(factor_data['Date']):
        factor_data['Date'] = pd.to_datetime(factor_data['Date'])
        
    for _, row in tqdm(window_schedule.iterrows(), total=len(window_schedule), desc="Slicing Window Data"):
        window_id = row['Window_ID']
        train_start, train_end = pd.to_datetime(row['Training_Start_Date']), pd.to_datetime(row['Training_End_Date'])
        val_start, val_end = pd.to_datetime(row['Validation_Start_Date']), pd.to_datetime(row['Validation_End_Date'])
        # Prediction date is a single month in the schedule
        pred_date_single = pd.to_datetime(row['Prediction_Date'])
        
        windows_data[window_id] = {
            'training': factor_data[(factor_data['Date'] >= train_start) & (factor_data['Date'] <= train_end)].copy(),
            'validation': factor_data[(factor_data['Date'] >= val_start) & (factor_data['Date'] <= val_end)].copy(),
            # Prediction features are typically for this single date
            'prediction': factor_data[factor_data['Date'] == pred_date_single].copy(),
        }
    return windows_data

def precompute_multi_target_next_month_returns(factor_data, all_factor_base_names, date_column='Date'):
    """
    Precompute next month returns for ALL factors. Returns a DataFrame with Date and FXXX_NextReturn columns.
    """
    if VERBOSE: print("Precomputing multi-target next month returns...")
    
    if not pd.api.types.is_datetime64_any_dtype(factor_data[date_column]):
        factor_data[date_column] = pd.to_datetime(factor_data[date_column])
    
    data_indexed = factor_data.set_index(date_column)
    all_returns_df = pd.DataFrame(index=data_indexed.index)
    
    for factor_name in tqdm(all_factor_base_names, desc="Calculating Next Month Returns"):
        if factor_name in data_indexed.columns: # Base factor column (e.g., F001) should contain its raw returns
            all_returns_df[f'{factor_name}_NextReturn'] = data_indexed[factor_name].shift(-1)
        else:
            if VERBOSE: print(f"  Warning: Base factor column {factor_name} not found in factor_data for return calculation. Its NextReturn will be NaN.")
            all_returns_df[f'{factor_name}_NextReturn'] = np.nan
            
    return all_returns_df.reset_index() 

def create_mf_feature_sets_for_window_period(window_id, period_name, current_period_df, all_factor_base_names, factor_ma_col_mapping, multi_target_next_returns_df):
    """
    Creates wide feature matrix (X) and multi-target matrix (Y) for a specific window's specific period (train/val/pred).
    """
    if current_period_df.empty:
        if VERBOSE: print(f"  Window {window_id}, period {period_name}: Data is empty. Returning None.")
        return None

    if VERBOSE: print(f"  Creating MF feature set for window {window_id}, period {period_name} (shape: {current_period_df.shape}) ...")
    merged_data = pd.merge(current_period_df, multi_target_next_returns_df, on='Date', how='left')

    # Construct X: all MA variants for all factors
    feature_columns_for_x = []
    for factor_base_name in all_factor_base_names:
        if factor_base_name in factor_ma_col_mapping:
            feature_columns_for_x.extend(factor_ma_col_mapping[factor_base_name])
        else: # Should not happen if factor_ma_col_mapping is comprehensive
            if VERBOSE: print(f"    Warning: Base factor {factor_base_name} not in factor_ma_col_mapping for X features in window {window_id}, period {period_name}.")
            feature_columns_for_x.extend([f"{factor_base_name}{suffix}" for suffix in MA_VARIANTS]) 
    
    feature_columns_for_x = sorted(list(set(feature_columns_for_x))) 
    
    existing_x_cols_in_data = [col for col in feature_columns_for_x if col in merged_data.columns]
    if len(existing_x_cols_in_data) != len(feature_columns_for_x) and VERBOSE:
        missing = set(feature_columns_for_x) - set(existing_x_cols_in_data)
        print(f"    Warning: For Win {window_id}/{period_name}, {len(missing)} X feature columns were expected but not found in data: {missing}. Using available columns.")
    
    X_df = merged_data[existing_x_cols_in_data].copy()

    # Construct Y: next month returns for all factors
    target_columns_for_y = [f'{f}_NextReturn' for f in all_factor_base_names]
    existing_y_cols_in_data = [col for col in target_columns_for_y if col in merged_data.columns]
    if len(existing_y_cols_in_data) != len(target_columns_for_y) and VERBOSE:
        missing = set(target_columns_for_y) - set(existing_y_cols_in_data)
        print(f"    Warning: For Win {window_id}/{period_name}, {len(missing)} Y target columns were expected but not found in data: {missing}. Using available columns.")

    Y_df = merged_data[existing_y_cols_in_data].copy()
    
    # Ensure X and Y have same number of rows as original period slice after merge
    if len(X_df) != len(current_period_df) or len(Y_df) != len(current_period_df):
         if VERBOSE: print(f"    Warning: Row count mismatch for Win {window_id}/{period_name}. X:{len(X_df)}, Y:{len(Y_df)}, Original period:{len(current_period_df)}")

    return {'X': X_df, 'Y': Y_df, 'dates': merged_data['Date'].copy(), 
            'X_colnames': X_df.columns.tolist(), 'Y_colnames': Y_df.columns.tolist()}

def save_mf_feature_sets_to_h5(mf_feature_sets_all_windows, output_file, all_base_factor_names_ordered, all_x_feature_names_ordered):
    """
    Save the multi-factor feature sets (wide X, multi-target Y) to HDF5.
    """
    if VERBOSE: print(f"Saving MF feature sets to {output_file}...")
    with h5py.File(output_file, 'w') as hf:
        # Store global attributes for interpreting the data structure
        hf.attrs['base_factor_names_ordered'] = [bfn.encode('utf-8') for bfn in all_base_factor_names_ordered]
        hf.attrs['x_feature_names_ordered'] = [xfn.encode('utf-8') for xfn in all_x_feature_names_ordered]
        hf.attrs['y_target_names_ordered'] = [f"{bfn}_NextReturn".encode('utf-8') for bfn in all_base_factor_names_ordered]
        hf.attrs['MA_variants_used'] = [mav.encode('utf-8') for mav in MA_VARIANTS]

        for window_id, data_periods in tqdm(mf_feature_sets_all_windows.items(), desc="Saving to HDF5"):
            win_group = hf.create_group(str(window_id))
            for period_name, period_data_dict in data_periods.items(): 
                if period_data_dict and isinstance(period_data_dict, dict) and 'X' in period_data_dict and 'Y' in period_data_dict:
                    X_df = period_data_dict['X']
                    Y_df = period_data_dict['Y']
                    dates_series = period_data_dict['dates']
                    
                    # Ensure X_df columns match all_x_feature_names_ordered, reindex if necessary filling with NaN
                    X_df_aligned = X_df.reindex(columns=all_x_feature_names_ordered, fill_value=np.nan)
                    # Ensure Y_df columns match Y_target_names_ordered
                    Y_df_aligned = Y_df.reindex(columns=[f"{bfn}_NextReturn" for bfn in all_base_factor_names_ordered], fill_value=np.nan)

                    win_group.create_dataset(f'{period_name}_X', data=X_df_aligned.to_numpy(dtype=np.float32), compression='gzip')
                    win_group.create_dataset(f'{period_name}_Y', data=Y_df_aligned.to_numpy(dtype=np.float32), compression='gzip')
                    win_group.create_dataset(f'{period_name}_dates', data=dates_series.astype('int64').to_numpy(), compression='gzip')
                else:
                    if VERBOSE: print(f"  Skipping save for window {window_id}, period {period_name} due to missing/invalid data dict.")
    if VERBOSE: print("MF Feature sets saved.")

def process_single_window_for_mf_sets(window_id_and_data, all_factor_base_names, factor_ma_col_mapping, multi_target_next_returns_df):
    window_id, period_dataframes_for_window = window_id_and_data
    window_results = {}
    for period_name in ['training', 'validation', 'prediction']:
        current_period_df = period_dataframes_for_window[period_name]
        mf_set_dict = create_mf_feature_sets_for_window_period(
            window_id, period_name, current_period_df,
            all_factor_base_names, factor_ma_col_mapping, multi_target_next_returns_df
        )
        window_results[period_name] = mf_set_dict
    return window_id, window_results

def main():
    """ Main execution function. """
    start_time = time.time()
    print("="*80)
    print("=== Step 8.1 (Multifactor): Creating Wide Feature Sets ===")
    print("="*80)

    factor_data_file = project_root_output / "S2_T2_Optimizer_with_MA.xlsx"
    window_schedule_file = project_root_output / "S4_Window_Schedule.xlsx"
    column_mapping_file_str = str(project_root_output / "S2_Column_Mapping.xlsx")

    mf_feature_sets_output_h5 = output_dir_mf / "S8_1_MF_Feature_Sets.h5"

    for file_path_obj in [factor_data_file, window_schedule_file, Path(column_mapping_file_str)]:
        if not file_path_obj.exists():
            print(f"Error: Input file {file_path_obj} not found."); return

    try:
        if VERBOSE: print("\nStep 1: Loading input data and mappings...")
        factor_data_full = pd.read_excel(factor_data_file)
        window_schedule = pd.read_excel(window_schedule_file)
        factor_ma_col_mapping, all_factor_base_names = load_column_mapping(column_mapping_file_str)
        if not all_factor_base_names:
            print("Error: No base factor names derived. Exiting."); return
        if VERBOSE: print(f"Using {len(all_factor_base_names)} base factors for MF features.")

        # Define the full ordered list of X features (e.g., F001_MA1, F001_MA3, ..., F106_MA60)
        all_x_feature_names_ordered = []
        for bf_name in all_factor_base_names:
            all_x_feature_names_ordered.extend(factor_ma_col_mapping[bf_name])
        all_x_feature_names_ordered = sorted(list(set(all_x_feature_names_ordered))) # Should be 106*4 = 424 features

        if VERBOSE: print("\nStep 2: Preprocessing data for windows and returns...")
        window_specific_period_dataframes = get_windows_data(factor_data_full, window_schedule) # {win_id: {'training': df, ...}}
        multi_target_next_returns_df = precompute_multi_target_next_month_returns(factor_data_full, all_factor_base_names)

        if VERBOSE: print("\nStep 3: Creating multi-factor feature sets...")
        all_windows_mf_data = {}
        
        items_to_process = list(window_specific_period_dataframes.items())

        if NUM_PROCESSES > 1 and len(items_to_process) > 1:
            if VERBOSE: print(f"Using {NUM_PROCESSES} processes for parallel window processing...")
            # Prepare partial function for multiprocessing
            partial_process_func = partial(process_single_window_for_mf_sets,
                                           all_factor_base_names=all_factor_base_names,
                                           factor_ma_col_mapping=factor_ma_col_mapping,
                                           multi_target_next_returns_df=multi_target_next_returns_df)
            with mp.Pool(processes=NUM_PROCESSES) as pool:
                results = list(tqdm(pool.imap(partial_process_func, items_to_process), total=len(items_to_process), desc="Processing Windows (Parallel)"))
            for window_id, window_result in results:
                all_windows_mf_data[window_id] = window_result
        else:
            if VERBOSE: print("Processing windows sequentially...")
            for window_id, period_dataframes_for_window in tqdm(items_to_process, desc="Processing Windows (Sequential)"):
                window_results_s = {}
                for period_name_s in ['training', 'validation', 'prediction']:
                    current_period_df_s = period_dataframes_for_window[period_name_s]
                    mf_set_dict_s = create_mf_feature_sets_for_window_period(
                        window_id, period_name_s, current_period_df_s,
                        all_factor_base_names, factor_ma_col_mapping, multi_target_next_returns_df
                    )
                    window_results_s[period_name_s] = mf_set_dict_s
                all_windows_mf_data[window_id] = window_results_s

        if VERBOSE: print("\nStep 4: Saving MF feature sets...")
        save_mf_feature_sets_to_h5(all_windows_mf_data, mf_feature_sets_output_h5, all_factor_base_names, all_x_feature_names_ordered)
        
        elapsed_time = time.time() - start_time
        print(f"\nStep 8.1 (MF) completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Output saved to: {mf_feature_sets_output_h5}")
        print("="*80)
        
    except Exception as e:
        print(f"Error in Step 8.1 (MF): {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
