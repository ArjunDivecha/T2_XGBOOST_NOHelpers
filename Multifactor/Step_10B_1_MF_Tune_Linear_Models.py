# File Documentation
'''
----------------------------------------------------------------------------------------------------
MODIFIED FOR MULTIFACTOR REGRESSION (ALL FACTORS PREDICTED SIMULTANEOUSLY)
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S8_1_MF_Feature_Sets.h5
  - Path: ./output/S8_1_MF_Feature_Sets.h5 (output from Step 8.1 MF)
  - Description: HDF5 file containing the WIDE feature matrix (X) and MULTI-TARGET matrix (Y)
                 for each window. X is (num_samples, num_total_features), Y is (num_samples, num_target_factors).
  - Format: HDF5 with window-specific datasets.

- S4_Window_Schedule.xlsx
  - Path: ../output/S4_Window_Schedule.xlsx (output from Step 4 in parent directory)
  - Description: Excel file containing the schedule of all 236 rolling windows.
  - Format: Excel (.xlsx) with window parameters and dates.

OUTPUT FILES:
- S10B_1_MF_Linear_Models_Tuning_Results.xlsx
  - Path: ./output/S10B_1_MF_Linear_Models_Tuning_Results.xlsx
  - Description: Detailed results of parameter tuning for linear models on multi-factor data.
                 Metrics are typically averaged across all target factors.
  - Format: Excel (.xlsx).

- S10B_1_MF_Linear_Models_Optimal_Params.json
  - Path: ./output/S10B_1_MF_Linear_Models_Optimal_Params.json
  - Description: JSON file containing ONE set of optimal parameters for each linear model type
                 (e.g., Ridge) that performs best on average across all target factors.
  - Format: JSON.

# TODO_MULTIFACTOR: Visualizations (PDF) and Sample Predictions (Excel) require significant adaptation for multi-output models.
# These are simplified or placeholders in this version.

----------------------------------------------------------------------------------------------------
Purpose:
This script tunes different types of linear regression models for a MULTI-FACTOR setup:
1. Ordinary Least Squares (OLS)
2. Ridge Regression (with alpha parameters)
3. LASSO Regression (with alpha parameters)
4. Non-Negative Least Squares (NNLS) - Note: sklearn's LinearRegression(positive=True) can be used.

For each model type, it tests different hyperparameter configurations using the wide feature matrix (X)
and the multi-target matrix (Y). The goal is to find ONE set of optimal hyperparameters per model
type that performs best (e.g., lowest average RMSE) across ALL target factors simultaneously.

Version: 1.0 (Multifactor Adaptation)
Last Updated: 2025-05-16
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
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
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from itertools import product

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

# Suppress warnings
warnings.filterwarnings('ignore')

# Define root project output directory to load shared inputs like S4_Window_Schedule.xlsx
project_root_output_parent = Path("../output") 

# Section: Configuration & Constants
RANDOM_SEED = 42
NUM_TUNING_WINDOWS = 5       # Number of windows to use for tuning. Use a small number for faster tuning.
VERBOSE = True               # Print detailed progress information
N_JOBS_SKLEARN = -1          # For sklearn models that support it, -1 means use all processors.
                             # For Ridge/Lasso, this is not a direct param for .fit(), but good to keep in mind.

# Output directory for this Multifactor script's results
output_dir_mf = Path("output") 
output_dir_mf.mkdir(parents=True, exist_ok=True)

# Ensure deterministic behavior
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Section: Utility Functions
def load_mf_data():
    """
    Load the window schedule and the MULTIFACTOR feature sets (S8_1_MF_Feature_Sets.h5).
    Feature sets will be structured as: feature_sets[window_id]['X_train'], feature_sets[window_id]['Y_train'] (multi-target).
    """
    if VERBOSE: print("Loading multifactor data...")
    
    window_schedule_file = project_root_output_parent / "S4_Window_Schedule.xlsx"
    # The MF feature set is in ./output/ relative to this script's location (Multifactor/)
    feature_sets_file = output_dir_mf.parent / "output" / "S8_1_MF_Feature_Sets.h5" 
    # Correction: S8_1_MF_Feature_Sets.h5 is created by Step_8_1 which is in Multifactor folder itself.
    # So the output path for S8_1_MF is ./output/S8_1_MF_Feature_Sets.h5 relative to Multifactor/
    # Thus, for Step_10B_1_MF (also in Multifactor/), it's in its own output folder.
    feature_sets_file_corrected = output_dir_mf / "S8_1_MF_Feature_Sets.h5" 

    if not feature_sets_file_corrected.exists():
        # Check the original intended path if the corrected one fails, assuming Step_8_1 output was meant to be one level up
        # This was an error in my previous thinking about paths.
        # Step_8_1_MF_Create_Feature_Sets.py's output_dir_mf = Path("output") which is Multifactor/output/
        # So, Step_10B_1_MF_Tune_Linear_Models.py (in Multifactor/) should read from Multifactor/output/
        alt_path_check = Path("./output/S8_1_MF_Feature_Sets.h5") # relative to current script
        if alt_path_check.exists():
             feature_sets_file_corrected = alt_path_check
        else:
             raise FileNotFoundError(f"Multifactor feature set file not found at {feature_sets_file_corrected} or {alt_path_check}")

    if VERBOSE: print(f"  Loading window schedule from: {window_schedule_file}")
    window_schedule = pd.read_excel(window_schedule_file)
    
    if VERBOSE: print(f"  Loading MF feature sets from: {feature_sets_file_corrected}")
    feature_sets = {}
    with h5py.File(feature_sets_file_corrected, 'r') as hf:
        # Optional: Load global attributes if needed later (e.g., column names)
        # x_colnames = [col.decode('utf-8') for col in hf.attrs.get('x_feature_names_ordered', [])]
        # y_colnames = [col.decode('utf-8') for col in hf.attrs.get('y_target_names_ordered', [])]
        
        for window_id_str in tqdm(hf.keys(), desc="Loading H5 Window Data"):
            window_id = int(window_id_str) # Assuming window_id is stored as string key
            feature_sets[window_id] = {}
            for period in ['training', 'validation', 'prediction']:
                try:
                    feature_sets[window_id][f'X_{period}'] = pd.DataFrame(hf[window_id_str][f'{period}_X'][:])
                    feature_sets[window_id][f'Y_{period}'] = pd.DataFrame(hf[window_id_str][f'{period}_Y'][:])
                    # Dates might be useful for some visualizations later, but not directly for tuning here
                    # feature_sets[window_id][f'{period}_dates'] = pd.to_datetime(hf[window_id_str][f'{period}_dates'][:])
                except KeyError as e:
                    if VERBOSE: print(f"    Warning: Missing data for window {window_id}, period {period}, key {e}. Skipping.")
                    feature_sets[window_id][f'X_{period}'] = pd.DataFrame() # Empty DF
                    feature_sets[window_id][f'Y_{period}'] = pd.DataFrame()
                    
    if VERBOSE: print(f"Loaded data for {len(feature_sets)} windows.")
    return window_schedule, feature_sets

def select_tuning_windows(window_schedule, num_windows=NUM_TUNING_WINDOWS):
    """ Randomly select a subset of windows for tuning. """
    if VERBOSE: print(f"Selecting {num_windows} windows for tuning...")
    
    # Ensure num_windows is not greater than available windows
    available_windows = window_schedule['Window_ID'].unique()
    if num_windows > len(available_windows):
        if VERBOSE: print(f"  Warning: Requested {num_windows} tuning windows, but only {len(available_windows)} available. Using all available.")
        num_windows = len(available_windows)
        
    selected_ids = random.sample(list(available_windows), num_windows)
    if VERBOSE: print(f"  Selected window IDs: {selected_ids}")
    return selected_ids

def define_parameter_grids():
    """ Define parameter grids for OLS, Ridge, and Lasso. NNLS (positive=True for LinearRegression) has no specific hyperparameters to tune here beyond fit_intercept. """
    param_grids = {
        'OLS': {
            'fit_intercept': [True, False]
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'solver': ['auto'] # 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000] # Lasso might need more iterations to converge
        },
        'NNLS': { # Achieved with LinearRegression(positive=True)
            'fit_intercept': [True, False] # If True, only intercept can be negative.
                                          # If False, all coeffs must be non-negative.
        }
    }
    if VERBOSE: print("Defined parameter grids.")
    return param_grids

# Model Training and Evaluation Functions (adapted for multi-target)
def train_and_evaluate_model(X_train, Y_train, X_val, Y_val, model_type, params):
    """Trains a specified model type with given parameters and evaluates on validation data."""
    try:
        if model_type == 'OLS':
            model = LinearRegression(**{k: v for k, v in params.items() if k in LinearRegression().get_params()})
        elif model_type == 'Ridge':
            model = Ridge(**{k: v for k, v in params.items() if k in Ridge().get_params()})
        elif model_type == 'Lasso':
            model = Lasso(**{k: v for k, v in params.items() if k in Lasso().get_params()}, random_state=RANDOM_SEED)
        elif model_type == 'NNLS':
            # For NNLS, use LinearRegression with positive=True
            # fit_intercept is the main param from our grid
            model = LinearRegression(positive=True, fit_intercept=params.get('fit_intercept', True))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, Y_train)
        Y_pred_val = model.predict(X_val)
        
        # For multi-output, sklearn metrics return array of scores (one per target) by default
        # We want to average these to get a single performance score for this hyperparameter set.
        val_mse = mean_squared_error(Y_val, Y_pred_val, multioutput='uniform_average')
        val_rmse = np.sqrt(val_mse) # Calculate RMSE from MSE
        val_mae = mean_absolute_error(Y_val, Y_pred_val, multioutput='uniform_average')
        val_r2 = r2_score(Y_val, Y_pred_val, multioutput='uniform_average')
        
        return {'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2}
    except Exception as e:
        if VERBOSE: print(f"    Error training/evaluating {model_type} with params {params}: {e}")
        return {'val_rmse': np.nan, 'val_mae': np.nan, 'val_r2': np.nan}

def evaluate_parameter_combination_mf(args):
    """ 
    Worker function for parallel processing. Evaluates one param combination over selected windows.
    Receives (model_type, params, feature_sets_subset, selected_windows_for_this_worker).
    feature_sets_subset contains only the data for selected_windows_for_this_worker.
    """
    model_type, params, feature_sets, selected_windows = args
    
    all_metrics = []
    for window_id in selected_windows:
        # Use 'X_training' and 'Y_training' to match HDF5 loading keys
        if window_id not in feature_sets or feature_sets[window_id]['X_training'].empty or feature_sets[window_id]['Y_training'].empty:
            if VERBOSE > 1: print(f"      Skipping window {window_id} for {model_type} params {params} due to missing/empty data.")
            continue

        X_train_df = feature_sets[window_id]['X_training']
        Y_train_df = feature_sets[window_id]['Y_training']
        X_val_df = feature_sets[window_id]['X_validation']
        Y_val_df = feature_sets[window_id]['Y_validation']
        
        # Drop rows with any NaNs in Y_train or Y_val for reliable training/evaluation
        # This is crucial as shifted returns can create NaNs at ends of series.
        # Align X and Y by index before dropping NaNs from Y, then select X rows.
        
        # Training data prep
        valid_train_indices = Y_train_df.dropna().index
        Y_train_clean = Y_train_df.loc[valid_train_indices]
        X_train_clean = X_train_df.loc[valid_train_indices]

        # Validation data prep
        valid_val_indices = Y_val_df.dropna().index
        Y_val_clean = Y_val_df.loc[valid_val_indices]
        X_val_clean = X_val_df.loc[valid_val_indices]

        # Fill NaNs in feature matrices (X) with 0 before training/evaluation
        X_train_clean = X_train_clean.fillna(0)
        X_val_clean = X_val_clean.fillna(0)

        if X_train_clean.empty or Y_train_clean.empty or X_val_clean.empty or Y_val_clean.empty:
            if VERBOSE > 1: print(f"      Skipping window {window_id} for {model_type} params {params} after NaN drop (empty DFs).")
            continue
            
        # Convert to numpy for sklearn, ensuring consistent column order (already handled by H5 read if DFs)
        metrics = train_and_evaluate_model(X_train_clean.values, Y_train_clean.values, 
                                           X_val_clean.values, Y_val_clean.values, 
                                           model_type, params)
        all_metrics.append(metrics)

    if not all_metrics:
        return {'params': params, 'avg_val_rmse': np.nan, 'avg_val_mae': np.nan, 'avg_val_r2': np.nan, 'num_eval_windows': 0}

    avg_metrics = pd.DataFrame(all_metrics).mean().to_dict()
    return {
        'params': params,
        'avg_val_rmse': avg_metrics.get('val_rmse', np.nan),
        'avg_val_mae': avg_metrics.get('val_mae', np.nan),
        'avg_val_r2': avg_metrics.get('val_r2', np.nan),
        'num_eval_windows': len(all_metrics)
    }

def perform_grid_search_mf(param_grids, feature_sets, selected_windows):
    """ Perform grid search over parameter combinations for multi-factor models. """
    if VERBOSE: print(f"Performing grid search for MF models over {len(selected_windows)} windows...")
    
    tuning_results = {}
    num_cpus_mp = min(mp.cpu_count(), 8) # Limit CPUs for this part to avoid excessive overhead

    for model_type, grid in param_grids.items():
        if VERBOSE: print(f"  Tuning {model_type}...")
        param_combinations = list(ParameterGrid(grid))
        if VERBOSE: print(f"    Total parameter combinations for {model_type}: {len(param_combinations)}")
        
        results_for_model = []
        
        # Prepare args for multiprocessing
        # feature_sets_subset can be large, so ensure it's efficiently passed or use shared memory if possible (complex)
        # For now, standard pickling will pass it.
        args_list = [(model_type, params, feature_sets, selected_windows) for params in param_combinations]

        if len(param_combinations) > 1 and len(selected_windows) > 0 and num_cpus_mp > 1:
            if VERBOSE: print(f"    Using {num_cpus_mp} processes for {model_type} grid search...")
            with mp.Pool(processes=num_cpus_mp) as pool:
                # Use imap for progress with tqdm if many tasks, or map for simpler cases
                model_param_results = list(tqdm(pool.imap(evaluate_parameter_combination_mf, args_list), total=len(args_list), desc=f"Tuning {model_type}"))
            results_for_model.extend(model_param_results)
        else:
            if VERBOSE: print(f"    Processing {model_type} sequentially...")
            for arg_set in tqdm(args_list, desc=f"Tuning {model_type} (Sequential)"):
                 results_for_model.append(evaluate_parameter_combination_mf(arg_set))

        tuning_results[model_type] = results_for_model
        
    return tuning_results

def find_best_parameters_mf(tuning_results):
    """ Find the best parameter set for each model type based on average validation RMSE. """
    if VERBOSE: print("Finding best parameters for MF models...")
    best_params_overall = {}
    for model_type, results in tuning_results.items():
        if not results:
            if VERBOSE: print(f"  No results for {model_type} to find best params.")
            best_params_overall[model_type] = {'params': None, 'avg_val_rmse': np.nan, 'avg_val_mae': np.nan, 'avg_val_r2': np.nan}
            continue
        
        # Filter out results where metrics are NaN (e.g., if all windows failed for a param set)
        valid_results = [r for r in results if pd.notna(r.get('avg_val_rmse'))]
        if not valid_results:
            if VERBOSE: print(f"  All results for {model_type} had NaN RMSE.")
            best_params_overall[model_type] = {'params': results[0]['params'] if results else None, 'avg_val_rmse': np.nan, 'avg_val_mae': np.nan, 'avg_val_r2': np.nan}
            continue
            
        # Sort by 'avg_val_rmse' (ascending, lower is better)
        best_run = sorted(valid_results, key=lambda x: x['avg_val_rmse'])[0]
        best_params_overall[model_type] = best_run
        if VERBOSE: print(f"  Best for {model_type}: RMSE={best_run['avg_val_rmse']:.4f} with params {best_run['params']}")
        
    return best_params_overall

# Section: Saving Results
def save_mf_tuning_results(tuning_results, output_file_xlsx):
    if VERBOSE: print(f"Saving MF tuning results to {output_file_xlsx}...")
    with pd.ExcelWriter(output_file_xlsx, engine='openpyxl') as writer:
        for model_type, results_list in tuning_results.items():
            if results_list:
                df = pd.DataFrame(results_list)
                # Expand the 'params' dict into separate columns for readability
                params_df = df['params'].apply(pd.Series)
                df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
                df.to_excel(writer, sheet_name=model_type, index=False)
            else:
                if VERBOSE: print(f"  No tuning results to save for {model_type}.")
    if VERBOSE: print("  MF Tuning results saved.")

def save_mf_optimal_parameters(best_params, output_file_json):
    if VERBOSE: print(f"Saving MF optimal parameters to {output_file_json}...")
    # Convert complex objects (like np.bool_) to standard Python types for JSON
    serializable_best_params = {}
    for model, params_dict in best_params.items():
        serializable_best_params[model] = {}
        for key, value in params_dict.items():
            if key == 'params' and isinstance(value, dict):
                serializable_best_params[model][key] = {p_key: (v.item() if hasattr(v, 'item') else v) for p_key, v in value.items()}
            else:
                serializable_best_params[model][key] = value.item() if hasattr(value, 'item') else value

    with open(output_file_json, 'w') as f:
        json.dump(serializable_best_params, f, indent=4)
    if VERBOSE: print("  MF Optimal parameters saved.")

# TODO_MULTIFACTOR: Visualization and Sample Predictions need careful adaptation for multi-output.
# For now, these are placeholders or significantly simplified.

def create_mf_tuning_visualization(tuning_results, best_params, output_file_pdf):
    if VERBOSE: print(f"Creating MF tuning visualization (simplified) to {output_file_pdf}...")
    # This is a very basic placeholder. Proper visualization of multi-output tuning is complex.
    try:
        with PdfPages(output_file_pdf) as pdf:
            for model_type, results in tuning_results.items():
                if not results or not best_params.get(model_type):
                    continue
                
                df = pd.DataFrame(results)
                if 'avg_val_rmse' not in df.columns or df['avg_val_rmse'].isnull().all():
                    continue

                plt.figure(figsize=(10, 6))
                # Example: Plot RMSE vs. a dominant hyperparameter (e.g., alpha for Ridge/Lasso)
                # This needs to be made generic or handle cases where 'alpha' isn't the param.
                param_key_to_plot = None
                if model_type in ['Ridge', 'Lasso'] and results:
                    # Check if 'alpha' was varied
                    alphas_varied = len(set(p['params'].get('alpha', None) for p in results)) > 1
                    if alphas_varied:
                        param_key_to_plot = 'alpha'
                        df['alpha_plot'] = df['params'].apply(lambda p: p.get('alpha', np.nan))
                        df_sorted = df.sort_values(by='alpha_plot')
                        plt.plot(df_sorted['alpha_plot'], df_sorted['avg_val_rmse'], marker='o', label='Avg Val RMSE')
                        plt.xlabel('Alpha (log scale if appropriate)')
                        try: 
                            plt.xscale('log')
                        except: pass # In case alpha is not suitable for log scale

                elif model_type in ['OLS', 'NNLS'] and results:
                     # For OLS/NNLS, 'fit_intercept' is boolean. Maybe bar plot?
                     # For now, just a title indicating the best.
                     pass # Simpler plot needed here.
                
                plt.title(f'Tuning for {model_type} - Avg RMSE vs. {param_key_to_plot if param_key_to_plot else "Params"}')
                plt.ylabel('Average Validation RMSE')
                best_model_params = best_params[model_type]
                if best_model_params and pd.notna(best_model_params.get('avg_val_rmse')):
                    plt.scatter( (best_model_params['params'].get(param_key_to_plot) if param_key_to_plot and best_model_params['params'] else 0.5 if not param_key_to_plot else np.nan),
                                best_model_params['avg_val_rmse'], 
                                color='red', s=100, label=f"Best ({best_model_params['avg_val_rmse']:.4f})", zorder=5)
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
        if VERBOSE: print("  MF Tuning visualization saved (simplified).")
    except Exception as e:
        if VERBOSE: print(f"  Error creating MF visualization: {e}")

# Section: Main Execution
def main():
    print("\n" + "="*80)
    print("STEP 10B.1 (Multifactor): TUNE LINEAR MODELS")
    print("="*80)
    
    start_time = time.time()
    
    # Output files (local to Multifactor/output/)
    tuning_results_excel_output = output_dir_mf / "S10B_1_MF_Linear_Models_Tuning_Results.xlsx"
    optimal_params_json_output = output_dir_mf / "S10B_1_MF_Linear_Models_Optimal_Params.json"
    visualization_pdf_output = output_dir_mf / "S10B_1_MF_Linear_Models_Tuning_Visualization.pdf"
    # sample_predictions_output = output_dir_mf / "S10B_1_MF_Linear_Models_Sample_Predictions.xlsx" # Deferred
    
    try:
        print("\n--- 10B.1.1 Loading Data ---")
        window_schedule, feature_sets_all_windows = load_mf_data()
        
        print("\n--- 10B.1.2 Selecting Windows for Tuning ---")
        selected_windows = select_tuning_windows(window_schedule, NUM_TUNING_WINDOWS)
        
        # Filter feature_sets to only include selected_windows to reduce memory for child processes
        feature_sets_for_tuning = {win_id: feature_sets_all_windows[win_id] for win_id in selected_windows if win_id in feature_sets_all_windows}
        if not feature_sets_for_tuning:
            print("Error: No valid data found for the selected tuning windows. Exiting.")
            return

        print("\n--- 10B.1.3 Defining Parameter Grids ---")
        param_grids = define_parameter_grids()
        
        print("\n--- 10B.1.4 Performing Grid Search for MF Models ---")
        start_grid_time = time.time()
        mf_tuning_results = perform_grid_search_mf(param_grids, feature_sets_for_tuning, selected_windows)
        grid_elapsed_time = time.time() - start_grid_time
        print(f"MF Grid search completed in {grid_elapsed_time:.2f} seconds ({grid_elapsed_time/60:.2f} minutes)")
        
        print("\n--- 10B.1.5 Finding Best Parameters for MF Models ---")
        mf_best_params = find_best_parameters_mf(mf_tuning_results)
        
        print("\nBest parameters for each MF model type (averaged over all target factors):")
        for model_type, params_info in mf_best_params.items():
            print(f"\n  {model_type}:")
            if params_info and params_info.get('params'):
                print(f"    Parameters: {params_info['params']}")
                print(f"    Avg Validation RMSE: {params_info.get('avg_val_rmse', np.nan):.6f}")
                print(f"    Avg Validation MAE: {params_info.get('avg_val_mae', np.nan):.6f}")
                print(f"    Avg Validation R²: {params_info.get('avg_val_r2', np.nan):.6f}")
            else:
                print("    No optimal parameters found or tuning failed.")
        
        print("\n--- 10B.1.6 Saving MF Tuning Results ---")
        save_mf_tuning_results(mf_tuning_results, tuning_results_excel_output)
        save_mf_optimal_parameters(mf_best_params, optimal_params_json_output)
        create_mf_tuning_visualization(mf_tuning_results, mf_best_params, visualization_pdf_output)
        # save_mf_sample_predictions(...) # TODO_MULTIFACTOR
        
        elapsed_time = time.time() - start_time
        print(f"\nStep 10B.1 (MF) completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Outputs saved in: {output_dir_mf}")
        print("="*80)
        
    except Exception as e:
        print(f"Error in Step 10B.1 (MF): {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
