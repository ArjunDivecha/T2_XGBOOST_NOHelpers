# File Documentation
'''
----------------------------------------------------------------------------------------------------
MODIFIED FOR MULTIFACTOR REGRESSION (ALL FACTORS PREDICTED SIMULTANEOUSLY)
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S8_1_MF_Feature_Sets.h5
  - Path: ./output/S8_1_MF_Feature_Sets.h5 (output from Step 8.1 MF)
  - Description: HDF5 file with WIDE X (features) and MULTI-TARGET Y (returns) for each window.

- S4_Window_Schedule.xlsx
  - Path: ../output/S4_Window_Schedule.xlsx (output from Step 4 in parent directory)
  - Description: Excel file with the schedule of all rolling windows.

- S10B_1_MF_Linear_Models_Optimal_Params.json
  - Path: ./output/S10B_1_MF_Linear_Models_Optimal_Params.json (output from Step 10B.1 MF)
  - Description: JSON file with ONE set of optimal hyperparameters for each linear model type
                 (e.g., OLS, Ridge, Lasso, NNLS), best performing on average across all target factors.

OUTPUT FILES:
- S11B_1_MF_Linear_Models.h5
  - Path: ./output/S11B_1_MF_Linear_Models.h5
  - Description: HDF5 file storing the TRAINED MULTI-OUTPUT linear models for each window.
                 For each window and model type (e.g., Ridge), it stores one model trained
                 on all features to predict all target factor returns simultaneously.
                 Coefficients will be (n_targets, n_features), intercepts (n_targets,).

- S11B_1_MF_Linear_Models_Training_Summary.xlsx
  - Path: ./output/S11B_1_MF_Linear_Models_Training_Summary.xlsx
  - Description: Excel file summarizing training/validation performance (e.g., avg. RMSE, R2)
                 of each multi-output model per window.

# TODO_MULTIFACTOR: Visualizations (PDF) require significant adaptation for multi-output models.
# These are simplified or placeholders in this version.

----------------------------------------------------------------------------------------------------
Purpose:
This script trains multi-output linear regression models for ALL factors simultaneously across all
relevant windows. It uses the optimal hyperparameters found in Step 10B.1 (MF).
Models trained:
1. Ordinary Least Squares (OLS)
2. Ridge Regression
3. LASSO Regression
4. Non-Negative Least Squares (NNLS) - using LinearRegression(positive=True)

For each window, ONE model of each type is trained using the wide feature matrix (X)
and multi-target returns matrix (Y). Trained models and performance metrics are saved.

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
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Define root project output directory to load S4_Window_Schedule.xlsx
project_root_output_parent = Path("../output")

# Output directory for this Multifactor script's results (local to Multifactor/)
output_dir_mf = Path("output")
output_dir_mf.mkdir(parents=True, exist_ok=True)

# Section: Configuration & Constants
RANDOM_SEED = 42
NUM_PROCESSES = min(mp.cpu_count(), 8)  # Adjust based on system resources for window-level parallelization
VERBOSE = True

# Ensure deterministic behavior
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Section: Utility Functions
def load_mf_train_data():
    """
    Load window schedule, MULTIFACTOR feature sets (S8_1_MF), and MULTIFACTOR optimal params (S10B_1_MF).
    """
    if VERBOSE: print("Loading multifactor training data and optimal parameters...")

    window_schedule_file = project_root_output_parent / "S4_Window_Schedule.xlsx"
    feature_sets_h5_file = output_dir_mf / "S8_1_MF_Feature_Sets.h5" # From Step 8.1 MF
    optimal_params_json_file = output_dir_mf / "S10B_1_MF_Linear_Models_Optimal_Params.json" # From Step 10B.1 MF

    for f_path in [window_schedule_file, feature_sets_h5_file, optimal_params_json_file]:
        if not f_path.exists():
            raise FileNotFoundError(f"Required input file not found: {f_path}")

    if VERBOSE: print(f"  Loading window schedule from: {window_schedule_file}")
    window_schedule = pd.read_excel(window_schedule_file)

    if VERBOSE: print(f"  Loading MF optimal parameters from: {optimal_params_json_file}")
    with open(optimal_params_json_file, 'r') as f:
        optimal_params_mf = json.load(f)
        # Post-process params if needed (e.g., string 'True' to boolean True - json should handle bools)

    if VERBOSE: print(f"  Loading MF feature sets from: {feature_sets_h5_file}")
    feature_sets = {}
    with h5py.File(feature_sets_h5_file, 'r') as hf:
        # Store column names from HDF5 attributes if needed for consistency
        x_colnames = [col.decode('utf-8') for col in hf.attrs.get('x_feature_names_ordered', [])]
        y_colnames = [col.decode('utf-8') for col in hf.attrs.get('y_target_names_ordered', [])]
        
        feature_sets['x_colnames'] = x_colnames
        feature_sets['y_colnames'] = y_colnames

        for window_id_str in tqdm(hf.keys(), desc="Loading H5 Window Data for Training"):
            if not window_id_str.isdigit(): continue # Skip attribute keys like 'x_colnames'
            window_id = int(window_id_str)
            feature_sets[window_id] = {}
            for period in ['training', 'validation', 'prediction']:
                try:
                    X_data = hf[window_id_str][f'{period}_X'][:]
                    Y_data = hf[window_id_str][f'{period}_Y'][:]
                    # Use stored column names if available and match dimensions
                    current_x_cols = x_colnames if len(x_colnames) == X_data.shape[1] else [f'X{i}' for i in range(X_data.shape[1])]
                    current_y_cols = y_colnames if len(y_colnames) == Y_data.shape[1] else [f'Y{i}' for i in range(Y_data.shape[1])]

                    feature_sets[window_id][f'X_{period}'] = pd.DataFrame(X_data, columns=current_x_cols)
                    feature_sets[window_id][f'Y_{period}'] = pd.DataFrame(Y_data, columns=current_y_cols)
                except KeyError:
                    feature_sets[window_id][f'X_{period}'] = pd.DataFrame()
                    feature_sets[window_id][f'Y_{period}'] = pd.DataFrame()
    
    if VERBOSE: print(f"Loaded data for {len(window_schedule)} windows configurations.")
    return window_schedule, feature_sets, optimal_params_mf

def get_model_hyperparams(model_type, optimal_params_mf):
    """ Get the single set of optimal hyperparameters for a given model_type from the MF tuning results. """
    if model_type in optimal_params_mf and optimal_params_mf[model_type].get('params'):
        return optimal_params_mf[model_type]['params']
    if VERBOSE: print(f"  Warning: No optimal params found for {model_type} in optimal_params_mf. Using default.")
    return {} # Default if not found

# Section: Model Training and Evaluation Core Functions
def create_mf_model(model_type, params):
    """ Creates a scikit-learn model instance for multi-output regression. """
    actual_params = {k: v for k, v in params.items()} # Make a copy
    if model_type == 'OLS':
        return LinearRegression(**{p: actual_params[p] for p in actual_params if p in LinearRegression().get_params()})
    elif model_type == 'Ridge':
        return Ridge(**{p: actual_params[p] for p in actual_params if p in Ridge().get_params()}, random_state=RANDOM_SEED)
    elif model_type == 'Lasso':
        return Lasso(**{p: actual_params[p] for p in actual_params if p in Lasso().get_params()}, random_state=RANDOM_SEED)
    elif model_type == 'NNLS':
        # fit_intercept is the main param from our grid for NNLS via LinearRegression(positive=True)
        return LinearRegression(positive=True, fit_intercept=actual_params.get('fit_intercept', True))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_mf_model(model, X_train_np, Y_train_np):
    """ Trains the multi-output model. """
    model.fit(X_train_np, Y_train_np)
    return model

def evaluate_mf_model(model, X_np, Y_np, data_type='validation'):
    """ Evaluates the trained multi-output model. Returns dict of averaged metrics. """
    if X_np.shape[0] == 0: # No data to evaluate
        return {f'{data_type}_rmse_avg': np.nan, f'{data_type}_mae_avg': np.nan, f'{data_type}_r2_avg': np.nan, f'{data_type}_metrics_per_target': {}}

    Y_pred_np = model.predict(X_np)
    
    # Averaged metrics across all targets
    rmse_avg = mean_squared_error(Y_np, Y_pred_np, squared=False, multioutput='uniform_average')
    mae_avg = mean_absolute_error(Y_np, Y_pred_np, multioutput='uniform_average')
    r2_avg = r2_score(Y_np, Y_pred_np, multioutput='uniform_average')
    
    # Per-target metrics (optional, can be large)
    # rmse_per_target = mean_squared_error(Y_np, Y_pred_np, squared=False, multioutput='raw_values')
    # mae_per_target = mean_absolute_error(Y_np, Y_pred_np, multioutput='raw_values')
    # r2_per_target = r2_score(Y_np, Y_pred_np, multioutput='raw_values')
    # metrics_per_target = {'rmse': rmse_per_target.tolist(), 'mae': mae_per_target.tolist(), 'r2': r2_per_target.tolist()}

    return {
        f'{data_type}_rmse_avg': rmse_avg,
        f'{data_type}_mae_avg': mae_avg,
        f'{data_type}_r2_avg': r2_avg,
        # f'{data_type}_metrics_per_target': metrics_per_target # Can be enabled if detailed per-target needed
    }

def train_and_evaluate_window_mf(args):
    """ 
    Worker function for parallel processing. Trains all model types for one window.
    Args: (window_id, window_data_dict from feature_sets, optimal_params_mf_for_all_models)
    """
    window_id, window_feature_data, optimal_params_mf = args
    if VERBOSE: print(f"  Processing window {window_id}...")
    
    window_results = {'window_id': window_id, 'models_data': {}}

    X_train_df = window_feature_data.get('X_training', pd.DataFrame())
    Y_train_df = window_feature_data.get('Y_training', pd.DataFrame())
    X_val_df = window_feature_data.get('X_validation', pd.DataFrame())
    Y_val_df = window_feature_data.get('Y_validation', pd.DataFrame())

    # NaN Handling: Drop rows where *any* target in Y_train or Y_val is NaN.
    # This ensures all targets are valid for samples used in training/validation.
    train_valid_indices = Y_train_df.dropna().index
    Y_train_clean = Y_train_df.loc[train_valid_indices]
    X_train_clean = X_train_df.loc[train_valid_indices]

    val_valid_indices = Y_val_df.dropna().index
    Y_val_clean = Y_val_df.loc[val_valid_indices]
    X_val_clean = X_val_df.loc[val_valid_indices]

    if X_train_clean.empty or Y_train_clean.empty:
        if VERBOSE: print(f"    Window {window_id}: Training data empty after NaN drop. Skipping.")
        return window_id, window_results # Return empty results for this window

    model_types_to_train = ['OLS', 'Ridge', 'Lasso', 'NNLS']
    for model_type in model_types_to_train:
        if VERBOSE > 1: print(f"    Window {window_id}: Training {model_type}...")
        model_hparams = get_model_hyperparams(model_type, optimal_params_mf)
        
        model_instance = create_mf_model(model_type, model_hparams)
        trained_model = train_mf_model(model_instance, X_train_clean.values, Y_train_clean.values)
        
        train_metrics = evaluate_mf_model(trained_model, X_train_clean.values, Y_train_clean.values, 'train')
        val_metrics = evaluate_mf_model(trained_model, X_val_clean.values, Y_val_clean.values, 'validation')
        
        # Store model coefficients and intercept
        # Coef shape: (n_targets, n_features), Intercept shape: (n_targets,)
        # For OLS, Ridge, Lasso: .coef_, .intercept_
        # For NNLS (LinearRegression positive=True): .coef_, .intercept_
        coeffs = trained_model.coef_ if hasattr(trained_model, 'coef_') else None
        intercept = trained_model.intercept_ if hasattr(trained_model, 'intercept_') else None
        
        window_results['models_data'][model_type] = {
            'hyperparameters': model_hparams,
            'coefficients': coeffs.tolist() if coeffs is not None else None, # Convert to list for saving
            'intercept': intercept.tolist() if intercept is not None else None,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'num_train_samples': len(X_train_clean),
            'num_val_samples': len(X_val_clean),
            'num_features': X_train_clean.shape[1],
            'num_targets': Y_train_clean.shape[1]
        }
    return window_id, window_results

# Section: Saving Results
def save_trained_mf_models_to_h5(all_windows_results, output_h5_file, feature_sets_metadata):
    if VERBOSE: print(f"Saving trained MF models to {output_h5_file}...")
    with h5py.File(output_h5_file, 'w') as hf:
        # Save global metadata like feature names if available
        if 'x_colnames' in feature_sets_metadata:
             hf.attrs['x_feature_names_ordered'] = [col.encode('utf-8') for col in feature_sets_metadata['x_colnames']]
        if 'y_colnames' in feature_sets_metadata:
             hf.attrs['y_target_names_ordered'] = [col.encode('utf-8') for col in feature_sets_metadata['y_colnames']]

        for window_id, window_data in tqdm(all_windows_results.items(), desc="Saving Models to HDF5"):
            if not window_data or not window_data.get('models_data'): continue # Skip if window processing failed
            win_group = hf.create_group(str(window_id))
            for model_type, model_info in window_data['models_data'].items():
                model_group = win_group.create_group(model_type)
                model_group.attrs['hyperparameters'] = json.dumps(model_info['hyperparameters'])
                if model_info['coefficients'] is not None:
                    model_group.create_dataset('coefficients', data=np.array(model_info['coefficients']))
                if model_info['intercept'] is not None:
                    model_group.create_dataset('intercept', data=np.array(model_info['intercept']))
                
                for metric_type in ['train_metrics', 'validation_metrics']:
                    metrics_group = model_group.create_group(metric_type)
                    for m_name, m_val in model_info[metric_type].items():
                        if isinstance(m_val, dict): # e.g. per_target metrics
                            # metrics_group.attrs[m_name] = json.dumps(m_val)
                            pass # Not saving per-target to keep H5 simpler for now
                        elif pd.notna(m_val):
                            metrics_group.attrs[m_name] = m_val
                model_group.attrs['num_train_samples'] = model_info['num_train_samples']
                model_group.attrs['num_val_samples'] = model_info['num_val_samples']
                model_group.attrs['num_features'] = model_info['num_features']
                model_group.attrs['num_targets'] = model_info['num_targets']
    if VERBOSE: print("  Trained MF models saved.")

def create_mf_training_summary_excel(all_windows_results, output_excel_file):
    if VERBOSE: print(f"Creating MF training summary Excel: {output_excel_file}...")
    summary_data = []
    for window_id, window_data in all_windows_results.items():
        if not window_data or not window_data.get('models_data'): continue
        for model_type, model_info in window_data['models_data'].items():
            row = {'window_id': window_id, 'model_type': model_type}
            row.update(model_info.get('hyperparameters', {}))
            row.update({f"train_{k.replace('train_', '')}": v for k,v in model_info.get('train_metrics', {}).items() if not isinstance(v,dict)})
            row.update({f"val_{k.replace('validation_', '')}": v for k,v in model_info.get('validation_metrics', {}).items() if not isinstance(v,dict)})
            row['num_train_samples'] = model_info.get('num_train_samples')
            row['num_val_samples'] = model_info.get('num_val_samples')
            summary_data.append(row)
    
    if not summary_data: 
        if VERBOSE: print("  No summary data to write to Excel."); return
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(output_excel_file, index=False, sheet_name='MF_Training_Summary')
    if VERBOSE: print("  MF Training summary Excel saved.")

# TODO_MULTIFACTOR: create_mf_visualizations needs careful design for multi-output models.

# Section: Main Execution
def main():
    print("\n" + "="*80)
    print("STEP 11B.1 (Multifactor): TRAIN MULTI-OUTPUT LINEAR MODELS")
    print("="*80)
    start_time = time.time()

    # Output file paths (local to Multifactor/output/)
    trained_models_h5_output = output_dir_mf / "S11B_1_MF_Linear_Models.h5"
    training_summary_excel_output = output_dir_mf / "S11B_1_MF_Linear_Models_Training_Summary.xlsx"
    # visualization_pdf_output = output_dir_mf / "S11B_1_MF_Linear_Models_Training_Visualization.pdf" # Deferred

    try:
        print("\n--- 11B.1.1 Loading Data & Optimal Parameters ---")
        window_schedule, feature_sets, optimal_params_mf = load_mf_train_data()
        
        window_ids_to_process = sorted([k for k in feature_sets.keys() if isinstance(k, int)]) # Get actual window IDs
        if not window_ids_to_process:
            print("Error: No window data loaded from feature sets. Exiting."); return

        print(f"\n--- 11B.1.2 Training Multi-Output Models for {len(window_ids_to_process)} Windows ---")
        
        all_windows_results_dict = {}
        # Prepare arguments for parallel processing
        args_for_mp = []
        for win_id in window_ids_to_process:
            if win_id in feature_sets and feature_sets[win_id].get('X_training') is not None:
                 args_for_mp.append((win_id, feature_sets[win_id], optimal_params_mf))
            else:
                if VERBOSE: print(f"  Skipping window {win_id} from parallel processing due to missing data in feature_sets dict.")
        
        if not args_for_mp: print("No valid windows with data to process after filtering. Exiting."); return

        if NUM_PROCESSES > 1 and len(args_for_mp) > 1:
            if VERBOSE: print(f"  Using {NUM_PROCESSES} processes for parallel window training...")
            with mp.Pool(processes=NUM_PROCESSES) as pool:
                # results is a list of tuples: [(window_id, window_result_dict), ...]
                mp_results = list(tqdm(pool.imap(train_and_evaluate_window_mf, args_for_mp), total=len(args_for_mp), desc="Training MF Models"))
            for win_id_res, win_data_res in mp_results:
                all_windows_results_dict[win_id_res] = win_data_res
        else:
            if VERBOSE: print("  Processing windows sequentially...")
            for arg_set in tqdm(args_for_mp, desc="Training MF Models (Sequential)"):
                win_id_res_s, win_data_res_s = train_and_evaluate_window_mf(arg_set)
                all_windows_results_dict[win_id_res_s] = win_data_res_s

        print("\n--- 11B.1.3 Saving Trained Models and Summaries ---")
        # Pass general feature_sets for metadata like column names
        save_trained_mf_models_to_h5(all_windows_results_dict, trained_models_h5_output, feature_sets)
        create_mf_training_summary_excel(all_windows_results_dict, training_summary_excel_output)
        # create_mf_visualizations(...) # TODO_MULTIFACTOR

        elapsed_time = time.time() - start_time
        print(f"\nStep 11B.1 (MF) completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Outputs saved in: {output_dir_mf}")
        print("="*80)

    except Exception as e:
        print(f"Error in Step 11B.1 (MF): {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
