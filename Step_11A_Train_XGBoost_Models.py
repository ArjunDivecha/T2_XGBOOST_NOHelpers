'''
Step_11A_Train_XGBoost_Models.py

This script trains XGBoost models for each factor and each window using the optimal
hyperparameters identified in Step_10A. It iterates through all relevant windows
and factors, trains a model with early stopping, and saves each trained model.

----------------------------------------------------------------------------------------------------
INPUT FILES:
- S4_Window_Schedule.xlsx
  - Path: ./output/S4_Window_Schedule.xlsx
  - Description: Contains the schedule of all rolling windows.

- S8_Feature_Sets.h5
  - Path: ./output/S8_Feature_Sets.h5
  - Description: Feature sets for each factor and window.

- S10A_XGBoost_Optimal_Params.json
  - Path: ./output/S10A_XGBoost_Optimal_Params.json
  - Description: Optimal XGBoost parameters identified from tuning.

OUTPUT FILES:
- Directory: ./output/S11A_XGBoost_Models/
  - Description: Contains trained XGBoost models for each factor and window.
  - Format: Models saved as .joblib files (e.g., window_1/factor_X.joblib)

- S11A_Training_Log.log
  - Path: ./output/S11A_Training_Log.log
  - Description: Log file for the training process.
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import warnings
import os
import sys
import json
import time
import h5py
import joblib
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import re

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure XGBoost to be less verbose
os.environ['XGBOOST_VERBOSE'] = '0'

# Section: Define Constants
RANDOM_SEED = 42
VERBOSE = True
OUTPUT_DIR = Path("output")
MODELS_SUBDIR = OUTPUT_DIR / "S11A_XGBoost_Models"
LOG_FILE = OUTPUT_DIR / "S11A_Training_Log.log"

# Determine number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() > 2 else 1 # Leave cores free

# Ensure deterministic behavior for aspects not controlled by XGBoost's seed
np.random.seed(RANDOM_SEED)

# Section: Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # Overwrite log file each run
        logging.StreamHandler() # Also print to console
    ]
)

# Section: Utility Functions

def load_optimal_params(output_dir_path):
    """Load optimal XGBoost parameters from the JSON file."""
    params_path = output_dir_path / "S10A_XGBoost_Optimal_Params.json"
    try:
        with open(params_path, 'r') as f:
            optimal_params = json.load(f)
        logging.info(f"Successfully loaded optimal parameters from {params_path}")
        # Ensure 'random_state' or 'seed' is set for XGBoost if not already
        if 'seed' not in optimal_params and 'random_state' not in optimal_params:
            optimal_params['seed'] = RANDOM_SEED
            logging.info(f"Added default seed {RANDOM_SEED} to XGBoost parameters.")
        elif 'random_state' in optimal_params and 'seed' not in optimal_params:
             optimal_params['seed'] = optimal_params['random_state'] # Ensure 'seed' is used, as XGBoost expects
        return optimal_params
    except FileNotFoundError:
        logging.error(f"Optimal parameters file not found: {params_path}")
        raise

def load_window_schedule(output_dir_path):
    """Load the window schedule."""
    schedule_path = output_dir_path / "S4_Window_Schedule.xlsx"
    try:
        schedule = pd.read_excel(schedule_path)
        logging.info(f"Successfully loaded window schedule from {schedule_path}")
        return schedule
    except FileNotFoundError:
        logging.error(f"Window schedule file not found: {schedule_path}")
        raise

def get_all_factor_ids(h5_path, first_window_id):
    """Get all factor IDs from the H5 feature set file using the first window."""
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Try to access the first window's training data to list factors
            # Assuming window_id is 1-based from schedule
            window_key = f"window_{first_window_id}"
            if window_key not in h5f['feature_sets']:
                # If first window ID from schedule isn't in H5, try the first available key
                available_window_keys = list(h5f['feature_sets'].keys())
                if not available_window_keys:
                    logging.error("No window data found in H5 file.")
                    return []
                window_key = available_window_keys[0]
                logging.warning(f"Window {first_window_id} not in H5, using first available: {window_key}")

            # Factors are keys within each split (e.g., 'training')
            # Assuming factors are consistent across splits for a given window
            if 'training' not in h5f['feature_sets'][window_key]:
                logging.error(f"'training' split not found in {window_key} to retrieve factor IDs.")
                return []
            all_factor_ids = list(h5f['feature_sets'][window_key]['training'].keys())
            
            # Filter out factors with _3m, _12m, or _60m suffixes
            factor_ids = [f for f in all_factor_ids if not (f.endswith('_3m') or f.endswith('_12m') or f.endswith('_60m'))]
            
            if len(factor_ids) < len(all_factor_ids):
                logging.info(f"Filtered out {len(all_factor_ids) - len(factor_ids)} factors with _3m, _12m, or _60m suffixes. Will only train models for the {len(factor_ids)} 1-month factors.")
            
        logging.info(f"Retrieved {len(factor_ids)} factor IDs from {h5_path} using window {window_key} and split 'training'. Factors: {factor_ids[:5]}...") # Log first 5 factors as sample
        return sorted(factor_ids)
    except Exception as e:
        logging.error(f"Error reading factor IDs from {h5_path}: {e}")
        raise

def load_factor_window_data(h5_path, window_id, factor_id):
    """Load training and validation data for a specific factor and window."""
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Construct the correct base paths
            train_path = f"feature_sets/window_{window_id}/training/{factor_id}"
            val_path = f"feature_sets/window_{window_id}/validation/{factor_id}"

            # Load training data
            X_train = h5f[f"{train_path}/X/data"][:]
            y_train = h5f[f"{train_path}/y"][:]
            
            # Load validation data
            X_val = h5f[f"{val_path}/X/data"][:]
            y_val = h5f[f"{val_path}/y"][:]
            
            # Reshape y if it's 1D for XGBoost
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)


            # Log shapes for debugging
            logging.debug(f"Loaded data for window {window_id}, factor {factor_id}:")
            logging.debug(f"  X_train shape: {X_train.shape}")
            logging.debug(f"  y_train shape: {y_train.shape}")
            logging.debug(f"  X_val shape: {X_val.shape}")
            logging.debug(f"  y_val shape: {y_val.shape}")

            return X_train, y_train, X_val, y_val

    except KeyError as e:
        missing_key_info = str(e)
        example_path_structure = f"feature_sets/window_<window_id>/[training|validation]/{factor_id}/[X/data|y]"
        logging.warning(f"Data not found for window {window_id}, factor {factor_id}. Missing key/path segment: {missing_key_info} (expected structure like: {example_path_structure}).")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Error loading data for window {window_id}, factor {factor_id}: {e}", exc_info=True)
        return None, None, None, None

def train_and_save_model_task(args):
    """Task function for parallel processing: train and save one model."""
    window_id, factor_id, optimal_params, h5_path, models_subdir_path = args

    model_dir = models_subdir_path / f"window_{window_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base factor name by removing time period suffixes (e.g., _3m, _60m)
    # First check for exact match against patterns like _1m, _3m, _12m, _60m
    time_period_pattern = r'_(?:1m|3m|12m|60m)$'
    base_factor_id = re.sub(time_period_pattern, '', factor_id)
    
    # If we still have _CS or _TS at the end, keep those as they are different factors
    # We're only removing the time period suffixes
    
    # Use the base factor name for the model file
    model_path = model_dir / f"factor_{base_factor_id}.joblib"

    # Skip if model already exists (useful for re-runs)
    if model_path.exists():
        # This local log won't be visible in main process unless returned or managed via multiprocessing logger
        # logging.info(f"Model already exists for window {window_id}, factor {factor_id}. Skipping.")
        return f"Skipped (exists): Window {window_id}, Factor {factor_id} (saved as {base_factor_id})"

    X_train, y_train, X_val, y_val = load_factor_window_data(h5_path, window_id, factor_id)

    if X_train is None or X_val is None or y_train is None or y_val is None: # Check if data loading failed
        return f"Failed (no data): Window {window_id}, Factor {factor_id}"
    
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        # logging.warning(f"No training or validation data for window {window_id}, factor {factor_id}. Skipping model training.")
        return f"Skipped (empty data): Window {window_id}, Factor {factor_id}"

    try:
        # Ensure 'eval_metric' is appropriate and use 'early_stopping_rounds'
        xgb_model = xgb.XGBRegressor(**optimal_params)
        eval_set = [(X_val, y_val)]
        
        # Make sure 'early_stopping_rounds' is in optimal_params or set a default
        early_stopping_rounds = optimal_params.get('early_stopping_rounds', 10)
        if 'early_stopping_rounds' in optimal_params: # XGBoost uses this param name directly
            pass # It's already in optimal_params
        else: # Add it if not present
            optimal_params_copy = optimal_params.copy()
            optimal_params_copy['early_stopping_rounds'] = early_stopping_rounds
            xgb_model = xgb.XGBRegressor(**optimal_params_copy)

        xgb_model.fit(X_train, y_train, 
                        eval_set=eval_set, 
                        verbose=False) # XGBoost internal verbosity

        joblib.dump(xgb_model, model_path)
        return f"Success: Window {window_id}, Factor {factor_id} (saved as {base_factor_id})"
    except Exception as e:
        # logging.error(f"Error training model for window {window_id}, factor {factor_id}: {e}")
        return f"Failed (training error {e}): Window {window_id}, Factor {factor_id}"

def main():
    """Main function to orchestrate model training."""
    logging.info("--- Step 11A: Train XGBoost Models --- Starting --- ")
    start_time_main = time.time()

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_SUBDIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    optimal_params = load_optimal_params(OUTPUT_DIR)
    window_schedule = load_window_schedule(OUTPUT_DIR)
    h5_features_path = OUTPUT_DIR / "S8_Feature_Sets.h5"

    if not h5_features_path.exists():
        logging.error(f"Features H5 file not found: {h5_features_path}")
        return

    # Get list of all factors and windows to process
    # Assuming 'Window_ID' is the column in the schedule
    all_window_ids = window_schedule['Window_ID'].unique().tolist()
    if not all_window_ids:
        logging.error("No window IDs found in schedule.")
        return
        
    all_factor_ids = get_all_factor_ids(h5_features_path, all_window_ids[0])
    if not all_factor_ids:
        logging.error("No factor IDs could be retrieved.")
        return

    # Log detailed information about windows and factors
    total_tasks = len(all_window_ids) * len(all_factor_ids)
    logging.info(f"Found {len(all_window_ids)} windows in schedule.")
    logging.info(f"Found {len(all_factor_ids)} factors in the feature sets.")
    logging.info(f"Total models to process: {total_tasks}")
    
    # Log a sample of window IDs and factor IDs for verification
    logging.info(f"Sample window IDs (first 5): {all_window_ids[:5]}")
    logging.info(f"Sample factor IDs (first 5): {all_factor_ids[:5]}")
    
    # Create tasks for parallel processing
    tasks = [(window_id, factor_id, optimal_params, h5_features_path, MODELS_SUBDIR) 
             for window_id in all_window_ids 
             for factor_id in all_factor_ids]
    
    logging.info(f"Created {len(tasks)} tasks for parallel processing")

    logging.info(f"Created {len(tasks)} tasks for parallel execution using {NUM_CORES} cores.")

    # Configure logging to only show errors
    logging.getLogger().setLevel(logging.ERROR)
    
    # Calculate chunksize for better progress tracking
    chunksize = max(1, len(tasks) // (NUM_CORES * 4))
    
    # Parallel execution with better progress tracking
    pool = None
    try:
        pool = multiprocessing.Pool(processes=NUM_CORES)
        results = []
        
        # Use tqdm for progress bar with better formatting
        with tqdm(total=len(tasks), desc="Training Models", unit="model", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
            
            # Process in chunks to update progress more smoothly
            for i, result in enumerate(pool.imap_unordered(train_and_save_model_task, tasks, chunksize=chunksize)):
                results.append(result)
                pbar.update(1)
                
                # Only log errors, not warnings
                if "Failed" in result and VERBOSE:
                    logging.error(result)
                    
                # Update progress description with success rate
                if i % 100 == 0 and i > 0:
                    success_rate = sum(1 for r in results if "Success" in r) / len(results) * 100
                    pbar.set_postfix(success=f"{success_rate:.1f}%")
        
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        if pool:
            pool.terminate()
        raise
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")
        if pool:
            pool.terminate()
        raise

    except Exception as e:
        logging.error(f"An error occurred during parallel processing: {e}")
    finally:
        if pool:
            pool.close()
            pool.join()
    
    logging.info("--- Parallel processing completed ---")

    # Summarize results
    success_count = sum(1 for r in results if "Success" in r)
    skipped_exists_count = sum(1 for r in results if "Skipped (exists)" in r)
    skipped_empty_data_count = sum(1 for r in results if "Skipped (empty data)" in r)
    failed_no_data_count = sum(1 for r in results if "Failed (no data)" in r)
    failed_training_count = sum(1 for r in results if "Failed (training error" in r)
    
    logging.info(f"Training Summary:")
    logging.info(f"  Total tasks: {len(tasks)}")
    logging.info(f"  Successfully trained models: {success_count}")
    logging.info(f"  Skipped (already existed): {skipped_exists_count}")
    logging.info(f"  Skipped (empty data): {skipped_empty_data_count}")
    logging.info(f"  Failed (no data found): {failed_no_data_count}")
    logging.info(f"  Failed (training error): {failed_training_count}")

    end_time_main = time.time()
    logging.info(f"--- Step 11A: Train XGBoost Models --- Completed in {end_time_main - start_time_main:.2f} seconds --- ")

if __name__ == "__main__":
    # It's good practice to also set seed for the main process if using multiprocessing
    # Though XGBoost's internal seed should handle its own randomness primarily.
    multiprocessing.freeze_support() # For Windows compatibility, good to have
    main()
