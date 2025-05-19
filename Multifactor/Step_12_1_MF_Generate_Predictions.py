# File Documentation
'''
----------------------------------------------------------------------------------------------------
MULTIFACTOR - STEP 12.1: GENERATE PREDICTIONS & FORMAT OUTPUTS
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S11B_1_MF_Linear_Models.h5
  - Path: ./output/S11B_1_MF_Linear_Models.h5 (Output from Step 11B.1 MF)
  - Description: HDF5 file containing trained MULTI-OUTPUT linear models for each window.

- S8_1_MF_Feature_Sets.h5
  - Path: ./output/S8_1_MF_Feature_Sets.h5 (Output from Step 8.1 MF)
  - Description: HDF5 file with WIDE X_prediction, MULTI-TARGET Y_prediction (actuals),
                 and importantly, 'prediction_dates_df' which links features/targets
                 to specific dates and factor IDs for the prediction period of each window.

- S4_Window_Schedule.xlsx
  - Path: ../output/S4_Window_Schedule.xlsx (Output from Step 4 in parent directory)
  - Description: Provides start/end dates for prediction periods of each window, crucial for
                 selecting the latest available prediction for a given month.

OUTPUT FILES (all in ./output/):
1. S12_1_MF_Monthly_Predictions.xlsx
   - Description: Excel file with monthly predicted returns for each factor.
                  Structure: Dates (first of month) as rows, Factor IDs as columns.
                  Separate sheets for each model type (OLS, Ridge, Lasso, NNLS).
                  Uses the prediction from the LATEST window covering each month-factor.

2. S12_1_MF_Monthly_Actual_Returns.xlsx
   - Description: Excel file with actual monthly returns for each factor.
                  Format identical to S12_1_MF_Monthly_Predictions.xlsx.

3. S12_1_MF_Prediction_Performance.xlsx
   - Description: Excel file summarizing out-of-sample prediction performance metrics
                  (e.g., avg. R2, RMSE, MAE across all targets) for each model type per window.

----------------------------------------------------------------------------------------------------
Purpose:
This script uses the trained multi-factor linear models to generate predictions on the
out-of-sample prediction sets. It then formats these predictions and actual returns into
monthly tables similar to T2_Optimizer.xlsx, and calculates prediction performance metrics.

Version: 1.0 (Multifactor)
Last Updated: 2025-05-16
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import h5py
import json
from pathlib import Path
import time
from tqdm import tqdm
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Define root project output directory (for S4_Window_Schedule.xlsx)
project_root_output_parent = Path("../output")

# Output directory for this Multifactor script's results (local to Multifactor/)
output_dir_mf = Path("output")
output_dir_mf.mkdir(parents=True, exist_ok=True)

VERBOSE = True
MODEL_TYPES = ['OLS', 'Ridge', 'Lasso', 'NNLS'] # Models to generate predictions for

# Section: Utility Functions
def load_prediction_data():
    """Loads trained models, feature sets (prediction part), and window schedule."""
    if VERBOSE: print("Loading data for prediction generation...")

    trained_models_h5_file = output_dir_mf / "S11B_1_MF_Linear_Models.h5"
    feature_sets_h5_file = output_dir_mf / "S8_1_MF_Feature_Sets.h5"
    window_schedule_file = project_root_output_parent / "S4_Window_Schedule.xlsx"

    for f_path in [trained_models_h5_file, feature_sets_h5_file, window_schedule_file]:
        if not f_path.exists():
            raise FileNotFoundError(f"Required input file not found: {f_path}")

    trained_models = {}
    if VERBOSE: print(f"  Loading trained models from: {trained_models_h5_file}")
    with h5py.File(trained_models_h5_file, 'r') as hf:
        # Extract target factor names if stored (should match Y_colnames from S8)
        target_factor_names = [col.decode('utf-8') for col in hf.attrs.get('y_target_names_ordered', [])]
        trained_models['target_factor_names'] = target_factor_names

        for window_id_str in tqdm(hf.keys(), desc="Loading Trained Models"):
            if not window_id_str.isdigit(): continue # Skip attributes
            trained_models[int(window_id_str)] = {}
            for model_type_key in hf[window_id_str].keys():
                model_group = hf[window_id_str][model_type_key]
                trained_models[int(window_id_str)][model_type_key] = {
                    'coefficients': model_group['coefficients'][:] if 'coefficients' in model_group else None,
                    'intercept': model_group['intercept'][:] if 'intercept' in model_group else None,
                    'hyperparameters': json.loads(model_group.attrs.get('hyperparameters', '{}'))
                }
    
    prediction_data = {'feature_sets': {}, 'prediction_dates_df': pd.DataFrame()}
    if VERBOSE: print(f"  Loading prediction feature sets from: {feature_sets_h5_file}")
    with h5py.File(feature_sets_h5_file, 'r') as hf:
        # Store column names from HDF5 attributes if needed for consistency
        x_colnames = [col.decode('utf-8') for col in hf.attrs.get('x_feature_names_ordered', [])]
        y_colnames = [col.decode('utf-8') for col in hf.attrs.get('y_target_names_ordered', [])]
        
        prediction_data['x_colnames'] = x_colnames # For X_prediction
        prediction_data['y_colnames'] = y_colnames # For Y_prediction and target factor names

        if not trained_models['target_factor_names'] and y_colnames:
            trained_models['target_factor_names'] = y_colnames # Fallback if not in models H5

        all_pred_dates_dfs = []
        for window_id_str in tqdm(hf.keys(), desc="Loading Prediction Features/Actuals"):
            if not window_id_str.isdigit(): continue # Skip attributes
            window_id = int(window_id_str)
            prediction_data['feature_sets'][window_id] = {}
            try:
                X_pred = hf[window_id_str]['prediction_X'][:]
                Y_pred_actual = hf[window_id_str]['prediction_Y'][:]
                pred_dates_df_bytes = hf[window_id_str]['prediction_dates_df'][:]
                pred_dates_df_str = pred_dates_df_bytes.astype(str)
                current_pred_dates = pd.read_json(pred_dates_df_str)
                current_pred_dates['Window_ID'] = window_id # Add window ID for later merging

                current_x_cols = x_colnames if len(x_colnames) == X_pred.shape[1] else [f'X_pred_{i}' for i in range(X_pred.shape[1])]
                current_y_cols = y_colnames if len(y_colnames) == Y_pred_actual.shape[1] else [f'Y_pred_{i}' for i in range(Y_pred_actual.shape[1])]

                prediction_data['feature_sets'][window_id]['X_prediction'] = pd.DataFrame(X_pred, columns=current_x_cols)
                prediction_data['feature_sets'][window_id]['Y_prediction_actual'] = pd.DataFrame(Y_pred_actual, columns=current_y_cols)
                all_pred_dates_dfs.append(current_pred_dates)

            except KeyError as e:
                if VERBOSE: print(f"    Warning: Missing prediction data for window {window_id}: {e}")
                prediction_data['feature_sets'][window_id]['X_prediction'] = pd.DataFrame()
                prediction_data['feature_sets'][window_id]['Y_prediction_actual'] = pd.DataFrame()
        
        if all_pred_dates_dfs:
            prediction_data['prediction_dates_df'] = pd.concat(all_pred_dates_dfs, ignore_index=True)
            prediction_data['prediction_dates_df']['Date'] = pd.to_datetime(prediction_data['prediction_dates_df']['Date'])
        else:
            raise ValueError("Critical: 'prediction_dates_df' could not be loaded or is empty.")

    if VERBOSE: print(f"  Loading window schedule from: {window_schedule_file}")
    window_schedule = pd.read_excel(window_schedule_file)
    # Convert date columns to datetime if not already
    for col in ['Train_Start_Date', 'Train_End_Date', 'Validation_End_Date', 'Prediction_End_Date']:
        if col in window_schedule.columns:
            window_schedule[col] = pd.to_datetime(window_schedule[col])

    return trained_models, prediction_data, window_schedule

def generate_predictions_for_window(model_coeffs, model_intercept, X_prediction_df):
    """Generates Y_hat predictions using model coefficients and intercept."""
    if model_coeffs is None or X_prediction_df.empty:
        return pd.DataFrame() # Return empty if no model or no data
    
    Y_hat = X_prediction_df.values @ model_coeffs.T # (n_samples, n_features) @ (n_features, n_targets)
    if model_intercept is not None:
        Y_hat += model_intercept # Add intercept (n_targets,)
    return Y_hat # Returns numpy array

# Section: Main Logic - Processing and Aggregation
def process_all_windows_predictions(trained_models, prediction_data, window_schedule):
    """ 
    Iterates through windows and model types, generates predictions, 
    and collects them along with actuals for structuring.
    """
    if VERBOSE: print("\nProcessing predictions for all windows and model types...")
    
    all_predictions_raw = [] # Store (Window_ID, Date, Factor_ID, Model_Type, Predicted_Return, Actual_Return)
    prediction_performance_data = [] # Store window-level prediction performance

    target_factor_names = trained_models.get('target_factor_names', prediction_data.get('y_colnames'))
    if not target_factor_names:
        raise ValueError("Target factor names could not be determined.")

    # Link prediction_dates_df with window_schedule for Prediction_End_Date for tie-breaking
    pred_dates_master_df = prediction_data['prediction_dates_df'].copy()
    pred_dates_master_df = pd.merge(pred_dates_master_df, window_schedule[['Window_ID', 'Prediction_End_Date']], on='Window_ID', how='left')

    for window_id in tqdm(trained_models.keys(), desc="Generating All Predictions"):
        if not isinstance(window_id, int): continue # Skip metadata keys like 'target_factor_names'
        if window_id not in prediction_data['feature_sets'] or prediction_data['feature_sets'][window_id]['X_prediction'].empty:
            if VERBOSE: print(f"  Skipping window {window_id}: No X_prediction data.")
            continue

        X_pred_df = prediction_data['feature_sets'][window_id]['X_prediction']
        Y_actual_df = prediction_data['feature_sets'][window_id]['Y_prediction_actual'] # (n_samples, n_targets)
        
        # Get the relevant subset of prediction_dates_df for this window's X_prediction samples
        # Assumes X_pred_df and Y_actual_df rows align with entries for this window_id in pred_dates_master_df
        # This needs careful alignment if X_pred_df is not directly indexed as per pred_dates_master_df
        # For now, assume direct row-wise correspondence based on how S8 data is typically structured.
        # A robust way would be to ensure X_pred_df has an index that can map to pred_dates_master_df.
        # We assume the number of rows in X_pred_df matches the number of unique prediction instances for this window.
        
        # Get all prediction instances for this window_id from the master dates df
        window_pred_instances_df = pred_dates_master_df[pred_dates_master_df['Window_ID'] == window_id].copy()
        if len(window_pred_instances_df) != len(X_pred_df):
            # This can happen if prediction_dates_df stores one row per factor-date, 
            # while X_prediction stores one row per unique date (if features are date-common).
            # For multi-target, X_prediction is likely one row per date, Y_prediction has multiple columns (factors).
            # The number of rows in X_pred_df and Y_actual_df should be the same.
            # Let's assume `prediction_dates_df` has one row per (Date, Factor_ID) for this window's scope.
            # We need to get the unique dates for which X_prediction exists.
            unique_pred_dates_for_window = window_pred_instances_df['Date'].unique()
            if len(unique_pred_dates_for_window) != len(X_pred_df):
                 if VERBOSE: print(f"  Warning Window {window_id}: Mismatch between X_prediction rows ({len(X_pred_df)}) and unique prediction dates ({len(unique_pred_dates_for_window)}). Check data structure.")
                 # Continue cautiously, assuming X_pred_df rows correspond to unique dates in order.

        for model_type in MODEL_TYPES:
            if model_type not in trained_models[window_id]:
                if VERBOSE: print(f"  Skipping model {model_type} for window {window_id}: Not found in trained models.")
                continue
            
            model_details = trained_models[window_id][model_type]
            coeffs = model_details['coefficients']
            intercept = model_details['intercept']

            # Y_hat_pred_np shape: (n_prediction_samples_for_window, n_target_factors)
            Y_hat_pred_np = generate_predictions_for_window(coeffs, intercept, X_pred_df)
            if Y_hat_pred_np.shape[0] == 0: continue

            # Performance on this window's prediction set
            # Ensure Y_actual_df columns match target_factor_names order for consistent metrics
            Y_actual_ordered_df = Y_actual_df[target_factor_names] if set(target_factor_names).issubset(Y_actual_df.columns) else Y_actual_df

            avg_rmse = mean_squared_error(Y_actual_ordered_df.values, Y_hat_pred_np, squared=False, multioutput='uniform_average')
            avg_mae = mean_absolute_error(Y_actual_ordered_df.values, Y_hat_pred_np, multioutput='uniform_average')
            avg_r2 = r2_score(Y_actual_ordered_df.values, Y_hat_pred_np, multioutput='uniform_average')
            prediction_performance_data.append({
                'Window_ID': window_id, 'Model_Type': model_type,
                'Prediction_RMSE_avg': avg_rmse, 'Prediction_MAE_avg': avg_mae, 'Prediction_R2_avg': avg_r2
            })

            # Reshape Y_hat_pred_np and Y_actual_df for long format output table
            # Y_hat_pred_df columns should be target_factor_names
            Y_hat_pred_df = pd.DataFrame(Y_hat_pred_np, columns=target_factor_names, index=X_pred_df.index)

            # Iterate through each unique date within this window's prediction span
            # `window_pred_instances_df` contains `Date` and `Factor_ID` mapping
            # For each unique date in X_pred_df.index (assuming index corresponds to unique dates):
            # We need to associate predictions for each factor with the correct Date from `window_pred_instances_df`
            # This part is tricky due to the structure of prediction_dates_df (one row per factor-date)
            # vs. Y_hat_pred_df (one row per date, multiple factor columns)

            # Let's reconstruct based on unique dates in the window's X_prediction
            # And then map to the prediction_dates_df to get the factor IDs for each prediction cell.
            # Each row in Y_hat_pred_df corresponds to a unique prediction date in the window's prediction period.
            # We need to match these dates back to the `window_pred_instances_df` to get the full (Window_ID, Date, Factor_ID, Prediction_End_Date)
            
            # Assume X_pred_df's index maps to the unique dates from `window_pred_instances_df`
            # Get the unique dates for which X_prediction rows were made
            unique_dates_in_X_pred = window_pred_instances_df.loc[X_pred_df.index]['Date'].unique() if X_pred_df.index.equals(window_pred_instances_df.loc[X_pred_df.index].index) else window_pred_instances_df['Date'].unique()[:len(X_pred_df)]
            if len(unique_dates_in_X_pred) != Y_hat_pred_df.shape[0]:
                # Fallback: create a date range if index isn't reliable. This is less robust.
                # This part needs to be extremely robust to match X_pred rows to actual dates
                # For now, let's assume X_pred_df rows correspond one-to-one with the sequence of unique dates
                # in this window's prediction period. This is a strong assumption.
                pass # This indicates a structural assumption that needs validation from S8 output.

            for i, pred_date in enumerate(unique_dates_in_X_pred):
                # Get all factor instances for this specific date and window
                date_specific_instances = window_pred_instances_df[
                    (window_pred_instances_df['Date'] == pred_date) & 
                    (window_pred_instances_df['Window_ID'] == window_id)
                ]
                for factor_idx, factor_name in enumerate(target_factor_names):
                    # Find the specific instance in date_specific_instances that matches factor_name
                    current_factor_instance = date_specific_instances[date_specific_instances['Factor_ID'] == factor_name]
                    if not current_factor_instance.empty:
                        actual_return = Y_actual_ordered_df.iloc[i, factor_idx] # Get actual for this factor-date
                        predicted_return = Y_hat_pred_df.iloc[i, factor_idx]   # Get prediction for this factor-date
                        pred_end_date_for_window = current_factor_instance['Prediction_End_Date'].iloc[0]

                        all_predictions_raw.append({
                            'Window_ID': window_id,
                            'Date': pred_date, # This is the date FOR WHICH the prediction is made (e.g., end of Feb for March return)
                            'Factor_ID': factor_name,
                            'Model_Type': model_type,
                            'Predicted_Return': predicted_return,
                            'Actual_Return': actual_return,
                            'Prediction_End_Date_Window': pred_end_date_for_window # For tie-breaking
                        })
                    else:
                         if VERBOSE: print(f"  Warning: No match found in prediction_dates_df for Date {pred_date}, Factor {factor_name}, Window {window_id}")

    all_predictions_df = pd.DataFrame(all_predictions_raw)
    prediction_performance_df = pd.DataFrame(prediction_performance_data)
    return all_predictions_df, prediction_performance_df

def assemble_monthly_outputs(all_predictions_df):
    """Assembles monthly prediction and actuals tables, handling overlaps."""
    if VERBOSE: print("\nAssembling monthly prediction and actuals tables...")
    if all_predictions_df.empty:
        if VERBOSE: print("  No raw predictions to assemble. Skipping monthly outputs.")
        return {}, pd.DataFrame()

    # Convert 'Date' to datetime and ensure it's the first of the month for pivot
    # The 'Date' from prediction_dates_df is typically the end of the month for which features are available,
    # and the prediction is for the *next* month. T2_Optimizer uses first-of-month for the *return month*.
    # So, if 'Date' is 2000-02-29, it's for March 2000 return. Table should show 2000-03-01.
    all_predictions_df['Return_Month_Start'] = all_predictions_df['Date'] + pd.offsets.MonthBegin(1)
    
    # Sort by Prediction_End_Date_Window (latest window first), then by other fields if needed
    # This helps in selecting the prediction from the latest window
    all_predictions_df.sort_values(by=['Return_Month_Start', 'Factor_ID', 'Model_Type', 'Prediction_End_Date_Window'], 
                                   ascending=[True, True, True, False], inplace=True)

    monthly_predictions_pivots = {}
    for model_type in MODEL_TYPES:
        model_specific_df = all_predictions_df[all_predictions_df['Model_Type'] == model_type]
        # Keep only the first occurrence (latest window's prediction due to sort)
        unique_preds_df = model_specific_df.drop_duplicates(subset=['Return_Month_Start', 'Factor_ID'], keep='first')
        
        if not unique_preds_df.empty:
            pivot_table = unique_preds_df.pivot_table(index='Return_Month_Start', 
                                                      columns='Factor_ID', 
                                                      values='Predicted_Return')
            pivot_table.index.name = 'Date' # Match T2_Optimizer format
            monthly_predictions_pivots[model_type] = pivot_table.copy()
        else:
            monthly_predictions_pivots[model_type] = pd.DataFrame()

    # Assemble actual returns table (should be unique per Return_Month_Start, Factor_ID)
    actuals_df = all_predictions_df.drop_duplicates(subset=['Return_Month_Start', 'Factor_ID'], keep='first')
    if not actuals_df.empty:
        monthly_actuals_pivot = actuals_df.pivot_table(index='Return_Month_Start', 
                                                       columns='Factor_ID', 
                                                       values='Actual_Return')
        monthly_actuals_pivot.index.name = 'Date' # Match T2_Optimizer format
    else:
        monthly_actuals_pivot = pd.DataFrame()

    return monthly_predictions_pivots, monthly_actuals_pivot

# Section: Saving Outputs
def save_outputs(monthly_predictions_pivots, monthly_actuals_pivot, prediction_performance_df):
    if VERBOSE: print("\nSaving output files...")

    # 1. Monthly Predictions Excel
    predictions_excel_file = output_dir_mf / "S12_1_MF_Monthly_Predictions.xlsx"
    with pd.ExcelWriter(predictions_excel_file, engine='openpyxl') as writer:
        for model_type, df_pivot in monthly_predictions_pivots.items():
            if not df_pivot.empty:
                df_pivot.to_excel(writer, sheet_name=f'{model_type}_Predictions')
            else:
                if VERBOSE: print(f"  No predictions to save for model type {model_type}.")
    if VERBOSE: print(f"  Monthly predictions saved to: {predictions_excel_file}")

    # 2. Monthly Actual Returns Excel
    actuals_excel_file = output_dir_mf / "S12_1_MF_Monthly_Actual_Returns.xlsx"
    if not monthly_actuals_pivot.empty:
        monthly_actuals_pivot.to_excel(actuals_excel_file, sheet_name='Monthly_Actual_Returns')
        if VERBOSE: print(f"  Monthly actual returns saved to: {actuals_excel_file}")
    else:
        if VERBOSE: print(f"  No actual returns to save.")

    # 3. Prediction Performance Excel
    performance_excel_file = output_dir_mf / "S12_1_MF_Prediction_Performance.xlsx"
    if not prediction_performance_df.empty:
        prediction_performance_df.to_excel(performance_excel_file, index=False, sheet_name='Prediction_Performance_Summary')
        if VERBOSE: print(f"  Prediction performance summary saved to: {performance_excel_file}")
    else:
        if VERBOSE: print(f"  No prediction performance data to save.")

# Section: Main Execution
def main():
    print("\n" + "="*80)
    print("STEP 12.1 (Multifactor): GENERATE PREDICTIONS AND ASSEMBLE OUTPUTS")
    print("="*80)
    overall_start_time = time.time()

    try:
        # 1. Load data
        trained_models, prediction_data, window_schedule = load_prediction_data()

        # 2. Generate raw predictions and window-level performance
        all_predictions_df, prediction_performance_df = process_all_windows_predictions(
            trained_models, prediction_data, window_schedule
        )

        # 3. Assemble monthly tables (handling overlaps)
        monthly_predictions_pivots, monthly_actuals_pivot = assemble_monthly_outputs(all_predictions_df)

        # 4. Save all outputs
        save_outputs(monthly_predictions_pivots, monthly_actuals_pivot, prediction_performance_df)

    except Exception as e:
        print(f"Error during Step 12.1 (MF) execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        overall_elapsed_time = time.time() - overall_start_time
        print(f"\nStep 12.1 (MF) completed in {overall_elapsed_time:.2f} seconds ({overall_elapsed_time/60:.2f} minutes).")
        print(f"Output files are in: {output_dir_mf}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
