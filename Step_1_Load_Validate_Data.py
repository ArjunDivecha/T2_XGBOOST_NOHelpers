# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- T2_Optimizer.xlsx
  - Path: ./T2_Optimizer.xlsx (assumed to be in the same directory as the script)
  - Description: Excel file containing monthly factor return data.
                 Expected to have a 'Date' column and 106 factor columns.
  - Format: Excel (.xlsx) with a header row. The 'Date' column should be parsable as dates.
            Factor columns should contain numeric data.

OUTPUT FILES:
- S1_T2_Optimizer_cleaned.xlsx
  - Path: ./output/S1_T2_Optimizer_cleaned.xlsx
  - Description: Excel file containing the cleaned and validated factor return data.
                 Missing values for each date are filled with the cross-sectional mean of
                 available factor values for that specific date.
                 Data is sorted by date.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

----------------------------------------------------------------------------------------------------
PURPOSE:
This script performs the first step of the data preparation phase for the Factor Return Forecasting project.
It loads the raw factor data from an Excel file, validates its structure and integrity,
handles missing values by imputing cross-sectional means (the mean of other available factors for the same date),
sorts the data by date, and saves the cleaned dataset to a new Excel file.
It also provides a summary report of the operations performed.

----------------------------------------------------------------------------------------------------
VERSION HISTORY:
- v1.0 (YYYY-MM-DD): Initial version.

DATE OF LAST UPDATE:
- YYYY-MM-DD (This will be updated to the actual date upon first run/save)
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
from datetime import datetime
import numpy as np # Import NumPy
import os # Add os for directory operations

# Section: Main Script Logic
def main():
    '''
    Main function to execute the data loading, validation, and cleaning process.
    '''
    print("----------------------------------------------------------------------")
    print("Step 1: Load, Validate, and Clean Factor Data")
    print("----------------------------------------------------------------------")

    # Update Date of Last Update
    # This is a placeholder; in a real scenario, this might be handled by version control
    # or a more sophisticated script update mechanism.
    current_date = datetime.now().strftime("%Y-%m-%DD")
    print(f"Script last updated (simulated for this run): {current_date}\\n")

    # --- Configuration ---
    input_file_path = 'T2_Optimizer.xlsx'
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output file with S1 prefix in output directory
    output_file_path = os.path.join(output_dir, 'S1_T2_Optimizer_cleaned.xlsx')
    
    date_column_name = 'Date'

    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Date column: {date_column_name}\\n")

    # --- Step 1.1: Load Data ---
    print("--- 1.1 Loading Data ---")
    try:
        df = pd.read_excel(input_file_path, parse_dates=[date_column_name])
        print(f"Successfully loaded data from '{input_file_path}'.\\n")
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file_path}' not found. Please ensure the file exists in the correct location.")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading the Excel file: {e}")
        return

    # --- Step 1.2: Initial Validation and Inspection ---
    print("--- 1.2 Initial Validation and Inspection ---")
    print("DataFrame shape (rows, columns):", df.shape)
    print("\\nDataFrame Info:")
    df.info(verbose=True, show_counts=True) # verbose and show_counts for more details
    print("\\nDataFrame Head (first 5 rows):")
    print(df.head())
    print("\\nDataFrame Tail (last 5 rows):")
    print(df.tail())

    # Date column validation
    if date_column_name not in df.columns:
        print(f"\\nERROR: Date column '{date_column_name}' not found in the DataFrame.")
        return

    print(f"\\nDate column '{date_column_name}' data type: {df[date_column_name].dtype}")
    if pd.api.types.is_datetime64_any_dtype(df[date_column_name]):
        print("Date column is recognized as datetime objects.")
        print(f"Date range: {df[date_column_name].min()} to {df[date_column_name].max()}")
        print(f"Number of unique dates: {df[date_column_name].nunique()}")
        if not df[date_column_name].is_monotonic_increasing:
            print("Note: Date column is not strictly monotonically increasing before sorting. Sorting will be applied.")
    else:
        print("WARNING: Date column is NOT recognized as datetime. Attempting conversion or check file format.")
        # Attempt conversion if not already parsed correctly, though parse_dates should handle it.
        try:
            df[date_column_name] = pd.to_datetime(df[date_column_name])
            print("Successfully converted date column to datetime.")
            print(f"New date range: {df[date_column_name].min()} to {df[date_column_name].max()}")
        except Exception as e:
            print(f"ERROR: Could not convert date column '{date_column_name}' to datetime: {e}")
            return
    print("\\n")

    # --- Step 1.3: Handle Missing Values (Cross-Sectional Mean Imputation) ---
    print("--- 1.3 Handling Missing Values (Cross-Sectional Mean Imputation) ---")
    
    # Identify all factor columns (numeric columns excluding the date column)
    factor_columns = df.select_dtypes(include=np.number).columns.tolist()
    # Ensure date_column_name is not in factor_columns if it was numeric for some reason, though unlikely
    if date_column_name in factor_columns:
        factor_columns.remove(date_column_name)

    initial_missing_values_count = df[factor_columns].isnull().sum().sum()

    if initial_missing_values_count == 0:
        print("No missing values found in numeric factor columns.")
        missing_data_log = [] # Ensure missing_data_log is initialized
        initial_missing_per_column_summary = pd.Series(dtype=float) # For report consistency
    else:
        print(f"Initial total missing values in factor columns: {initial_missing_values_count}")
        initial_missing_per_column_summary = df[factor_columns].isnull().sum()
        initial_missing_per_column_summary = initial_missing_per_column_summary[initial_missing_per_column_summary > 0]
        print("Initial missing values per factor column (only columns with missing data):")
        print(initial_missing_per_column_summary)
        print("\\nFilling missing values with the cross-sectional mean for each date...")

        missing_data_log = []
        
        # Create a copy to avoid SettingWithCopyWarning if we modify slices later,
        # though direct df.loc assignment should be fine.
        df_copy = df.copy()

        for index, row in df_copy.iterrows():
            # Get all factor values for the current row
            row_factor_values = row[factor_columns]
            
            # Calculate cross-sectional mean for this row (ignoring NaNs)
            cross_sectional_mean = row_factor_values.mean() # mean() on a Series ignores NaNs by default
            num_values_for_mean = row_factor_values.notna().sum()

            if pd.isna(cross_sectional_mean):
                # This happens if all factor values in the row are NaN
                if row_factor_values.isnull().all():
                    current_row_date_str = row[date_column_name].strftime('%Y-%m-%d')
                    print(f"  - WARNING: Date {current_row_date_str}: All factor values are missing. Cannot calculate cross-sectional mean. NaNs will remain for this date if any.")
                    missing_data_log.append({
                        'date': current_row_date_str,
                        'columns_filled': 'None - All factors missing',
                        'num_filled_in_row': 0,
                        'cross_sectional_mean_used': 'N/A',
                        'factors_in_mean_calc': 0
                    })
                continue # Move to the next row

            # Identify columns in this row that are NaN and need filling
            cols_to_fill_in_row = row_factor_values[row_factor_values.isnull()].index.tolist()
            
            if cols_to_fill_in_row:
                num_filled_this_row = 0
                filled_cols_this_row_names = []
                current_row_date_str = row[date_column_name].strftime('%Y-%m-%d') # get date string once
                for col_to_fill in cols_to_fill_in_row:
                    # Ensure we are modifying the original df, not the copy or a slice of a slice
                    df.loc[index, col_to_fill] = cross_sectional_mean
                    num_filled_this_row += 1
                    filled_cols_this_row_names.append(col_to_fill)
                
                if num_filled_this_row > 0:
                    filled_cols_str = ", ".join(filled_cols_this_row_names)
                    print(f"  - Date {current_row_date_str}: Filled {num_filled_this_row} NaN(s) in columns [{filled_cols_str}] with cross-sectional mean: {cross_sectional_mean:.4f} (calculated from {num_values_for_mean} factors).")
                    missing_data_log.append({
                        'date': current_row_date_str,
                        'columns_filled': filled_cols_str,
                        'num_filled_in_row': num_filled_this_row,
                        'cross_sectional_mean_used': cross_sectional_mean,
                        'factors_in_mean_calc': num_values_for_mean
                    })
        
        if not missing_data_log and initial_missing_values_count > 0:
             print("No missing values were ultimately filled by cross-sectional mean (e.g. all NaNs were in rows where all factors were NaN).")


    final_missing_values_count = df[factor_columns].isnull().sum().sum()
    print(f"\\nTotal missing values in factor columns after cross-sectional imputation: {final_missing_values_count}\\n")

    # --- Step 1.4: Sort Data ---
    print("--- 1.4 Sorting Data ---")
    # Sort by date column, ensure it's done after potential date conversion and before saving
    df.sort_values(by=date_column_name, inplace=True)
    print(f"DataFrame sorted by '{date_column_name}' in ascending order.")
    # Verify sort
    if df[date_column_name].is_monotonic_increasing:
        print("Date column is now monotonically increasing.")
    else:
        print("WARNING: Date column is NOT monotonically increasing even after sorting. Check data for duplicate dates or other issues.")
    print("\\n")

    # --- Step 1.5: Data Quality and Completeness Report (Summary) ---
    print("--- 1.5 Data Quality and Completeness Report ---")
    print("Final DataFrame shape (rows, columns):", df.shape)
    
    print("\\nSummary of Missing Values Handled (Cross-Sectional):")
    if initial_missing_values_count > 0:
        if missing_data_log:
            total_filled_entries = sum(log['num_filled_in_row'] for log in missing_data_log)
            print(f"  - A total of {total_filled_entries} missing factor entries were addressed using cross-sectional means.")
            # Optionally, print more details from the log here or save the log to a file
            # For brevity, we'll just summarize
            if final_missing_values_count == 0:
                print("  - All identified missing values in factor columns have been handled using cross-sectional imputation.")
            else:
                print(f"  - WARNING: {final_missing_values_count} missing values remain in factor columns. This might occur if entire rows had no data to compute a mean.")
        else:
            print("  - No missing values were filled by cross-sectional mean (e.g., all NaNs were in rows where all factors were NaN, or no NaNs initially).")

    elif initial_missing_values_count == 0:
        print("  - No missing values were present in the numeric factor columns of the original dataset.")
    
    if df.isnull().sum().sum() == 0: # Check all columns now, including date if it had issues
        print("\\nOverall, the DataFrame has no remaining missing values.")
    else:
        remaining_total_nans = df.isnull().sum().sum()
        print(f"\\nWARNING: {remaining_total_nans} total missing values remain in the ENTIRE dataset (could be non-factor columns or factors if all were NaN in a row).")
    print("\\n")

    # --- Step 1.6: Save Cleaned Data ---
    print("--- 1.6 Saving Cleaned Data ---")
    try:
        df.to_excel(output_file_path, index=False)
        print(f"Successfully saved cleaned data to '{output_file_path}'.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving the Excel file: {e}")
        return
    
    print("----------------------------------------------------------------------")
    print("Step 1 Processing Complete.")
    print("----------------------------------------------------------------------")

# Section: Script Execution Guard
if __name__ == '__main__':
    main() 