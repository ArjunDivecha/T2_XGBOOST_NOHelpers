# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S1_T2_Optimizer_cleaned.xlsx
  - Path: ./output/S1_T2_Optimizer_cleaned.xlsx (output from Step 1)
  - Description: Excel file containing cleaned monthly factor return data.
                 Contains 'Date' column and 106 factor columns.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

OUTPUT FILES:
- S2_T2_Optimizer_with_MA.xlsx
  - Path: ./output/S2_T2_Optimizer_with_MA.xlsx
  - Description: Excel file containing the original data and all calculated moving averages.
                 For each factor, includes:
                 - Original values (1-month MA)
                 - 3-month MA (short-term trend)
                 - 12-month MA (medium-term trend)
                 - 60-month MA (long-term trend)
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, all other columns as numeric.

- S2_MA_Visualization.pdf
  - Path: ./output/S2_MA_Visualization.pdf
  - Description: PDF file with visualizations of 60-month moving averages for all factors.
  - Format: PDF with multiple plots.

- S2_Column_Mapping.xlsx
  - Path: ./output/S2_Column_Mapping.xlsx
  - Description: Excel file mapping original factor names to their MA column names.
  - Format: Excel (.xlsx) with column mapping information.

----------------------------------------------------------------------------------------------------
Version: 2.0
Last Updated: Current Date
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
import sys
import re

# Section: Utility Functions
def calculate_moving_averages(series, column_name, windows=[1, 3, 12, 60]):
    """
    Calculate moving averages for a time series.
    
    IMPORTANT: Each factor (including those with CS/TS suffixes) is treated 
    as a completely independent factor.
    
    Parameters:
    -----------
    series : pandas.Series
        The time series data to calculate MAs for.
    column_name : str
        The original column name of the series.
    windows : list
        List of window sizes to calculate.
        
    Returns:
    --------
    dict
        Dictionary of pandas.Series with the moving averages and their column names.
    """
    result = {}
    
    # Check if the column already has time period suffix like "_3m", "_12m", "_60m"
    existing_period_match = re.search(r'_(\d+)m$', column_name)
    
    # Store original series under the original column name
    result[column_name] = series
    
    # If the column already has a time period, we may want to skip calculating some windows
    skip_windows = []
    if existing_period_match:
        existing_period = int(existing_period_match.group(1))
        # Skip calculating the same window as already exists in the name
        skip_windows.append(existing_period)
        
    # Generate MA series and their column names
    for window in windows:
        if window == 1 or window in skip_windows:
            # Skip 1-month (original data) and any windows that match existing periods
            continue
        else:
            # For other windows, calculate the rolling mean
            ma_series = series.rolling(window=window, min_periods=1).mean()
            
            # Simply append the MA period to the original factor name - treat each factor as distinct
            ma_column_name = f"{column_name}_{window}m"
            
            result[ma_column_name] = ma_series
    
    return result

def create_ma_visualizations(df, date_col, factor_cols, output_file):
    """
    Create visualizations of 60-month moving averages for all factors.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the original and MA data.
    date_col : str
        Name of the date column.
    factor_cols : list
        List of factor column names.
    output_file : str
        Path to output PDF file.
    """
    # Set plot style
    plt.style.use('ggplot')
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, '60-Month Moving Average Visualizations\nAll Factors',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()
        
        # For each factor, create a visualization
        for factor in factor_cols:
            # Find the 60-month MA column for this factor
            ma_60m_col = f"{factor}_60m"
            
            if ma_60m_col not in df.columns:
                # If no 60-month MA column found, skip this factor
                print(f"Warning: No 60-month MA column found for {factor}, skipping visualization")
                continue
            
            plt.figure(figsize=(11, 8.5))
            
            # Plot 60-month MA
            plt.plot(df[date_col], df[ma_60m_col], label="60-month MA", linewidth=2.5, color='blue')
            
            plt.title(f"60-Month MA for {factor}")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Add to PDF
            pdf.savefig()
            plt.close()
            
        # Add a summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Summary:\nVisualized 60-month MA for all {len(factor_cols)} factors\n' +
                 f'Date range: {df[date_col].min()} to {df[date_col].max()}\n' +
                 f'Total data points: {len(df)}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()

# Section: Main Script Logic
def main():
    print("=== Step 2: Calculate Moving Averages ===")
    
    # --- Step 2.1: Load Cleaned Data ---
    print("--- 2.1 Loading Cleaned Data ---")
    
    # Define file paths
    input_file = os.path.join("output", "S1_T2_Optimizer_cleaned.xlsx")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output files with S2 prefix in output directory
    output_file = os.path.join(output_dir, "S2_T2_Optimizer_with_MA.xlsx")
    visualization_file = os.path.join(output_dir, "S2_MA_Visualization.pdf")
    column_mapping_file = os.path.join(output_dir, "S2_Column_Mapping.xlsx")
    
    try:
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {input_file}")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)
    
    # Identify the date column and factor columns
    date_column_name = "Date"  # Assuming this is the name from Step 1
    
    if date_column_name not in df.columns:
        print(f"Error: Date column '{date_column_name}' not found in the file. Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    # Ensure date column is in datetime format
    df[date_column_name] = pd.to_datetime(df[date_column_name])
    
    # Sort by date
    df = df.sort_values(by=date_column_name)
    
    # Identify factor columns (all columns except date)
    factor_columns = [col for col in df.columns if col != date_column_name]
    print(f"Identified {len(factor_columns)} factor columns")
    
    # --- Step 2.2: Calculate Moving Averages ---
    print("--- 2.2 Calculating Moving Averages ---")
    
    # Create a new DataFrame to hold results, starting with the date column
    result_df = pd.DataFrame({date_column_name: df[date_column_name]})
    
    # Create a mapping dictionary to keep track of column name relationships
    # This will help Step 8 understand the naming conventions
    column_mapping = {
        "factor_name": [],
        "original_column": [],
        "column_3m": [],
        "column_12m": [],
        "column_60m": []
    }
    
    # Process each factor column
    for factor_col in factor_columns:
        print(f"  Calculating MAs for {factor_col}")
        
        # Calculate moving averages
        ma_dict = calculate_moving_averages(df[factor_col], factor_col)
        
        # Add all series from the result to the result_df
        for col_name, series in ma_dict.items():
            result_df[col_name] = series
        
        # Find the actual column names created for this factor
        col_3m = f"{factor_col}_3m" if f"{factor_col}_3m" in ma_dict else "N/A"
        col_12m = f"{factor_col}_12m" if f"{factor_col}_12m" in ma_dict else "N/A"
        col_60m = f"{factor_col}_60m" if f"{factor_col}_60m" in ma_dict else "N/A"
        
        # IMPORTANT: Treat each factor (including those with CS/TS suffix) as a completely independent factor
        column_mapping["factor_name"].append(factor_col)
        column_mapping["original_column"].append(factor_col)
        column_mapping["column_3m"].append(col_3m)
        column_mapping["column_12m"].append(col_12m)
        column_mapping["column_60m"].append(col_60m)
    
    # Save the column mapping for future steps
    pd.DataFrame(column_mapping).to_excel(column_mapping_file, index=False)
    print(f"Saved column mapping information to {column_mapping_file}")
    
    # --- Step 2.3: Save Results ---
    print("--- 2.3 Saving Results ---")
    result_df.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")
    
    # --- Step 2.4: Generate Visualizations ---
    print("--- 2.4 Generating Visualizations ---")
    create_ma_visualizations(result_df, date_column_name, factor_columns, visualization_file)
    print(f"Generated visualization samples in {visualization_file}")
    
    # --- Step 2.5: Report Summary Statistics ---
    print("--- 2.5 Summary Statistics ---")
    print(f"Total rows: {len(result_df)}")
    print(f"Date range: {result_df[date_column_name].min()} to {result_df[date_column_name].max()}")
    print(f"Total columns in output: {len(result_df.columns)}")
    print(f"  - Original factor columns: {len(factor_columns)}")
    print(f"  - New columns created: {len(result_df.columns) - len(factor_columns) - 1}")  # -1 for date column
    print(f"  - Total columns (including date): {len(result_df.columns)}")
    
    print("\nStep 2 completed successfully!")

if __name__ == "__main__":
    main()