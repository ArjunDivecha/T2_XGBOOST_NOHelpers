# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S2_T2_Optimizer_with_MA.xlsx
  - Path: ./output/S2_T2_Optimizer_with_MA.xlsx (output from Step 2)
  - Description: Excel file containing the original factor data and all calculated moving averages.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

OUTPUT FILES:
- S3_Benchmark_Series.xlsx
  - Path: ./output/S3_Benchmark_Series.xlsx
  - Description: Excel file containing the equal-weighted benchmark series calculated by
                 averaging all factors. Includes both the original factor values and their MAs.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, all benchmark series as numeric.

- S3_Benchmark_Visualization.pdf
  - Path: ./output/S3_Benchmark_Visualization.pdf
  - Description: PDF file with visualizations of the equal-weighted benchmark series.
  - Format: PDF with benchmark plots.

----------------------------------------------------------------------------------------------------
Purpose:
This script creates an equal-weighted benchmark series by averaging all factors for each time period.
The benchmark represents the average performance across all factors and serves as a reference point
for evaluating factor performance.

Version: 1.0
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

# Section: Utility Functions
def create_benchmark_series(df, date_col, factor_cols, ma_windows=['3m', '12m', '60m']):
    """
    Create equal-weighted benchmark series for original factors and their MAs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the original and MA data.
    date_col : str
        Name of the date column.
    factor_cols : list
        List of original factor column names (without MA suffixes).
    ma_windows : list
        List of MA window suffixes.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the benchmark series.
    """
    # Initialize result DataFrame with date column
    result = pd.DataFrame({date_col: df[date_col]})
    
    # Calculate benchmark for original factor values
    result['Benchmark_Original'] = df[factor_cols].mean(axis=1)
    
    # Calculate benchmark for each MA window
    for window in ma_windows:
        ma_cols = [f"{col}_{window}" for col in factor_cols]
        result[f'Benchmark_{window}'] = df[ma_cols].mean(axis=1)
    
    return result

def create_benchmark_visualizations(df, date_col, output_file):
    """
    Create visualizations of benchmark series.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the benchmark series.
    date_col : str
        Name of the date column.
    output_file : str
        Path to output PDF file.
    """
    benchmark_cols = [col for col in df.columns if col.startswith('Benchmark_')]
    
    # Set plot style
    plt.style.use('ggplot')
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Equal-Weighted Benchmark Series\nAll Factors',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()
        
        # Create a plot with all benchmark series
        plt.figure(figsize=(11, 8.5))
        for col in benchmark_cols:
            plt.plot(df[date_col], df[col], label=col.replace('Benchmark_', ''), linewidth=2)
        
        plt.title("Equal-Weighted Benchmark Series")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Create individual plots for each benchmark series
        for col in benchmark_cols:
            plt.figure(figsize=(11, 8.5))
            plt.plot(df[date_col], df[col], linewidth=2.5)
            
            plt.title(f"Equal-Weighted Benchmark: {col.replace('Benchmark_', '')}")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Add a summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Summary:\nCreated {len(benchmark_cols)} benchmark series\n' +
                 f'Date range: {df[date_col].min()} to {df[date_col].max()}\n' +
                 f'Total data points: {len(df)}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()

# Section: Main Script Logic
def main():
    print("=== Step 3: Create Benchmark Series ===")
    
    # --- Step 3.1: Load Moving Average Data ---
    print("--- 3.1 Loading Moving Average Data ---")
    
    # Define file paths
    input_file = os.path.join("output", "S2_T2_Optimizer_with_MA.xlsx")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output files with S3 prefix in output directory
    output_file = os.path.join(output_dir, "S3_Benchmark_Series.xlsx")
    visualization_file = os.path.join(output_dir, "S3_Benchmark_Visualization.pdf")
    
    try:
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {input_file}")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)
    
    # Identify the date column and factor columns
    date_column_name = "Date"  # Assuming this is the name from previous steps
    
    if date_column_name not in df.columns:
        print(f"Error: Date column '{date_column_name}' not found in the file. Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    # Ensure date column is in datetime format
    df[date_column_name] = pd.to_datetime(df[date_column_name])
    
    # Sort by date
    df = df.sort_values(by=date_column_name)
    
    # Identify original factor columns (all columns except date and MA columns)
    all_columns = df.columns.tolist()
    factor_columns = [col for col in all_columns if col != date_column_name and 
                     not any(col.endswith(f"_{window}") for window in ['3m', '12m', '60m'])]
    
    print(f"Identified {len(factor_columns)} original factor columns")
    
    # --- Step 3.2: Calculate Benchmark Series ---
    print("--- 3.2 Calculating Equal-Weighted Benchmark Series ---")
    
    benchmark_df = create_benchmark_series(df, date_column_name, factor_columns)
    
    print(f"Created benchmark series with {len(benchmark_df.columns) - 1} benchmarks")  # -1 for date column
    
    # --- Step 3.3: Save Benchmark Series ---
    print("--- 3.3 Saving Benchmark Series ---")
    benchmark_df.to_excel(output_file, index=False)
    print(f"Saved benchmark series to {output_file}")
    
    # --- Step 3.4: Generate Visualizations ---
    print("--- 3.4 Generating Visualizations ---")
    create_benchmark_visualizations(benchmark_df, date_column_name, visualization_file)
    print(f"Generated benchmark visualizations in {visualization_file}")
    
    # --- Step 3.5: Report Summary Statistics ---
    print("--- 3.5 Summary Statistics ---")
    benchmark_cols = [col for col in benchmark_df.columns if col.startswith('Benchmark_')]
    
    print(f"Created {len(benchmark_cols)} benchmark series:")
    for col in benchmark_cols:
        print(f"  - {col}: Min={benchmark_df[col].min():.4f}, Max={benchmark_df[col].max():.4f}, Mean={benchmark_df[col].mean():.4f}")
    
    print(f"Date range: {benchmark_df[date_column_name].min()} to {benchmark_df[date_column_name].max()}")
    print(f"Total observations: {len(benchmark_df)}")
    
    print("\nStep 3 completed successfully!")

if __name__ == "__main__":
    main() 