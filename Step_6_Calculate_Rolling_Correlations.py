# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S2_T2_Optimizer_with_MA.xlsx
  - Path: ./output/S2_T2_Optimizer_with_MA.xlsx (output from Step 2)
  - Description: Excel file containing factor data with all moving averages.
                 Contains 'Date' column and many factor columns with various MA periods.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

- S4_Window_Schedule.xlsx
  - Path: ./output/S4_Window_Schedule.xlsx (output from Step 4)
  - Description: Excel file containing the schedule of all 236 rolling windows.
  - Format: Excel (.xlsx) with window parameters and dates.

OUTPUT FILES:
- S6_Rolling_Correlations.h5
  - Path: ./output/S6_Rolling_Correlations.h5
  - Description: HDF5 file containing correlation matrices for each window's training period.
                 Each window has a separate correlation matrix for the 106 factors.
  - Format: HDF5 with window-specific correlation matrices.

- S6_Correlation_Sample_Visualizations.pdf
  - Path: ./output/S6_Correlation_Sample_Visualizations.pdf
  - Description: PDF with visualizations of correlation matrices for sample windows.
  - Format: PDF with heatmaps and other correlation visualizations.

----------------------------------------------------------------------------------------------------
Purpose:
This script calculates correlation matrices for all 106 factors for each rolling window's 
60-month training period. These correlations will be used for feature selection in subsequent steps.

Version: 1.0
Last Updated: Current Date
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
import sys
import h5py
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Section: Utility Functions
def calculate_correlation_matrix(df, factor_columns):
    """
    Calculate correlation matrix for specified factor columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing factor data.
    factor_columns : list
        List of factor column names to include in correlation matrix.
    
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix as DataFrame.
    """
    # Calculate correlations
    corr_matrix = df[factor_columns].corr()
    return corr_matrix

def save_correlations_to_h5(correlations_dict, output_file):
    """
    Save dictionary of correlation matrices to HDF5 file.
    
    Parameters:
    -----------
    correlations_dict : dict
        Dictionary with window IDs as keys and correlation matrices as values.
    output_file : str
        Path to output HDF5 file.
    """
    with h5py.File(output_file, 'w') as hf:
        # Create a group for the correlations metadata
        meta = hf.create_group('metadata')
        meta.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta.attrs['total_windows'] = len(correlations_dict)
        
        # Create a group for correlations
        corr_group = hf.create_group('correlations')
        
        # Loop through each window's correlation matrix
        for window_id, corr_matrix in correlations_dict.items():
            # Convert the window_id to string for group name
            window_group = corr_group.create_group(f'window_{window_id}')
            
            # Store correlation matrix
            window_group.create_dataset('matrix', data=corr_matrix.values)
            
            # Store column names as attributes
            window_group.attrs['column_names'] = np.array(corr_matrix.columns.tolist(), dtype='S')

def create_correlation_visualizations(correlations_dict, window_schedule, factor_data, output_file, num_samples=5):
    """
    Create visualizations of correlation matrices for sample windows.
    
    Parameters:
    -----------
    correlations_dict : dict
        Dictionary with window IDs as keys and correlation matrices as values.
    window_schedule : pandas.DataFrame
        DataFrame containing window schedule information.
    factor_data : pandas.DataFrame
        DataFrame containing the original factor data.
    output_file : str
        Path to output PDF file.
    num_samples : int
        Number of sample windows to visualize (default: 5).
    """
    # Set plot style
    plt.style.use('ggplot')
    sns.set(font_scale=0.7)
    
    # Sample window IDs
    all_window_ids = list(correlations_dict.keys())
    if len(all_window_ids) > num_samples:
        sample_window_ids = np.linspace(min(all_window_ids), max(all_window_ids), num_samples, dtype=int)
    else:
        sample_window_ids = all_window_ids
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Rolling Correlation Matrix Visualizations\nFactor Return Forecasting Project',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        plt.text(0.5, 0.3, f'Sample of {len(sample_window_ids)} windows from total {len(all_window_ids)} windows',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        pdf.savefig()
        plt.close()
        
        # Correlation heatmaps for sampled windows
        for window_id in sample_window_ids:
            if window_id not in correlations_dict:
                continue
                
            corr_matrix = correlations_dict[window_id]
            window_info = window_schedule[window_schedule['Window_ID'] == window_id].iloc[0]
            
            # Create a figure with correlation heatmap
            plt.figure(figsize=(11, 9))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
            cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Red-Blue diverging colormap
            
            # Plot heatmap
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            
            # Add title
            plt.title(f'Window {window_id} Correlation Matrix\n'
                     f'Training Period: {window_info["Training_Start_Date"].strftime("%Y-%m-%d")} to {window_info["Training_End_Date"].strftime("%Y-%m-%d")}')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # For each window, also show a density plot of correlation values
            plt.figure(figsize=(8, 6))
            # Extract the lower triangle of the correlation matrix (to avoid duplicates)
            corr_values = corr_matrix.values[np.tril_indices_from(corr_matrix.values, k=-1)]
            sns.histplot(corr_values, kde=True, bins=50)
            plt.title(f'Window {window_id} - Distribution of Correlation Values')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # For each window, show top 10 most correlated factor pairs
            plt.figure(figsize=(10, 6))
            # Convert to 1D series with MultiIndex
            corr_series = corr_matrix.stack().reset_index()
            corr_series.columns = ['Factor1', 'Factor2', 'Correlation']
            # Filter to keep only lower triangle (no self-correlations)
            corr_series = corr_series[corr_series['Factor1'] != corr_series['Factor2']]
            # Get top 10 most positively correlated pairs
            top_pos = corr_series.nlargest(10, 'Correlation')
            # Create a horizontal bar chart
            plt.barh(y=[f"{row['Factor1']} - {row['Factor2']}" for _, row in top_pos.iterrows()], 
                    width=top_pos['Correlation'])
            plt.title(f'Window {window_id} - Top 10 Most Positively Correlated Factor Pairs')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # For each window, show bottom 10 most negatively correlated factor pairs
            plt.figure(figsize=(10, 6))
            # Get bottom 10 most negatively correlated pairs
            top_neg = corr_series.nsmallest(10, 'Correlation')
            # Create a horizontal bar chart
            plt.barh(y=[f"{row['Factor1']} - {row['Factor2']}" for _, row in top_neg.iterrows()], 
                    width=top_neg['Correlation'])
            plt.title(f'Window {window_id} - Top 10 Most Negatively Correlated Factor Pairs')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Add correlation stability analysis across windows
        if len(sample_window_ids) > 1:
            # Select a few factor pairs and track their correlation across windows
            all_matrices = [correlations_dict[w_id] for w_id in all_window_ids if w_id in correlations_dict]
            if len(all_matrices) > 1:
                # Get all factors
                all_factors = all_matrices[0].columns.tolist()
                
                # Select a few random factor pairs
                np.random.seed(42)  # For reproducibility
                num_pairs = min(5, len(all_factors))
                factor_pairs = []
                for _ in range(num_pairs):
                    i, j = np.random.choice(range(len(all_factors)), 2, replace=False)
                    factor_pairs.append((all_factors[i], all_factors[j]))
                
                # Plot correlation over time for these pairs
                plt.figure(figsize=(10, 6))
                for f1, f2 in factor_pairs:
                    # Extract correlation over time
                    corr_over_time = [matrix.loc[f1, f2] for matrix in all_matrices]
                    # Get corresponding dates
                    dates = [window_schedule[window_schedule['Window_ID'] == w_id].iloc[0]['Training_End_Date'] 
                            for w_id in all_window_ids if w_id in correlations_dict]
                    
                    plt.plot(dates, corr_over_time, marker='o', label=f'{f1} - {f2}')
                
                plt.title('Correlation Stability for Selected Factor Pairs Across Windows')
                plt.xlabel('Window End Date')
                plt.ylabel('Correlation Coefficient')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        # Summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Summary:\n\n'
                          f'Total Windows Analyzed: {len(all_window_ids)}\n\n'
                          f'First Window Training Period:\n'
                          f'  {window_schedule.iloc[0]["Training_Start_Date"].strftime("%Y-%m-%d")} to {window_schedule.iloc[0]["Training_End_Date"].strftime("%Y-%m-%d")}\n\n'
                          f'Last Window Training Period:\n'
                          f'  {window_schedule.iloc[-1]["Training_Start_Date"].strftime("%Y-%m-%d")} to {window_schedule.iloc[-1]["Training_End_Date"].strftime("%Y-%m-%d")}\n\n'
                          f'Number of Factors: {len(correlations_dict[all_window_ids[0]].columns)}\n\n'
                          f'Notes:\n'
                          f'- Correlation matrices are calculated using only the original factor values (not MAs)\n'
                          f'- Each window uses its 60-month training period for correlation calculation\n',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()

# Section: Main Script Logic
def main():
    print("=== Step 6: Calculate Rolling Correlations ===")
    
    # --- Step 6.1: Load Data ---
    print("--- 6.1 Loading Data ---")
    
    # Define file paths
    input_factor_file = os.path.join("output", "S2_T2_Optimizer_with_MA.xlsx")
    input_window_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output files with S6 prefix in output directory
    output_corr_file = os.path.join(output_dir, "S6_Rolling_Correlations.h5")
    output_viz_file = os.path.join(output_dir, "S6_Correlation_Sample_Visualizations.pdf")
    
    try:
        # Load factor data
        factor_data = pd.read_excel(input_factor_file)
        print(f"Loaded factor data from {input_factor_file}")
        
        # Load window schedule
        window_schedule = pd.read_excel(input_window_file)
        print(f"Loaded window schedule from {input_window_file}")
    except Exception as e:
        print(f"Error loading input files: {e}")
        sys.exit(1)
    
    # --- Step 6.2: Identify Columns ---
    print("\n--- 6.2 Identifying Columns ---")
    
    date_column_name = "Date"
    
    # Ensure date column is in datetime format
    factor_data[date_column_name] = pd.to_datetime(factor_data[date_column_name])
    
    # Extract the original factor columns (exclude MA columns)
    all_columns = factor_data.columns.tolist()
    factor_columns = []
    
    for col in all_columns:
        if col == date_column_name:
            continue
        
        # Only include original factors (not MA columns)
        if "_MA_" not in col:
            factor_columns.append(col)
    
    print(f"Identified {len(factor_columns)} original factor columns for correlation analysis")
    
    # --- Step 6.3: Calculate Rolling Correlations ---
    print("\n--- 6.3 Calculating Rolling Correlations ---")
    
    # Dictionary to store correlation matrices
    correlations_dict = {}
    
    # Process each window
    total_windows = len(window_schedule)
    print(f"Processing {total_windows} windows...")
    
    # Set up progress tracking
    milestone_interval = max(1, total_windows // 10)  # Report progress at 10% intervals
    next_milestone = milestone_interval
    
    for idx, window in window_schedule.iterrows():
        window_id = window['Window_ID']
        
        # Extract training data for this window
        start_idx = window['Training_Start_Index']
        end_idx = window['Training_End_Index']
        
        # Get the corresponding rows from factor data
        training_data = factor_data.iloc[start_idx:end_idx+1].copy()
        
        # Calculate correlation matrix using only the original factors
        corr_matrix = calculate_correlation_matrix(training_data, factor_columns)
        
        # Store in dictionary
        correlations_dict[window_id] = corr_matrix
        
        # Show progress
        if idx + 1 >= next_milestone or idx + 1 == total_windows:
            progress_pct = (idx + 1) / total_windows * 100
            print(f"Processed window {idx + 1}/{total_windows} ({progress_pct:.1f}%)")
            next_milestone += milestone_interval
    
    # --- Step 6.4: Save Correlation Matrices ---
    print("\n--- 6.4 Saving Correlation Matrices ---")
    
    # Save to HDF5 file
    save_correlations_to_h5(correlations_dict, output_corr_file)
    print(f"Saved correlation matrices to {output_corr_file}")
    
    # --- Step 6.5: Generate Correlation Visualizations ---
    print("\n--- 6.5 Generating Correlation Visualizations ---")
    
    create_correlation_visualizations(correlations_dict, window_schedule, factor_data, 
                                     output_viz_file)
    print(f"Generated correlation visualizations in {output_viz_file}")
    
    # --- Step 6.6: Summary ---
    print("\n--- 6.6 Summary ---")
    
    print(f"Total windows processed: {total_windows}")
    print(f"Number of factors in each correlation matrix: {len(factor_columns)}")
    print(f"Output files:")
    print(f"  - HDF5 file with all correlation matrices: {output_corr_file}")
    print(f"  - PDF with correlation visualizations: {output_viz_file}")
    
    print("\nStep 6 completed successfully!")

if __name__ == "__main__":
    main() 