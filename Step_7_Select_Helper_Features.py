# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S2_T2_Optimizer_with_MA.xlsx
  - Path: ./output/S2_T2_Optimizer_with_MA.xlsx (output from Step 2)
  - Description: Excel file containing factor data with all moving averages.
                 Used only to get the list of factors.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates, factor columns as numeric.

- S4_Window_Schedule.xlsx
  - Path: ./output/S4_Window_Schedule.xlsx (output from Step 4)
  - Description: Excel file containing the schedule of all 236 rolling windows.
  - Format: Excel (.xlsx) with window parameters and dates.

- S6_Rolling_Correlations.h5
  - Path: ./output/S6_Rolling_Correlations.h5 (output from Step 6)
  - Description: HDF5 file containing correlation matrices for each window's training period.
  - Format: HDF5 with window-specific correlation matrices.

OUTPUT FILES:
- S7_Helper_Features.h5
  - Path: ./output/S7_Helper_Features.h5
  - Description: HDF5 file containing top 10 most correlated helper features for each factor and window.
  - Format: HDF5 with window and factor-specific helper features.

- S7_Helper_Features_Sample.xlsx
  - Path: ./output/S7_Helper_Features_Sample.xlsx
  - Description: Excel file with helper features for a sample of windows and factors.
                 For easy inspection and verification.
  - Format: Excel (.xlsx) with header and organized by window and factor.

- S7_Helper_Features_Visualization.pdf
  - Path: ./output/S7_Helper_Features_Visualization.pdf
  - Description: PDF with visualizations of helper feature selection patterns.
  - Format: PDF with various charts and analysis.

----------------------------------------------------------------------------------------------------
Purpose:
This script selects the top 10 most correlated helper features for each factor in each window.
These helper features will be used in the next step to create the feature sets for the model.

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
from matplotlib.gridspec import GridSpec
from datetime import datetime
import os
import sys
import h5py
import warnings
from collections import Counter, defaultdict
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# Section: Utility Functions
def load_correlations_from_h5(input_file):
    """
    Load correlation matrices from HDF5 file.
    
    Parameters:
    -----------
    input_file : str
        Path to input HDF5 file with correlation matrices.
    
    Returns:
    --------
    dict
        Dictionary with window IDs as keys and correlation matrices as values.
    list
        List of factor column names.
    """
    correlations_dict = {}
    factor_columns = None
    
    with h5py.File(input_file, 'r') as hf:
        # Get correlation matrices for each window
        corr_group = hf['correlations']
        
        for window_name in corr_group:
            window_id = int(window_name.split('_')[1])  # Extract window ID
            window_group = corr_group[window_name]
            
            # Get correlation matrix data
            matrix_data = window_group['matrix'][:]
            
            # Get column names 
            column_names = [name.decode('utf-8') for name in window_group.attrs['column_names']]
            
            # Create DataFrame with proper column and index names
            corr_matrix = pd.DataFrame(matrix_data, columns=column_names, index=column_names)
            
            # Store in dictionary
            correlations_dict[window_id] = corr_matrix
            
            # Store factor columns from first window (assuming all windows have the same factors)
            if factor_columns is None:
                factor_columns = column_names
    
    return correlations_dict, factor_columns

def select_helper_features(correlations_dict, factor_columns, num_helpers=10):
    """
    Select top helper features for each factor in each window.
    Only considers factors with '_60m' suffix as potential helper features.
    
    Parameters:
    -----------
    correlations_dict : dict
        Dictionary with window IDs as keys and correlation matrices as values.
    factor_columns : list
        List of factor column names.
    num_helpers : int
        Number of helper features to select (default: 10).
    
    Returns:
    --------
    dict
        Nested dictionary with window IDs and factors as keys and list of helper features as values.
    """
    helper_features = {}
    
    # Filter to get only 60m factors as potential helpers
    helper_candidates = [col for col in factor_columns if '_60m' in col]
    print(f"Found {len(helper_candidates)} potential helper factors with '_60m' suffix")
    
    if len(helper_candidates) == 0:
        print("WARNING: No factors with '_60m' suffix found. Looking for '_TS_60m' or similar patterns...")
        helper_candidates = [col for col in factor_columns if '60m' in col]
        print(f"Found {len(helper_candidates)} potential helper factors containing '60m'")
        
        if len(helper_candidates) == 0:
            print("ERROR: No suitable helper factors with '60m' found. Using all factors as potential helpers.")
            helper_candidates = factor_columns
    
    for window_id, corr_matrix in correlations_dict.items():
        helper_features[window_id] = {}
        
        for factor in factor_columns:
            # Get correlation series for this factor, but only for 60m helper candidates
            factor_corr = corr_matrix.loc[factor, helper_candidates].copy()
            
            # More robust self-correlation removal
            # First check for exact match
            if factor in factor_corr.index:
                factor_corr = factor_corr.drop(factor)
            
            # Then identify the factor's core identity, preserving _CS and _TS suffixes
            parts = factor.split('_')
            
            # Handle factors with timing suffixes (_60m, _120m, etc.)
            if len(parts) >= 2 and any(timing in parts[-1] for timing in ['60m', '120m', '180m', '240m']):
                # If the factor has a timing suffix, the core is everything except the last part
                core_identity = '_'.join(parts[:-1])
            else:
                # Otherwise, the whole thing is the core identity
                core_identity = factor
                
            # Now exclude helpers that have the same core identity but different timing
            helpers_to_exclude = [helper for helper in factor_corr.index 
                                 if helper.startswith(core_identity + '_')]
            
            if helpers_to_exclude:
                factor_corr = factor_corr.drop(helpers_to_exclude)
                print(f"Excluded {len(helpers_to_exclude)} related helpers for {factor} in window {window_id}")
            
            # Rank by absolute correlation
            factor_corr_abs = factor_corr.abs()
            
            # Ensure we don't try to get more helpers than available
            available_helpers = min(num_helpers, len(factor_corr_abs))
            if available_helpers < num_helpers:
                print(f"Warning: Only {available_helpers} helper candidates available for {factor} in window {window_id}")
                
            top_helpers = factor_corr_abs.nlargest(available_helpers).index.tolist()
            
            # Store original correlation values for these helpers
            helper_with_corr = [(helper, corr_matrix.loc[factor, helper]) for helper in top_helpers]
            
            # Store in dictionary
            helper_features[window_id][factor] = helper_with_corr
    
    return helper_features

def save_helper_features_to_h5(helper_features, factor_columns, output_file):
    """
    Save helper features to HDF5 file.
    
    Parameters:
    -----------
    helper_features : dict
        Nested dictionary with window IDs and factors as keys and list of helper features as values.
    factor_columns : list
        List of factor column names.
    output_file : str
        Path to output HDF5 file.
    """
    with h5py.File(output_file, 'w') as hf:
        # Create a group for metadata
        meta = hf.create_group('metadata')
        meta.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta.attrs['total_windows'] = len(helper_features)
        meta.attrs['total_factors'] = len(factor_columns)
        meta.attrs['num_helpers'] = len(next(iter(next(iter(helper_features.values())).values())))
        
        # Store factor columns
        meta.create_dataset('factor_columns', data=np.array(factor_columns, dtype='S'))
        
        # Create a group for helper features
        helpers_group = hf.create_group('helper_features')
        
        # Loop through each window
        for window_id, window_helpers in helper_features.items():
            window_group = helpers_group.create_group(f'window_{window_id}')
            
            # Store helper features for each factor
            for factor, helpers in window_helpers.items():
                # Store helper factor names and correlation values
                helper_names = [h[0] for h in helpers]
                helper_corrs = [h[1] for h in helpers]
                
                factor_group = window_group.create_group(factor.replace('/', '_'))  # Replace / in factor names
                factor_group.create_dataset('names', data=np.array(helper_names, dtype='S'))
                factor_group.create_dataset('correlations', data=np.array(helper_corrs))

def save_sample_to_excel(helper_features, window_schedule, output_file, num_windows=5, num_factors=10):
    """
    Save a sample of helper features to Excel for easy inspection.
    
    Parameters:
    -----------
    helper_features : dict
        Nested dictionary with window IDs and factors as keys and list of helper features as values.
    window_schedule : pandas.DataFrame
        DataFrame containing window schedule information.
    output_file : str
        Path to output Excel file.
    num_windows : int
        Number of windows to sample (default: 5).
    num_factors : int
        Number of factors to sample for each window (default: 10).
    """
    # Sample windows evenly from all windows
    all_window_ids = sorted(helper_features.keys())
    if len(all_window_ids) > num_windows:
        sample_window_ids = np.linspace(min(all_window_ids), max(all_window_ids), num_windows, dtype=int)
    else:
        sample_window_ids = all_window_ids
    
    # Prepare data for Excel
    excel_data = []
    
    for window_id in sample_window_ids:
        if window_id not in helper_features:
            continue
            
        window_info = window_schedule[window_schedule['Window_ID'] == window_id].iloc[0]
        window_helpers = helper_features[window_id]
        
        # Sample factors
        all_factors = list(window_helpers.keys())
        if len(all_factors) > num_factors:
            # Ensure window_id is a proper integer for the seed
            random.seed(int(window_id))  # Cast to int to avoid potential issues
            sample_factors = random.sample(all_factors, num_factors)
        else:
            sample_factors = all_factors
            
        for factor in sample_factors:
            # Helper feature details
            helper_with_corr = window_helpers[factor]
            
            for i, (helper, corr) in enumerate(helper_with_corr, 1):
                excel_data.append({
                    'Window_ID': window_id,
                    'Training_Start_Date': window_info['Training_Start_Date'],
                    'Training_End_Date': window_info['Training_End_Date'],
                    'Target_Factor': factor,
                    'Helper_Rank': i,
                    'Helper_Factor': helper,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(excel_data)
    df = df.sort_values(['Window_ID', 'Target_Factor', 'Helper_Rank'])
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Helper Features Sample')
        
        # Add a summary sheet
        summary_data = {
            'Metric': [
                'Total Windows', 
                'Sampled Windows',
                'Factors per Window',
                'Sampled Factors per Window',
                'Helper Features per Factor',
                'Sample Generated Date'
            ],
            'Value': [
                len(all_window_ids),
                len(sample_window_ids),
                len(next(iter(helper_features.values()))),
                num_factors,
                len(next(iter(next(iter(helper_features.values())).values()))),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')

def create_helper_feature_visualizations(helper_features, window_schedule, factor_columns, output_file):
    """
    Create visualizations of helper feature selection patterns.
    
    Parameters:
    -----------
    helper_features : dict
        Nested dictionary with window IDs and factors as keys and list of helper features as values.
    window_schedule : pandas.DataFrame
        DataFrame containing window schedule information.
    factor_columns : list
        List of factor column names.
    output_file : str
        Path to output PDF file.
    """
    # Set plot style
    plt.style.use('ggplot')
    sns.set(font_scale=0.8)
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Helper Feature Selection Visualizations\nFactor Return Forecasting Project',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        plt.text(0.5, 0.3, f'Total Windows: {len(helper_features)}\nTotal Factors: {len(factor_columns)}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        pdf.savefig()
        plt.close()
        
        # === Analysis 1: Most Frequently Selected Helper Features ===
        # Count how often each factor appears as a helper across all windows and target factors
        helper_counts = Counter()
        
        for window_id, window_helpers in helper_features.items():
            for factor, helpers in window_helpers.items():
                for helper, _ in helpers:
                    helper_counts[helper] += 1
        
        # Create bar chart of top helpers
        plt.figure(figsize=(10, 8))
        top_n = 20
        most_common = helper_counts.most_common(top_n)
        factors = [item[0] for item in most_common]
        counts = [item[1] for item in most_common]
        
        plt.barh(range(len(counts)), counts, align='center')
        plt.yticks(range(len(counts)), factors)
        plt.xlabel('Number of Times Selected as Helper')
        plt.title(f'Top {top_n} Most Frequently Selected Helper Factors (Across All Windows)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 2: Helper Feature Network for a Sample Window ===
        # Select a middle window for network visualization
        middle_window_id = sorted(helper_features.keys())[len(helper_features)//2]
        middle_window_helpers = helper_features[middle_window_id]
        
        # Create a graph visualization showing helper relationships
        plt.figure(figsize=(10, 10))
        
        # This is just a simplified placeholder for a more complex network visualization
        # In a real network visualization, you would use networkx or another graph library
        # Here I'm just creating a dependency matrix
        num_sample_factors = min(30, len(factor_columns))
        sample_factors = random.sample(factor_columns, num_sample_factors)
        
        # Create an adjacency matrix 
        adj_matrix = np.zeros((num_sample_factors, num_sample_factors))
        factor_indices = {factor: i for i, factor in enumerate(sample_factors)}
        
        for i, factor in enumerate(sample_factors):
            if factor not in middle_window_helpers:
                continue
                
            helpers = [h[0] for h in middle_window_helpers[factor]]
            for helper in helpers:
                if helper in factor_indices:
                    j = factor_indices[helper]
                    adj_matrix[i, j] = 1
        
        # Plot the matrix
        plt.imshow(adj_matrix, cmap='Blues')
        plt.colorbar(label='Helper Relationship')
        plt.xticks(range(num_sample_factors), sample_factors, rotation=90)
        plt.yticks(range(num_sample_factors), sample_factors)
        plt.title(f'Helper Relationships for Window {middle_window_id} (Sample of {num_sample_factors} Factors)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 3: Correlation Strength Distribution ===
        # Analyze the strength of correlations for selected helpers
        all_correlations = []
        
        for window_id, window_helpers in helper_features.items():
            for factor, helpers in window_helpers.items():
                for _, corr in helpers:
                    all_correlations.append(abs(corr))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(all_correlations, bins=50, kde=True)
        plt.xlabel('Absolute Correlation')
        plt.ylabel('Frequency')
        plt.title('Distribution of Absolute Correlation Values for Selected Helper Features')
        plt.axvline(x=np.mean(all_correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_correlations):.3f}')
        plt.axvline(x=np.median(all_correlations), color='green', linestyle='--', 
                   label=f'Median: {np.median(all_correlations):.3f}')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # === Analysis 4: Helper Feature Stability Over Time ===
        # Analyze how helpers change over time for a sample of factors
        num_sample_factors = 5
        sample_factors = random.sample(factor_columns, num_sample_factors)
        
        # For each sample factor, track its top 3 helpers across windows
        for sample_factor in sample_factors:
            # Track top 3 helpers for this factor across all windows
            plt.figure(figsize=(12, 7))
            
            # Get top 3 helpers for each window
            helper_over_time = []
            window_dates = []
            
            for window_id in sorted(helper_features.keys()):
                window_helpers = helper_features[window_id]
                if sample_factor not in window_helpers:
                    continue
                    
                # Get top 3 helpers
                top_helpers = window_helpers[sample_factor][:3]
                helper_over_time.append([h[0] for h in top_helpers])
                
                # Get window end date for x-axis
                window_info = window_schedule[window_schedule['Window_ID'] == window_id].iloc[0]
                window_dates.append(window_info['Training_End_Date'])
            
            # Convert to array for easier manipulation
            helper_array = np.array(helper_over_time)
            
            # Plot lines for each helper position
            for i in range(3):
                # Count occurrences of each helper in this position
                if helper_array.shape[1] <= i:
                    continue
                    
                unique_helpers = np.unique(helper_array[:, i])
                for helper in unique_helpers:
                    # Create a mask where this helper appears at this position
                    mask = helper_array[:, i] == helper
                    plt.plot(np.array(window_dates)[mask], 
                             np.where(mask)[0], 
                             'o-', 
                             label=f"{helper} (Rank {i+1})" if sum(mask) > 5 else "_nolegend_")
            
            plt.title(f'Helper Feature Stability for Target Factor: {sample_factor}')
            plt.xlabel('Window End Date')
            plt.ylabel('Window Index')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # === Analysis 5: Correlation Strength by Window ===
        # Analyze how correlation strength changes over time
        mean_abs_corr_by_window = []
        window_ids = []
        
        for window_id in sorted(helper_features.keys()):
            window_helpers = helper_features[window_id]
            
            # Collect all correlation values for this window
            window_corrs = []
            for factor, helpers in window_helpers.items():
                for _, corr in helpers:
                    window_corrs.append(abs(corr))
            
            mean_abs_corr_by_window.append(np.mean(window_corrs))
            window_ids.append(window_id)
        
        plt.figure(figsize=(10, 6))
        
        # Get training end dates for each window
        window_dates = [window_schedule[window_schedule['Window_ID'] == w_id].iloc[0]['Training_End_Date'] 
                       for w_id in window_ids]
        
        plt.plot(window_dates, mean_abs_corr_by_window, marker='o')
        plt.xlabel('Window End Date')
        plt.ylabel('Mean Absolute Correlation')
        plt.title('Average Strength of Helper Correlations Over Time')
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Add a summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Summary of Helper Feature Analysis:\n\n'
                          f'Total Windows: {len(helper_features)}\n'
                          f'Total Factors: {len(factor_columns)}\n'
                          f'Helper Features per Factor: {len(next(iter(next(iter(helper_features.values())).values())))}\n\n'
                          f'Overall Correlation Statistics:\n'
                          f'  Mean Absolute Correlation: {np.mean(all_correlations):.3f}\n'
                          f'  Median Absolute Correlation: {np.median(all_correlations):.3f}\n'
                          f'  Min Absolute Correlation: {np.min(all_correlations):.3f}\n'
                          f'  Max Absolute Correlation: {np.max(all_correlations):.3f}\n\n'
                          f'Top 5 Most Frequently Selected Helper Factors:\n' + 
                          '\n'.join([f'  {i+1}. {factor} ({count} times)' 
                                     for i, (factor, count) in enumerate(helper_counts.most_common(5))]),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()

# Section: Main Script Logic
def main():
    print("=== Step 7: Select Helper Features ===")
    
    # --- Step 7.1: Load Data ---
    print("--- 7.1 Loading Data ---")
    
    # Define file paths
    input_corr_file = os.path.join("output", "S6_Rolling_Correlations.h5")
    input_factor_file = os.path.join("output", "S2_T2_Optimizer_with_MA.xlsx")
    input_window_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output files with S7 prefix in output directory
    output_helper_file = os.path.join(output_dir, "S7_Helper_Features.h5")
    output_sample_file = os.path.join(output_dir, "S7_Helper_Features_Sample.xlsx")
    output_viz_file = os.path.join(output_dir, "S7_Helper_Features_Visualization.pdf")
    
    try:
        # Load window schedule
        window_schedule = pd.read_excel(input_window_file)
        print(f"Loaded window schedule from {input_window_file}")
        
        # Load correlation matrices
        correlations_dict, factor_columns = load_correlations_from_h5(input_corr_file)
        print(f"Loaded correlation matrices for {len(correlations_dict)} windows from {input_corr_file}")
        print(f"Found {len(factor_columns)} factors in correlation matrices")
        
        # Quick check on factor data file (just to verify it exists)
        if os.path.exists(input_factor_file):
            print(f"Verified factor data file exists: {input_factor_file}")
        else:
            print(f"Warning: Factor data file not found: {input_factor_file}")
            
    except Exception as e:
        print(f"Error loading input files: {e}")
        sys.exit(1)
    
    # --- Step 7.2: Select Helper Features ---
    print("\n--- 7.2 Selecting Helper Features ---")
    
    # Define number of helper features to select
    num_helpers = 10
    print(f"Selecting top {num_helpers} most correlated helper features for each factor")
    
    # Process each window
    helper_features = select_helper_features(correlations_dict, factor_columns, num_helpers)
    
    print(f"Selected helper features for {len(helper_features)} windows")
    
    # --- Step 7.3: Save Helper Features ---
    print("\n--- 7.3 Saving Helper Features ---")
    
    # Save to HDF5 file
    save_helper_features_to_h5(helper_features, factor_columns, output_helper_file)
    print(f"Saved helper features to {output_helper_file}")
    
    # Save sample to Excel for easy inspection
    save_sample_to_excel(helper_features, window_schedule, output_sample_file)
    print(f"Saved helper features sample to {output_sample_file}")
    
    # --- Step 7.4: Generate Visualizations ---
    print("\n--- 7.4 Generating Visualizations ---")
    
    create_helper_feature_visualizations(helper_features, window_schedule, factor_columns, output_viz_file)
    print(f"Generated helper feature visualizations in {output_viz_file}")
    
    # --- Step 7.5: Summary ---
    print("\n--- 7.5 Summary ---")
    
    # Calculate some statistics
    all_corrs = []
    for window_id, window_helpers in helper_features.items():
        for factor, helpers in window_helpers.items():
            for _, corr in helpers:
                all_corrs.append(abs(corr))
    
    print(f"Total helper-factor relationships: {len(all_corrs)}")
    print(f"Average absolute correlation of selected helpers: {np.mean(all_corrs):.3f}")
    print(f"Median absolute correlation of selected helpers: {np.median(all_corrs):.3f}")
    print(f"Output files:")
    print(f"  - HDF5 file with all helper features: {output_helper_file}")
    print(f"  - Excel file with sample helper features: {output_sample_file}")
    print(f"  - PDF with helper feature visualizations: {output_viz_file}")
    
    print("\nStep 7 completed successfully!")

if __name__ == "__main__":
    main() 