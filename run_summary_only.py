import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import sys
import h5py
import warnings
import json
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Import necessary functions from Step_11B
from Step_11B_Train_Linear_Models import create_training_summary, save_summary_to_excel, create_visualizations

def extract_results_from_h5(h5_file):
    """
    Extract model results from the HDF5 file for summary generation.
    
    Parameters:
    -----------
    h5_file : str
        Path to the HDF5 file with trained models.
        
    Returns:
    --------
    dict
        Dictionary of model results.
    """
    all_results = {}
    
    with h5py.File(h5_file, 'r') as hf:
        if 'models' not in hf:
            raise ValueError("Invalid HDF5 structure: 'models' group not found")
        
        models_group = hf['models']
        window_count = 0
        
        # Process windows
        for window_name in list(models_group.keys()):
            window_id = int(window_name.split('_')[1])
            window_group = models_group[window_name]
            window_count += 1
            
            print(f"Extracting data for window {window_id}...")
            
            # Initialize results for this window
            all_results[window_id] = {}
            
            # Process factors
            for factor_key in list(window_group.keys()):
                factor_group = window_group[factor_key]
                
                # Get original factor ID
                if 'original_factor_id' in factor_group.attrs:
                    factor_id = factor_group.attrs['original_factor_id']
                else:
                    factor_id = factor_key.replace('_', '/')
                
                # Initialize results for this factor
                factor_results = {
                    'window_id': window_id,
                    'factor_id': factor_id,
                    'models': {},
                    'train_metrics': {},
                    'val_metrics': {},
                    'feature_importance': {}
                }
                
                # Extract models and metrics
                if 'models' in factor_group and 'metrics' in factor_group:
                    models_subgroup = factor_group['models']
                    metrics_subgroup = factor_group['metrics']
                    
                    # Model types
                    model_types = ['OLS', 'Ridge', 'Lasso', 'NNLS']
                    
                    for model_type in model_types:
                        if model_type in models_subgroup and model_type in metrics_subgroup:
                            model_group = models_subgroup[model_type]
                            metric_group = metrics_subgroup[model_type]
                            
                            # Extract model
                            if model_type == 'NNLS':
                                coefficients = model_group['coefficients'][:]
                                feature_names = [name.decode('utf-8') for name in model_group['feature_names'][:]]
                                model = {
                                    'coefficients': coefficients,
                                    'feature_names': feature_names
                                }
                            else:
                                coef = model_group['coef'][:]
                                intercept = model_group['intercept'][:]
                                model = {
                                    'coef': coef,
                                    'intercept': intercept
                                }
                            
                            # Extract train metrics
                            train_metrics = {}
                            if 'train' in metric_group:
                                train_group = metric_group['train']
                                for metric_name in train_group.attrs:
                                    train_metrics[metric_name] = train_group.attrs[metric_name]
                            
                            # Extract val metrics
                            val_metrics = {}
                            if 'val' in metric_group:
                                val_group = metric_group['val']
                                for metric_name in val_group.attrs:
                                    val_metrics[metric_name] = val_group.attrs[metric_name]
                            
                            # Create feature importance
                            if model_type == 'NNLS':
                                importance_data = {
                                    'feature': feature_names,
                                    'coefficient': coefficients
                                }
                            else:
                                if 'feature_names' in model_group:
                                    feature_names = [name.decode('utf-8') for name in model_group['feature_names'][:]]
                                else:
                                    feature_names = [f'Feature_{i}' for i in range(len(coef))]
                                
                                importance_data = {
                                    'feature': feature_names,
                                    'coefficient': coef
                                }
                            
                            importance_df = pd.DataFrame(importance_data)
                            importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
                            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
                            importance_df = importance_df.drop('abs_coefficient', axis=1)
                            
                            # Store results
                            factor_results['models'][model_type] = model
                            factor_results['train_metrics'][model_type] = train_metrics
                            factor_results['val_metrics'][model_type] = val_metrics
                            factor_results['feature_importance'][model_type] = importance_df
                
                # Add factor results to window results
                all_results[window_id][factor_id] = factor_results
    
    print(f"Processed {window_count} windows")
    return all_results

def main():
    """
    Main execution function.
    """
    start_time = time.time()
    print("=== Generating summaries from Step 11B results ===")
    
    # Input and output files
    h5_input_file = os.path.join("output", "S11B_Linear_Models.h5")
    excel_output_file = os.path.join("output", "S11B_Linear_Models_Training_Summary.xlsx")
    visualization_output_file = os.path.join("output", "S11B_Linear_Models_Training_Visualization.pdf")
    
    # Load window schedule
    window_schedule_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    if not os.path.exists(window_schedule_file):
        raise FileNotFoundError(f"Window schedule file not found: {window_schedule_file}")
    
    window_schedule = pd.read_excel(window_schedule_file)
    print(f"Loaded window schedule with {len(window_schedule)} windows")
    
    # Load model results from H5 file
    if not os.path.exists(h5_input_file):
        raise FileNotFoundError(f"Model file not found: {h5_input_file}")
    
    all_results = extract_results_from_h5(h5_input_file)
    print(f"Extracted results for {len(all_results)} windows")
    
    # Create training summary
    summary = create_training_summary(all_results, window_schedule)
    
    # Save summary to Excel
    save_summary_to_excel(summary, excel_output_file)
    
    # Create visualizations
    create_visualizations(summary, visualization_output_file)
    
    elapsed_time = time.time() - start_time
    print(f"=== Summary generation completed in {elapsed_time:.2f} seconds ===")
    print(f"Results saved to:")
    print(f"  - {excel_output_file}")
    print(f"  - {visualization_output_file}")

if __name__ == "__main__":
    main() 