import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import h5py
import warnings
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

def extract_results_from_h5(h5_file):
    """Extract model results from the HDF5 file for summary generation."""
    all_results = {}
    
    with h5py.File(h5_file, 'r') as hf:
        if 'models' not in hf:
            raise ValueError("Invalid HDF5 structure: 'models' group not found")
        
        models_group = hf['models']
        window_count = 0
        
        # Process a limited number of windows for testing
        for window_name in list(models_group.keys()):
            window_id = int(window_name.split('_')[1])
            window_group = models_group[window_name]
            window_count += 1
            
            # Initialize results for this window
            all_results[window_id] = {}
            
            # Process a limited number of factors per window
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

def create_training_summary(all_results, window_schedule):
    """Create a summary of training results for analysis and visualization."""
    print("Creating training summary...")
    
    # Print window schedule columns for debugging
    print(f"Window schedule columns: {window_schedule.columns.tolist()}")
    
    # Check window_schedule column names and make it case-insensitive
    column_map = {}
    for column in window_schedule.columns:
        if column.lower() == 'window_id':
            column_map['window_id'] = column
        elif column.lower() == 'prediction_index':
            column_map['pred_month'] = column
        elif column.lower() == 'prediction_date':
            column_map['pred_date'] = column
        elif column.lower() == 'training_start_date':
            column_map['training_start'] = column
        elif column.lower() == 'training_end_date':
            column_map['training_end'] = column
    
    # Initialize lists to store results
    rows = []
    
    # Model types
    model_types = ['OLS', 'Ridge', 'Lasso', 'NNLS']
    
    # Iterate through windows
    for window_id, window_results in all_results.items():
        # Get window information
        window_info_rows = window_schedule[window_schedule[column_map['window_id']] == window_id]
        
        if len(window_info_rows) == 0:
            print(f"Warning: Window ID {window_id} not found in window schedule")
            continue
        
        window_info = window_info_rows.iloc[0]
        
        # Iterate through factors
        for factor_id, factor_results in window_results.items():
            # Iterate through model types
            for model_type in model_types:
                if model_type not in factor_results['train_metrics'] or model_type not in factor_results['val_metrics']:
                    continue
                
                # Get metrics
                train_metrics = factor_results['train_metrics'][model_type]
                val_metrics = factor_results['val_metrics'][model_type]
                
                # Get feature importance
                feature_importance = factor_results['feature_importance'][model_type]
                
                # Find top features
                if len(feature_importance) >= 3:
                    top_features = feature_importance.iloc[:3]['feature'].tolist()
                else:
                    top_features = feature_importance['feature'].tolist()
                
                top_features_str = ', '.join(top_features)
                
                # Add row
                row_data = {
                    'window_id': window_id,
                    'pred_month': window_info[column_map['pred_month']],
                    'pred_date': window_info[column_map['pred_date']],
                    'training_start': window_info[column_map['training_start']],
                    'training_end': window_info[column_map['training_end']],
                    'factor_id': factor_id,
                    'model_type': model_type,
                    'top_features': top_features_str
                }
                
                # Add training metrics
                for metric_name, metric_value in train_metrics.items():
                    row_data[f'train_{metric_name}'] = metric_value
                
                # Add validation metrics
                for metric_name, metric_value in val_metrics.items():
                    row_data[f'val_{metric_name}'] = metric_value
                
                rows.append(row_data)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(rows)
    
    # If summary DataFrame is empty, return empty summaries
    if len(summary_df) == 0:
        print("Warning: No summary data generated")
        return {
            'full': pd.DataFrame(),
            'model': pd.DataFrame(),
            'factor': pd.DataFrame(),
            'window': pd.DataFrame()
        }
    
    # Create aggregated summaries
    metrics_to_agg = []
    for prefix in ['train_', 'val_']:
        for metric in ['rmse', 'mae', 'r2', 'directional_accuracy']:
            col = f'{prefix}{metric}'
            if col in summary_df.columns:
                metrics_to_agg.append(col)
    
    agg_dict = {metric: 'mean' for metric in metrics_to_agg}
    
    model_summary = summary_df.groupby('model_type').agg(agg_dict).reset_index()
    
    factor_summary = summary_df.groupby(['factor_id', 'model_type']).agg(agg_dict).reset_index()
    
    window_summary = summary_df.groupby(['window_id', 'pred_date', 'model_type']).agg(agg_dict).reset_index()
    
    # Create dictionary of summary DataFrames
    summary = {
        'full': summary_df,
        'model': model_summary,
        'factor': factor_summary,
        'window': window_summary
    }
    
    return summary

def save_summary_to_excel(summary, output_file):
    """Save training summary to an Excel file."""
    print(f"Saving summary to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save each summary DataFrame to a separate sheet
        summary['full'].to_excel(writer, sheet_name='Full_Results', index=False)
        summary['model'].to_excel(writer, sheet_name='Model_Summary', index=False)
        summary['factor'].to_excel(writer, sheet_name='Factor_Summary', index=False)
        summary['window'].to_excel(writer, sheet_name='Window_Summary', index=False)
        
        # Try to create a pivot table if possible
        if 'factor_id' in summary['factor'].columns and 'model_type' in summary['factor'].columns and 'val_r2' in summary['factor'].columns:
            try:
                # Create a pivot table of model performance by factor
                pivot_df = pd.pivot_table(
                    summary['factor'],
                    values='val_r2',
                    index='factor_id',
                    columns='model_type'
                )
                
                # Add a "best model" column
                pivot_df['best_model'] = pivot_df.idxmax(axis=1)
                
                # Save pivot table
                pivot_df.to_excel(writer, sheet_name='Model_by_Factor')
            except Exception as e:
                print(f"Could not create pivot table: {e}")

def create_visualizations(summary, output_file):
    """Create visualizations of training results and save to PDF."""
    print(f"Creating visualizations and saving to {output_file}...")
    
    with PdfPages(output_file) as pdf:
        # Set style
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Check if we have the necessary columns
        required_cols = ['model_type', 'train_rmse', 'val_rmse', 'train_r2', 'val_r2']
        if not all(col in summary['model'].columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in summary['model'].columns]
            print(f"Warning: Missing required columns for visualization: {missing_cols}")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Insufficient data for visualization.\nMissing columns: {', '.join(missing_cols)}", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            return
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training RMSE
        sns.barplot(x='model_type', y='train_rmse', data=summary['model'], ax=axes[0, 0])
        axes[0, 0].set_title('Training RMSE by Model Type')
        axes[0, 0].set_xlabel('Model Type')
        axes[0, 0].set_ylabel('RMSE')
        
        # Validation RMSE
        sns.barplot(x='model_type', y='val_rmse', data=summary['model'], ax=axes[0, 1])
        axes[0, 1].set_title('Validation RMSE by Model Type')
        axes[0, 1].set_xlabel('Model Type')
        axes[0, 1].set_ylabel('RMSE')
        
        # Training R²
        sns.barplot(x='model_type', y='train_r2', data=summary['model'], ax=axes[1, 0])
        axes[1, 0].set_title('Training R² by Model Type')
        axes[1, 0].set_xlabel('Model Type')
        axes[1, 0].set_ylabel('R²')
        
        # Validation R²
        sns.barplot(x='model_type', y='val_r2', data=summary['model'], ax=axes[1, 1])
        axes[1, 1].set_title('Validation R² by Model Type')
        axes[1, 1].set_xlabel('Model Type')
        axes[1, 1].set_ylabel('R²')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Only create additional visualizations if we have directional accuracy
        if 'val_directional_accuracy' in summary['model'].columns:
            # 2. Directional Accuracy by Model Type
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model_type', y='val_directional_accuracy', data=summary['model'])
            plt.title('Validation Directional Accuracy by Model Type')
            plt.xlabel('Model Type')
            plt.ylabel('Directional Accuracy (%)')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 3. Distribution of Validation R² by Model Type
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='model_type', y='val_r2', data=summary['full'])
            plt.title('Distribution of Validation R² by Model Type')
            plt.xlabel('Model Type')
            plt.ylabel('Validation R²')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 4. Distribution of Validation RMSE by Model Type
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='model_type', y='val_rmse', data=summary['full'])
            plt.title('Distribution of Validation RMSE by Model Type')
            plt.xlabel('Model Type')
            plt.ylabel('Validation RMSE')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 5. Model Performance Over Time (Windows)
            if 'pred_date' in summary['window'].columns and pd.api.types.is_datetime64_any_dtype(summary['window']['pred_date']):
                plt.figure(figsize=(15, 8))
                
                # Plot validation R² over time
                for model_type in summary['window']['model_type'].unique():
                    model_data = summary['window'][summary['window']['model_type'] == model_type]
                    plt.plot(model_data['pred_date'], model_data['val_r2'], label=model_type)
                
                plt.title('Validation R² Over Time by Model Type')
                plt.xlabel('Prediction Date')
                plt.ylabel('Validation R²')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

def main():
    """Main execution function."""
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