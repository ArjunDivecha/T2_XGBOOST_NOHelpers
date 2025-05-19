def visualize_tuning_results(tuning_results, output_file, num_windows_used=1, num_factors_used=10):
    """
    Create visualizations of the tuning results.
    
    Parameters:
    -----------
    tuning_results : list
        List of tuning results.
    output_file : str
        Path to save the visualizations.
    num_windows_used : int, optional
        Number of windows used for tuning.
    num_factors_used : int, optional
        Number of factors used for tuning.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(tuning_results)
    
    if len(results_df) == 0:
        print("No results to visualize. Skipping visualization.")
        return
        
    # Sort by RMSE
    results_df = results_df.sort_values('avg_rmse')
    
    # Create a simple visualization with the top 10 results
    plt.figure(figsize=(12, 8))
    
    # Get top 10 results
    top10 = results_df.head(10).copy()
    
    # Create readable labels
    labels = []
    for i, row in top10.iterrows():
        combo = row['param_combo']
        label = f"Combo {i}: "
        for k, v in combo.items():
            label += f"{k}={v}, "
        label = label[:-2]  # Remove trailing comma and space
        if len(label) > 40:
            label = label[:37] + "..."
        labels.append(label)
    
    plt.barh(range(len(top10)), top10['avg_rmse'], color='skyblue')
    plt.yticks(range(len(top10)), labels)
    plt.xlabel('Average RMSE (lower is better)')
    plt.title(f'Top 10 Parameter Combinations\n{num_windows_used} Windows, {num_factors_used} Factors')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    
    # Also create a more detailed PDF report
    pdf_output = output_file.replace('.pdf', '_detailed.pdf')
    if not pdf_output.endswith('.pdf'):
        pdf_output += '_detailed.pdf'
    
    with PdfPages(pdf_output) as pdf:
        # Title page
        plt.figure(figsize=(10, 7))
        plt.axis('off')
        plt.text(0.5, 0.9, 'XGBoost Hyperparameter Tuning Results', 
                horizontalalignment='center', fontsize=20, fontweight='bold')
        plt.text(0.5, 0.8, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                horizontalalignment='center', fontsize=14)
        plt.text(0.5, 0.7, f'Windows Used: {num_windows_used}', 
                horizontalalignment='center', fontsize=14)
        plt.text(0.5, 0.6, f'Factors Used: {num_factors_used}', 
                horizontalalignment='center', fontsize=14)
        plt.text(0.5, 0.5, f'Total Parameter Combinations: {len(results_df)}', 
                horizontalalignment='center', fontsize=14)
        
        # Best Parameters
        best_row = results_df.iloc[0]
        plt.text(0.5, 0.4, f'Best RMSE: {best_row["avg_rmse"]:.4f}', 
                horizontalalignment='center', fontsize=14)
        plt.text(0.5, 0.35, f'Best R²: {best_row["avg_r2"]:.4f}', 
                horizontalalignment='center', fontsize=14)
        plt.text(0.5, 0.3, 'Best Parameters:', 
                horizontalalignment='center', fontsize=14)
        
        y_pos = 0.25
        for k, v in best_row['param_combo'].items():
            plt.text(0.5, y_pos, f'{k}: {v}', 
                    horizontalalignment='center', fontsize=12)
            y_pos -= 0.05
            
        pdf.savefig()
        plt.close()
        
        # Top 10 parameter combinations
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top10)), top10['avg_rmse'], color='skyblue')
        plt.yticks(range(len(top10)), labels)
        plt.xlabel('Average RMSE (lower is better)')
        plt.title('Top 10 Parameter Combinations')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Individual parameter impact
        if len(results_df) > 0 and 'param_combo' in results_df.columns:
            # Get first param combo to extract keys
            first_param = results_df.iloc[0]['param_combo']
            if isinstance(first_param, dict):
                param_names = list(first_param.keys())
                
                for param in param_names:
                    # Create a new DataFrame with extracted parameter values
                    param_df = pd.DataFrame({
                        'param_value': [row['param_combo'][param] for _, row in results_df.iterrows()],
                        'avg_rmse': results_df['avg_rmse'].values
                    })
                    
                    # Group by parameter value
                    grouped = param_df.groupby('param_value')['avg_rmse'].mean().reset_index()
                    grouped = grouped.sort_values('param_value')
                    
                    # Create plot
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(grouped)), grouped['avg_rmse'], color='lightblue')
                    plt.xticks(range(len(grouped)), [str(x) for x in grouped['param_value']])
                    plt.title(f'Impact of {param} on Average RMSE')
                    plt.xlabel(f'{param} Value')
                    plt.ylabel('Average RMSE (lower is better)')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add value labels
                    for i, v in enumerate(grouped['avg_rmse']):
                        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
                        
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
