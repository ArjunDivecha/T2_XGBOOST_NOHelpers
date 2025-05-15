'''
Step_9_Configure_XGBoost.py

This script defines the XGBoost configuration for the factor forecasting models. It creates
and saves standard XGBoost parameters to be used across all factor models.

----------------------------------------------------------------------------------------------------
INPUT FILES:
- None directly. Configuration is based on project requirements.

OUTPUT FILES:
- S9_XGBoost_Config.xlsx
  - Path: ./output/S9_XGBoost_Config.xlsx
  - Description: Contains XGBoost hyperparameters to be used in model training.
  - Format: Excel (.xlsx) with parameter name-value pairs.

- S9_XGBoost_Config.json
  - Path: ./output/S9_XGBoost_Config.json
  - Description: JSON format of XGBoost configuration for easier loading in Python.
  - Format: JSON file with parameter dictionary.

Version: 1.0
Last Updated: 2024-06-28
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import json
import os
import datetime
from pathlib import Path

# Section: Ensure Output Directory Exists
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Section: Define XGBoost Parameters
def define_xgboost_params():
    """
    Define the standard XGBoost parameters for factor forecasting models.
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    dict
        Dictionary of XGBoost parameters.
    """
    xgb_params = {
        # Primary Parameters
        'max_depth': 4,                # Tree depth: 4
        'n_estimators': 500,           # Trees: 500 with early stopping
        'learning_rate': 0.01,         # Learning rate: 0.01
        'subsample': 0.8,              # Subsample: 80%
        'colsample_bytree': 0.7,       # Feature sample: 70%
        
        # Additional Parameters
        'objective': 'reg:squarederror',  # For regression tasks
        'eval_metric': ['rmse', 'mae'],   # Metrics to evaluate during training
        'early_stopping_rounds': 50,      # Stop if no improvement after 50 rounds
        'verbose': False,                 # Less output during training
        
        # Additional Tuning Parameters
        'min_child_weight': 1,
        'gamma': 0,
        'alpha': 0,                     # L1 regularization
        'lambda': 1,                    # L2 regularization
        'random_state': 42              # For reproducibility
    }
    
    return xgb_params

# Section: Create Parameter Documentation
def create_param_documentation(xgb_params):
    """
    Create a documentation dataframe for the XGBoost parameters.
    
    Parameters:
    -----------
    xgb_params : dict
        Dictionary of XGBoost parameters.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with parameter names, values and descriptions.
    """
    param_descriptions = {
        'max_depth': 'Maximum depth of a tree (higher = more complex)',
        'n_estimators': 'Number of trees to build (will use early stopping)',
        'learning_rate': 'Step size shrinkage used to prevent overfitting',
        'subsample': 'Fraction of samples used for fitting the trees',
        'colsample_bytree': 'Fraction of features used for building each tree',
        'objective': 'Specifies the learning task (regression)',
        'eval_metric': 'Metrics to be evaluated during validation',
        'early_stopping_rounds': 'Stop training if score doesn\'t improve',
        'verbose': 'Controls level of output during training',
        'min_child_weight': 'Minimum sum of instance weight in a child',
        'gamma': 'Minimum loss reduction required for a split',
        'alpha': 'L1 regularization term on weights',
        'lambda': 'L2 regularization term on weights',
        'random_state': 'Random number seed for reproducibility'
    }
    
    # Create documentation dataframe
    docs = []
    for param, value in xgb_params.items():
        docs.append({
            'Parameter': param,
            'Value': str(value),
            'Description': param_descriptions.get(param, 'No description available')
        })
    
    return pd.DataFrame(docs)

# Section: Save Configuration
def save_xgboost_config(xgb_params, param_docs):
    """
    Save the XGBoost configuration to output files.
    
    Parameters:
    -----------
    xgb_params : dict
        Dictionary of XGBoost parameters.
    param_docs : pandas.DataFrame
        DataFrame with parameter documentation.
    
    Returns:
    --------
    tuple
        Paths to the saved configuration files.
    """
    # Save as Excel
    excel_path = output_dir / "S9_XGBoost_Config.xlsx"
    param_docs.to_excel(excel_path, index=False)
    
    # Save as JSON
    json_path = output_dir / "S9_XGBoost_Config.json"
    with open(json_path, 'w') as f:
        json.dump(xgb_params, f, indent=4)
    
    return excel_path, json_path

# Section: Main Execution
def main():
    """Main execution function."""
    print("="*80)
    print("Step 9: Configure XGBoost")
    print("="*80)
    
    # Define XGBoost parameters
    print("\nDefining XGBoost parameters...")
    xgb_params = define_xgboost_params()
    
    # Create parameter documentation
    print("Creating parameter documentation...")
    param_docs = create_param_documentation(xgb_params)
    
    # Display configuration
    print("\nXGBoost Configuration:")
    print("-"*50)
    for param, value in xgb_params.items():
        print(f"{param:20}: {value}")
    
    # Save configuration
    excel_path, json_path = save_xgboost_config(xgb_params, param_docs)
    print("\nConfiguration saved to:")
    print(f"- {excel_path}")
    print(f"- {json_path}")
    
    print("\nStep 9 Complete: XGBoost Configuration\n")
    return xgb_params, param_docs

if __name__ == "__main__":
    main() 