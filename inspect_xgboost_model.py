#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_xgboost_model.py

This script loads and inspects an XGBoost model saved in joblib format.
It displays detailed information about the model's parameters, feature importances,
and tree structure.

Usage:
    python inspect_xgboost_model.py <path_to_model_file>

Example:
    python inspect_xgboost_model.py output/S11A_XGBoost_Models/window_1/factor_12-1MTR_CS.joblib
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import re
from pathlib import Path

# Mapping of features for specific models (add more as needed)
FEATURE_MAPPINGS = {
    "12-1MTR_CS": [
        "12-1MTR_CS_1m",
        "12-1MTR_CS_3m",
        "12-1MTR_CS_12m", 
        "12-1MTR_CS_60m"
    ]
}

def load_model(model_path):
    """Load XGBoost model from joblib file."""
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)

def print_model_info(model):
    """Print basic information about the model."""
    print("\n" + "="*80)
    print("MODEL BASIC INFORMATION")
    print("="*80)
    print(f"XGBoost Model Type: {type(model)}")
    print(f"Model Version: {model.__version__ if hasattr(model, '__version__') else 'Unknown'}")
    print(f"Number of trees (n_estimators): {model.n_estimators}")
    print(f"Actual trees used (best_iteration): {model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators}")
    print(f"Number of features: {len(model.feature_importances_)}")
    
def print_model_params(model):
    """Print detailed model parameters."""
    print("\n" + "="*80)
    print("MODEL PARAMETERS")
    print("="*80)
    # Get parameters sorted by name
    params = model.get_params()
    for key in sorted(params.keys()):
        print(f"{key}: {params[key]}")

def get_feature_names(model, model_path):
    """
    Attempt to get feature names using multiple methods:
    1. From model if available
    2. From feature sets H5 file
    3. From predefined mappings
    4. Generate plausible names based on project documentation
    """
    feature_names = None
    n_features = len(model.feature_importances_)
    
    # Extract factor ID from model path
    factor_id = None
    match = re.search(r'factor_([^\.]+)\.joblib', model_path)
    if match:
        factor_id = match.group(1)
        print(f"Extracted factor_id={factor_id} from model path")
    
    # Method 1: Try to get feature names from model booster if available
    try:
        booster = model.get_booster()
        if hasattr(booster, 'feature_names') and booster.feature_names:
            feature_names = booster.feature_names
            print(f"Feature names retrieved from model booster: {feature_names}")
            return feature_names
    except Exception as e:
        print(f"Could not retrieve feature names from booster: {e}")
    
    # Method 2: Check if we have a predefined mapping for this factor
    if factor_id and factor_id in FEATURE_MAPPINGS:
        feature_names = FEATURE_MAPPINGS[factor_id]
        print(f"Using predefined feature names for {factor_id}")
        
        # Check if the number of features matches
        if len(feature_names) != n_features:
            print(f"Warning: Predefined mapping has {len(feature_names)} features, but model has {n_features}")
            if len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
            else:
                # Add placeholder names for the missing features
                for i in range(len(feature_names), n_features):
                    feature_names.append(f"Unknown_Feature_{i}")
        
        return feature_names
    
    # Method 3: Try to extract from feature sets H5 file
    if factor_id:
        try:
            # Extract window ID from path
            window_id = None
            path_parts = model_path.split('/')
            for part in path_parts:
                if part.startswith('window_'):
                    window_id = part.replace('window_', '')
                    break
            
            if window_id:
                print(f"Extracted window_id={window_id} from model path")
                
                # Try to get feature names from feature sets H5 file
                feature_sets_path = Path("output") / "S8_Feature_Sets.h5"
                if feature_sets_path.exists():
                    with h5py.File(feature_sets_path, 'r') as h5f:
                        try:
                            # Try different possible paths
                            possible_paths = [
                                f"feature_sets/window_{window_id}/training/{factor_id}/X/columns",
                                f"window_{window_id}/training/{factor_id}/X/columns"
                            ]
                            
                            for path in possible_paths:
                                if path in h5f:
                                    columns_data = h5f[path]
                                    columns = columns_data[:]
                                    
                                    # If it's a byte string array, decode to UTF-8
                                    if columns_data.dtype.kind in ['S', 'O']:
                                        feature_names = [col.decode('utf-8') if isinstance(col, bytes) else col for col in columns]
                                    else:
                                        feature_names = list(columns)
                                    
                                    print(f"Feature names retrieved from H5 file: {feature_names}")
                                    return feature_names
                        except Exception as e:
                            print(f"Error extracting feature names from H5 file: {e}")
        except Exception as e:
            print(f"Error trying to extract feature names from H5: {e}")
    
    # Method 4: Generate plausible feature names based on project documentation
    # According to the project docs, each model uses 14 features:
    # - 4 features from the factor itself (1-month, 3-month, 12-month, and 60-month MAs)
    # - 10 features from helper factors (60-month MAs of top 10 correlated factors)
    factor_base = factor_id if factor_id else 'Unknown-Factor'
    
    # Clean up factor_id to get base name
    for suffix in ['_CS', '_TS', '_3m', '_12m', '_60m']:
        factor_base = factor_base.replace(suffix, '')
    
    # Create plausible feature names
    generated_names = []
    
    # Factor's own MAs
    generated_names.append(f"{factor_base}_1m_MA")
    generated_names.append(f"{factor_base}_3m_MA")
    generated_names.append(f"{factor_base}_12m_MA")
    generated_names.append(f"{factor_base}_60m_MA")
    
    # Helper factors' 60m MAs
    for i in range(1, 11):
        generated_names.append(f"Helper_{i}_60m_MA")
    
    # Ensure we have the right number of features
    if len(generated_names) != n_features:
        print(f"Warning: Generated {len(generated_names)} names but model has {n_features} features")
        # Adjust by adding or removing helper features
        if len(generated_names) < n_features:
            for i in range(len(generated_names), n_features):
                generated_names.append(f"Unknown_Feature_{i}")
        else:
            generated_names = generated_names[:n_features]
    
    print(f"Generated plausible feature names based on project documentation")
    return generated_names

def analyze_feature_importance(model, model_path):
    """Analyze and display feature importance."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Get feature names
    feature_names = get_feature_names(model, model_path)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature ranking:")
    for i, idx in enumerate(indices):
        feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
        print(f"  {i+1}. {feature_name}: {importances[idx]:.6f} ({importances[idx]/sum(importances)*100:.2f}%)")
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    
    # Use feature names for x-tick labels if available
    if feature_names:
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    else:
        plt.xticks(range(len(importances)), indices)
    
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Use factor name in output filename if available
    factor_str = ""
    if "_factor_" in model_path:
        factor_match = re.search(r'_factor_([^\.]+)', model_path)
        if factor_match:
            factor_str = f"_{factor_match.group(1)}"
    
    plt.savefig(output_dir / f"feature_importance{factor_str}.pdf")
    print(f"Feature importance plot saved to: {output_dir}/feature_importance{factor_str}.pdf")
    
    # Return feature names for other functions to use
    return feature_names

def examine_tree_structure(model, feature_names=None):
    """Examine the structure of trees in the model."""
    print("\n" + "="*80)
    print("TREE STRUCTURE ANALYSIS")
    print("="*80)
    
    booster = model.get_booster()
    
    # Get number of trees
    num_trees = booster.num_boosted_rounds()
    print(f"Number of boosted rounds: {num_trees}")
    
    # Get a sample of trees (first, middle, last)
    tree_indices = [0, num_trees//2, num_trees-1] if num_trees > 2 else range(num_trees)
    
    print("\nSample tree structures:")
    
    # Print information for a sample of trees
    for i in tree_indices:
        if i < num_trees:  # Ensure index is valid
            print(f"\nTree {i}:")
            tree_dump = booster.get_dump()[i]
            
            # Replace feature indices with names if available
            if feature_names:
                for idx, name in enumerate(feature_names):
                    tree_dump = tree_dump.replace(f"f{idx}", name)
            
            lines = tree_dump.split('\n')
            # Print first few lines of the tree
            for j, line in enumerate(lines[:10]):
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... ({len(lines)-10} more lines)")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_xgboost_model.py <path_to_model_file>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Load the model
    model = load_model(model_path)
    
    # Print model information
    print_model_info(model)
    print_model_params(model)
    
    # Get feature importances and names
    feature_names = analyze_feature_importance(model, model_path)
    
    # Examine tree structure
    examine_tree_structure(model, feature_names)
    
    print("\nModel inspection complete.")

if __name__ == "__main__":
    main() 