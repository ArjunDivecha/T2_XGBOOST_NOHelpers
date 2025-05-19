#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_h5_deeper.py

This script inspects the 'columns' dataset in the feature sets H5 file,
which might contain the feature names for XGBoost models.

Usage:
    python inspect_h5_deeper.py <h5_file_path> <window_id> <factor_id>

Example:
    python inspect_h5_deeper.py output/S8_Feature_Sets.h5 1 "12-1MTR_CS"
"""

import sys
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def inspect_columns(h5_file, window_id, factor_id):
    """
    Inspect the 'columns' dataset which might contain feature names.
    """
    print(f"Looking for feature names in columns for window {window_id}, factor {factor_id}")
    
    # Try different possible paths
    base_paths = [
        f"feature_sets/window_{window_id}/training/{factor_id}",
        f"window_{window_id}/training/{factor_id}",
        f"feature_sets/window_{window_id}/prediction/{factor_id}",
        f"window_{window_id}/prediction/{factor_id}"
    ]
    
    found = False
    for base_path in base_paths:
        columns_path = f"{base_path}/X/columns"
        
        if columns_path in h5_file:
            print(f"Found columns at: {columns_path}")
            columns_data = h5_file[columns_path]
            
            print(f"Columns shape: {columns_data.shape}, dtype: {columns_data.dtype}")
            
            try:
                # Try to read the data
                columns = columns_data[:]
                
                # If it's a byte string array, decode to UTF-8
                if columns_data.dtype.kind == 'S' or columns_data.dtype.kind == 'O':
                    columns = [col.decode('utf-8') if isinstance(col, bytes) else col for col in columns]
                
                print(f"Columns data: {columns}")
                found = True
                return list(columns)  # Convert numpy array to list
            except Exception as e:
                print(f"Error reading columns data: {e}")
    
    if not found:
        print("No 'columns' dataset found for the specified window and factor.")
        return None

def collect_feature_info(h5_file, window_id, factor_id):
    """
    Collect information about features from multiple sources.
    """
    # Try to get columns directly
    columns = inspect_columns(h5_file, window_id, factor_id)
    
    # If columns not found, try other methods
    if columns is None:
        print("\nTrying to infer feature names from data structure...")
        
        # Check data dimensions
        data_path = f"feature_sets/window_{window_id}/training/{factor_id}/X/data"
        if data_path in h5_file:
            data = h5_file[data_path]
            print(f"X data shape: {data.shape}, Number of features: {data.shape[1]}")
            
            # Generate plausible feature names based on project documentation
            n_features = data.shape[1]
            print(f"\nGenerating plausible feature names for {n_features} features:")
            
            # According to project docs, each factor model uses 14 features:
            # - 4 features from the factor itself (1-month, 3-month, 12-month, and 60-month MAs)
            # - 10 features from helper factors (60-month MAs of top 10 correlated factors)
            
            # Clean up factor_id to get base name
            base_factor = factor_id
            for suffix in ['_CS', '_TS', '_3m', '_12m', '_60m']:
                base_factor = base_factor.replace(suffix, '')
            
            print(f"Base factor name: {base_factor}")
            
            # Create plausible feature names
            generated_names = []
            
            # Factor's own MAs
            generated_names.append(f"{base_factor}_1m_MA")
            generated_names.append(f"{base_factor}_3m_MA")
            generated_names.append(f"{base_factor}_12m_MA")
            generated_names.append(f"{base_factor}_60m_MA")
            
            # Helper factors' 60m MAs
            for i in range(1, 11):
                generated_names.append(f"Helper_{i}_60m_MA")
            
            # Ensure we have the right number of features
            if len(generated_names) != n_features:
                print(f"Warning: Generated {len(generated_names)} names but data has {n_features} features")
                # Adjust by adding or removing helper features
                if len(generated_names) < n_features:
                    for i in range(len(generated_names), n_features):
                        generated_names.append(f"Unknown_Feature_{i}")
                else:
                    generated_names = generated_names[:n_features]
            
            columns = generated_names
        else:
            print(f"Could not find X data at path: {data_path}")
    
    return columns

def print_feature_importance_mapping(feature_names):
    """
    Print a mapping between feature indices and names for use with XGBoost feature importance.
    """
    if feature_names is None or len(feature_names) == 0:
        return
    
    print("\nFeature Index to Name Mapping (for XGBoost feature importance):")
    print("------------------------------------------------------------")
    for i, name in enumerate(feature_names):
        print(f"Feature {i}: {name}")

def analyze_feature_sample(h5_file, window_id, factor_id, feature_names):
    """
    Analyze a sample of the feature data to help understand what each feature represents.
    """
    if feature_names is None or len(feature_names) == 0:
        return
    
    data_path = f"feature_sets/window_{window_id}/training/{factor_id}/X/data"
    if data_path not in h5_file:
        print(f"Could not find X data at path: {data_path}")
        return
    
    print("\nAnalyzing feature data sample:")
    print("-----------------------------")
    
    # Get the data
    data = h5_file[data_path][:]
    
    # Create a DataFrame with feature names
    df = pd.DataFrame(data, columns=feature_names)
    
    # Print basic statistics
    print("\nBasic Statistics:")
    stats = df.describe().T
    
    # Format for better readability
    stats_formatted = stats.round(4)
    print(stats_formatted)
    
    # Print a few sample rows
    print("\nSample Rows (first 5):")
    print(df.head().round(4))
    
    # Check for correlations between features
    print("\nFeature Correlations (top 5 pairs):")
    corr = df.corr().unstack().sort_values(ascending=False)
    # Remove self-correlations
    corr = corr[corr < 0.999]  # Use 0.999 to handle floating point issues
    print(corr.head(5).round(4))

def main():
    """Main function."""
    if len(sys.argv) < 4:
        print("Usage: python inspect_h5_deeper.py <h5_file_path> <window_id> <factor_id>")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    window_id = sys.argv[2]
    factor_id = sys.argv[3]
    
    if not os.path.exists(h5_path):
        print(f"Error: H5 file not found: {h5_path}")
        sys.exit(1)
    
    print(f"Inspecting H5 file: {h5_path}")
    print(f"Window ID: {window_id}")
    print(f"Factor ID: {factor_id}")
    
    with h5py.File(h5_path, 'r') as h5f:
        # Collect feature information
        feature_names = collect_feature_info(h5f, window_id, factor_id)
        
        if feature_names is not None and len(feature_names) > 0:
            # Print a mapping between feature indices and names
            print_feature_importance_mapping(feature_names)
            
            # Analyze feature data
            analyze_feature_sample(h5f, window_id, factor_id, feature_names)
        else:
            print("Could not determine feature names.")

if __name__ == "__main__":
    main()
