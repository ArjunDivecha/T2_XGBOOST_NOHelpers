#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_h5.py

This script inspects the structure of an H5 file, particularly looking for
feature sets and feature names.

Usage:
    python inspect_h5.py <path_to_h5_file>

Example:
    python inspect_h5.py output/S8_Feature_Sets.h5
"""

import sys
import os
import h5py
import numpy as np
from pathlib import Path

def explore_h5_structure(h5_file, path='/', indent=0):
    """Recursively explore the structure of an H5 file."""
    if isinstance(h5_file[path], h5py.Dataset):
        dataset = h5_file[path]
        shape = dataset.shape
        dtype = dataset.dtype
        print(f"{' ' * indent}- Dataset: {path}")
        print(f"{' ' * (indent+2)}Shape: {shape}")
        print(f"{' ' * (indent+2)}Type: {dtype}")
        
        # If it's a small dataset, print out the values
        if len(shape) == 0 or (len(shape) == 1 and shape[0] < 20):
            try:
                values = dataset[:]
                if dtype.kind == 'S':  # String type
                    try:
                        values = [v.decode('utf-8') for v in values]
                    except:
                        pass
                print(f"{' ' * (indent+2)}Values: {values}")
            except Exception as e:
                print(f"{' ' * (indent+2)}Error reading values: {e}")
        
        # Print attributes
        for attr_name, attr_value in dataset.attrs.items():
            print(f"{' ' * (indent+2)}Attribute: {attr_name} = {attr_value}")
    else:
        print(f"{' ' * indent}+ Group: {path}")
        
        # Print attributes for groups
        if path != '/':
            for attr_name, attr_value in h5_file[path].attrs.items():
                print(f"{' ' * (indent+2)}Attribute: {attr_name} = {attr_value}")
        
        # Recursively explore children with limited depth to avoid too much output
        if indent < 16:  # Limit depth to avoid excessive output
            for name in h5_file[path]:
                child_path = f"{path}/{name}" if path != '/' else f"/{name}"
                explore_h5_structure(h5_file, child_path, indent + 2)
        else:
            num_children = len(h5_file[path])
            if num_children > 0:
                print(f"{' ' * (indent+2)}... ({num_children} more items)")

def find_feature_names(h5_file, window_id=None, factor_id=None):
    """Search for feature names in the H5 file."""
    print("\nSearching for feature names in the H5 file...")
    
    feature_names_paths = []
    
    def visitor_func(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith('feature_names'):
            feature_names_paths.append(name)
    
    # Visit all nodes in the file
    h5_file.visititems(visitor_func)
    
    if not feature_names_paths:
        print("No 'feature_names' datasets found in the file.")
        return
    
    print(f"Found {len(feature_names_paths)} feature_names datasets:")
    
    # Print the first few paths
    for i, path in enumerate(feature_names_paths[:5]):  # Limit to first 5
        print(f"{i+1}. {path}")
        try:
            values = h5_file[path][:]
            if values.dtype.kind == 'S':  # String type
                try:
                    values = [v.decode('utf-8') for v in values]
                except:
                    pass
            print(f"   Values: {values}")
        except Exception as e:
            print(f"   Error reading values: {e}")
    
    if len(feature_names_paths) > 5:
        print(f"... and {len(feature_names_paths) - 5} more paths")
    
    # If window_id and factor_id are provided, try to find specific feature names
    if window_id and factor_id:
        print(f"\nLooking for feature names for window_{window_id} and factor_{factor_id}:")
        
        # Try different possible paths
        possible_paths = [
            f"feature_sets/window_{window_id}/training/{factor_id}/X/feature_names",
            f"window_{window_id}/training/{factor_id}/X/feature_names",
            f"window_{window_id}/{factor_id}/training/X/feature_names",
            f"window_{window_id}/factor_{factor_id}/training/X/feature_names"
        ]
        
        found = False
        for path in possible_paths:
            if path in h5_file:
                print(f"Found at path: {path}")
                values = h5_file[path][:]
                if values.dtype.kind == 'S':  # String type
                    values = [v.decode('utf-8') for v in values]
                print(f"Values: {values}")
                found = True
                break
        
        if not found:
            print("No specific feature names found for the given window and factor.")

def examine_feature_sets(h5_file):
    """Examine the structure of feature sets in the H5 file."""
    print("\nExamining feature sets structure...")
    
    if 'feature_sets' not in h5_file:
        print("No 'feature_sets' group found in the file.")
        return
    
    feature_sets = h5_file['feature_sets']
    print(f"Found feature_sets group with {len(feature_sets)} items")
    
    # Check the first window
    if len(feature_sets) > 0:
        window_key = list(feature_sets.keys())[0]
        print(f"\nExamining first window: {window_key}")
        
        window_group = feature_sets[window_key]
        print(f"Window has {len(window_group)} items: {list(window_group.keys())}")
        
        # Check if window contains splits like training, validation, etc.
        for split_key in window_group.keys():
            split_group = window_group[split_key]
            print(f"\nSplit '{split_key}' has {len(split_group)} items")
            
            # Look at the first few items (factors)
            for i, factor_key in enumerate(list(split_group.keys())[:3]):  # Limit to first 3
                factor_group = split_group[factor_key]
                print(f"\n  Factor '{factor_key}' structure:")
                
                # Print the structure of this factor
                if 'X' in factor_group:
                    x_group = factor_group['X']
                    print(f"    X group has {len(x_group)} items: {list(x_group.keys())}")
                    
                    if 'data' in x_group:
                        data = x_group['data']
                        print(f"    X/data shape: {data.shape}, dtype: {data.dtype}")
                    
                    if 'feature_names' in x_group:
                        feature_names = x_group['feature_names']
                        values = feature_names[:]
                        if values.dtype.kind == 'S':  # String type
                            values = [v.decode('utf-8') for v in values]
                        print(f"    Feature names: {values}")
                
                if 'y' in factor_group:
                    y_dataset = factor_group['y']
                    print(f"    y shape: {y_dataset.shape}, dtype: {y_dataset.dtype}")
                    
                    # Print a sample of y values
                    try:
                        y_values = y_dataset[:]
                        print(f"    y sample: {y_values[:5]}")
                    except Exception as e:
                        print(f"    Error reading y values: {e}")
            
            if len(split_group) > 3:
                print(f"\n  ... and {len(split_group) - 3} more factors")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <path_to_h5_file>")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    if not os.path.exists(h5_path):
        print(f"Error: H5 file not found: {h5_path}")
        sys.exit(1)
    
    print(f"Inspecting H5 file: {h5_path}")
    
    window_id = None
    factor_id = None
    
    # Check for optional window and factor arguments
    if len(sys.argv) >= 3:
        window_id = sys.argv[2]
    if len(sys.argv) >= 4:
        factor_id = sys.argv[3]
    
    # Open the H5 file
    with h5py.File(h5_path, 'r') as h5f:
        # Basic file info
        print(f"H5 file contains {len(h5f)} top-level groups/datasets")
        print(f"Top-level keys: {list(h5f.keys())}")
        
        # Examine overall structure of first level
        print("\nOverall structure (first level):")
        for key in h5f.keys():
            if isinstance(h5f[key], h5py.Group):
                print(f"Group: {key} with {len(h5f[key])} items")
            else:
                print(f"Dataset: {key} with shape {h5f[key].shape}")
        
        # Examine feature sets
        examine_feature_sets(h5f)
        
        # Look for feature names
        find_feature_names(h5f, window_id, factor_id)
        
        # Option to explore full structure
        if '--full' in sys.argv:
            print("\nFull H5 file structure:")
            explore_h5_structure(h5f)

if __name__ == "__main__":
    main()
