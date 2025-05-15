import h5py
import pandas as pd
import numpy as np

# Open the feature sets file
print("Examining S8_Feature_Sets.h5 structure...")

with h5py.File("output/S8_Feature_Sets.h5", "r") as hf:
    # Print metadata
    print("\nMetadata:")
    meta = hf['metadata']
    for key, value in meta.attrs.items():
        print(f"  {key}: {value}")
    
    # Check factor columns
    factor_columns = [name.decode('utf-8') for name in meta['factor_columns'][:]]
    print(f"  Total factors: {len(factor_columns)}")
    print(f"  Sample factors: {factor_columns[:5]}...")
    
    # Check feature sets
    print("\nFeature Sets Structure:")
    feature_sets_group = hf['feature_sets']
    windows = list(feature_sets_group.keys())
    print(f"  Total windows: {len(windows)}")
    
    # Check if windows have any training data
    windows_with_data = 0
    factors_per_window = []
    
    for window_name in windows:
        window_group = feature_sets_group[window_name]
        
        # Check if training data exists
        if 'training' in window_group:
            training_group = window_group['training']
            factors = list(training_group.keys())
            
            if factors:
                windows_with_data += 1
                factors_per_window.append(len(factors))
                
                # Look at one specific factor as an example (if available)
                if len(factors) > 0:
                    sample_factor = factors[0]
                    print(f"\nSample data for window {window_name}, factor {sample_factor}:")
                    
                    factor_group = training_group[sample_factor]
                    
                    # Check X features
                    if 'X' in factor_group:
                        X_group = factor_group['X']
                        X_columns = [name.decode('utf-8') for name in X_group['columns'][:]]
                        X_data = X_group['data'][:]
                        
                        print(f"  X features shape: {X_data.shape}")
                        print(f"  X columns: {X_columns}")
                    
                    # Check y target
                    if 'y' in factor_group:
                        y_data = factor_group['y'][:]
                        print(f"  y target shape: {y_data.shape}")
                    
                    break  # Just show one example
    
    print(f"\nWindows with training data: {windows_with_data} out of {len(windows)}")
    
    if factors_per_window:
        print(f"Average factors per window: {np.mean(factors_per_window):.1f}")
    else:
        print("No factors found in any window.")
    
    # Check specific window (e.g., window_100) if it exists
    specific_window = 'window_100'
    if specific_window in feature_sets_group:
        print(f"\nExamining {specific_window}:")
        window_group = feature_sets_group[specific_window]
        
        for split_name in window_group:
            split_group = window_group[split_name]
            factors = list(split_group.keys())
            print(f"  {split_name} split has {len(factors)} factors")
            
            if factors:
                print(f"  Sample factors: {factors[:5]}")
            else:
                print("  No factors found in this split.") 