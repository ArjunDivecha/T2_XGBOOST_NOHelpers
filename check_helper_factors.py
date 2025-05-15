import h5py
import pandas as pd
import numpy as np

# Open the helper features file
print("Examining S7_Helper_Features.h5 structure...")

with h5py.File("output/S7_Helper_Features.h5", "r") as hf:
    # Print metadata
    print("\nMetadata:")
    meta = hf['metadata']
    for key, value in meta.attrs.items():
        print(f"  {key}: {value}")
    
    # Check factor columns
    factor_columns = [name.decode('utf-8') for name in meta['factor_columns'][:]]
    print(f"  Total factors: {len(factor_columns)}")
    print(f"  Sample factors: {factor_columns[:5]}...")
    
    # Check helper features
    print("\nHelper Features Structure:")
    helpers_group = hf['helper_features']
    windows = list(helpers_group.keys())
    print(f"  Total windows: {len(windows)}")
    
    # Sample a window to examine
    sample_window = windows[len(windows)//2]  # Middle window
    print(f"\nExamining window {sample_window}:")
    window_group = helpers_group[sample_window]
    
    # Get factors in this window
    factors = list(window_group.keys())
    print(f"  Total factors in window: {len(factors)}")
    
    if factors:
        # Check a sample factor
        sample_factor_key = factors[0]
        print(f"\n  Examining factor {sample_factor_key}:")
        factor_group = window_group[sample_factor_key]
        
        # Convert / in factor name that might have been replaced with _
        sample_factor = sample_factor_key.replace('_', '/')
        
        # Get helper names
        helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
        helper_corrs = factor_group['correlations'][:]
        
        print(f"    Number of helpers: {len(helper_names)}")
        print(f"    Helper factors: {helper_names}")
        print(f"    Correlations: {helper_corrs}")
    else:
        print("  No factors found in this window.")
    
    # Also check if any window has factors with valid helper features
    windows_with_factors = 0
    total_factors_with_helpers = 0
    
    for window_name in windows:
        window_group = helpers_group[window_name]
        factors = list(window_group.keys())
        
        if factors:
            windows_with_factors += 1
            total_factors_with_helpers += len(factors)
    
    print(f"\nWindows with factors: {windows_with_factors} out of {len(windows)}")
    print(f"Total factor-window combinations with helper features: {total_factors_with_helpers}")
    
    if windows_with_factors > 0:
        print(f"Average factors per window: {total_factors_with_helpers / windows_with_factors:.1f}")

# Next, check S2_T2_Optimizer_with_MA.xlsx to verify that the necessary factor columns exist
print("\n\nChecking factor data file (S2_T2_Optimizer_with_MA.xlsx)...")

try:
    # Load factor data with moving averages (just the column names for verification)
    factor_data = pd.read_excel("output/S2_T2_Optimizer_with_MA.xlsx", nrows=0)
    
    print(f"Factor data file columns: {len(factor_data.columns)}")
    print(f"First 10 columns: {list(factor_data.columns)[:10]}")
    
    # Check for a few key patterns
    patterns = ["_TS_3m", "_TS_12m", "_TS_60m", "_CS_60m"]
    pattern_counts = {pattern: sum(1 for col in factor_data.columns if pattern in col) for pattern in patterns}
    
    print(f"\nColumn pattern counts:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} columns")
        
    # Sample a specific factor to check all its moving averages
    if factor_columns:
        sample_factor = factor_columns[0]
        print(f"\nChecking columns for sample factor: {sample_factor}")
        
        # Check factor itself and its moving averages
        needed_columns = [
            sample_factor,
            f"{sample_factor}_TS_3m",
            f"{sample_factor}_TS_12m",
            f"{sample_factor}_TS_60m"
        ]
        
        for col in needed_columns:
            exists = col in factor_data.columns
            print(f"  {col}: {'✓ exists' if exists else '✗ missing'}")
    
except Exception as e:
    print(f"Error loading factor data file: {e}") 