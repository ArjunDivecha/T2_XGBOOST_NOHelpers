import h5py
import pandas as pd
import numpy as np

# Define a specific factor to check
TARGET_FACTOR = "Gold_TS"  # Change this to check different factors

# Open the helper features file
print(f"Examining factor '{TARGET_FACTOR}' in S7_Helper_Features.h5...")

with h5py.File("output/S7_Helper_Features.h5", "r") as hf:
    # Get factor columns
    factor_columns = [name.decode('utf-8') for name in hf['metadata']['factor_columns'][:]]
    
    if TARGET_FACTOR not in factor_columns:
        print(f"Error: '{TARGET_FACTOR}' not found in factor columns list")
        similar_factors = [f for f in factor_columns if TARGET_FACTOR in f]
        print(f"Similar factors: {similar_factors}")
    else:
        print(f"'{TARGET_FACTOR}' found in factor columns list")
    
    # Check helper features
    print("\nChecking helpers in each window:")
    helpers_group = hf['helper_features']
    windows = list(helpers_group.keys())
    
    # Sample a few windows
    sample_windows = [windows[0], windows[len(windows)//2], windows[-1]]
    
    for window_name in sample_windows:
        print(f"\nWindow {window_name}:")
        window_group = helpers_group[window_name]
        
        # Get factors in this window
        factors = list(window_group.keys())
        
        # Look for our target factor
        target_key = TARGET_FACTOR.replace('/', '_')
        if target_key in factors:
            print(f"  Found target factor")
            factor_group = window_group[target_key]
            
            # Get helper names
            helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
            helper_corrs = factor_group['correlations'][:]
            
            print(f"  Number of helpers: {len(helper_names)}")
            print(f"  Helper factors: {helper_names}")
            print(f"  Correlations: {helper_corrs}")
        else:
            print(f"  Target factor not found in this window")

# Now, check the factor data file
print("\n\nChecking if the factor and its helpers exist in S2_T2_Optimizer_with_MA.xlsx...")

try:
    # Load factor data with moving averages
    factor_data = pd.read_excel("output/S2_T2_Optimizer_with_MA.xlsx")
    
    # Check if target factor exists
    if TARGET_FACTOR in factor_data.columns:
        print(f"✓ Found target factor '{TARGET_FACTOR}' in data file")
    else:
        print(f"✗ Target factor '{TARGET_FACTOR}' NOT found in data file")
        similar_cols = [col for col in factor_data.columns if TARGET_FACTOR in col]
        print(f"  Similar columns: {similar_cols}")
    
    # Check for MA columns of target factor
    ma_patterns = ["_TS_3m", "_TS_12m", "_TS_60m"]
    for pattern in ma_patterns:
        ma_col = f"{TARGET_FACTOR}{pattern}"
        if ma_col in factor_data.columns:
            print(f"✓ Found MA column '{ma_col}' in data file")
        else:
            print(f"✗ MA column '{ma_col}' NOT found in data file")
            similar_cols = [col for col in factor_data.columns if col.endswith(pattern) and TARGET_FACTOR in col]
            print(f"  Similar columns: {similar_cols}")
    
    # If we found the factor in a window, check its helpers too
    if 'helper_names' in locals():
        print("\nChecking helper factors:")
        for helper in helper_names:
            if helper in factor_data.columns:
                print(f"✓ Helper '{helper}' found in data file")
            else:
                print(f"✗ Helper '{helper}' NOT found in data file")
                
                # Check if 60-month MA of this helper exists
                helper_60m = f"{helper}_TS_60m"
                if helper_60m in factor_data.columns:
                    print(f"  ✓ But found its 60-month MA: '{helper_60m}'")
                else:
                    # Try other patterns
                    similar_cols = [col for col in factor_data.columns if helper in col and "_TS_60m" in col]
                    if similar_cols:
                        print(f"  ✓ Found similar MA columns: {similar_cols}")
                    else:
                        print(f"  ✗ No related MA columns found for this helper")
    
except Exception as e:
    print(f"Error loading factor data file: {e}") 