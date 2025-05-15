'''
debug_naming_conventions.py

This script tests the improved naming conventions in Step 2 (Calculate Moving Averages)
and verifies that Step 8 (Create Feature Sets) can correctly find the moving average columns.

----------------------------------------------------------------------------------------------------
'''

import pandas as pd
import numpy as np
import re
import os
import h5py
from pathlib import Path

print("="*80)
print("DEBUG SCRIPT: Testing Factor Naming Conventions")
print("="*80)

# Step 1: Check if the mapping file was created by Step 2
mapping_file = "output/S2_Column_Mapping.xlsx"
if os.path.exists(mapping_file):
    print(f"\n1. Column mapping file exists: {mapping_file}")
    mapping_df = pd.read_excel(mapping_file)
    print(f"   - Contains {len(mapping_df)} factor mappings")
    print(f"   - Sample of mapping data:")
    print(mapping_df.head(5))
else:
    print(f"\n1. Column mapping file not found: {mapping_file}")
    print("   - This file should be created by Step 2. Please run Step 2 first.")

# Step 2: Check the actual column names in the MA file
ma_file = "output/S2_T2_Optimizer_with_MA.xlsx"
if os.path.exists(ma_file):
    print(f"\n2. Moving averages file exists: {ma_file}")
    ma_df = pd.read_excel(ma_file)
    print(f"   - Contains {len(ma_df.columns)} columns")
    
    # Analyze column naming patterns
    patterns = {
        'standard': 0,          # Factor_3m
        'with_ts': 0,           # Factor_TS_3m
        'with_cs': 0,           # Factor_CS_3m
        'double_period': 0,     # Factor_3m_12m
        'ts_double_period': 0,  # Factor_TS_3m_12m
        'cs_double_period': 0,  # Factor_CS_3m_12m
        'other': 0
    }
    
    for col in ma_df.columns:
        if col == 'Date':
            continue
            
        if re.search(r'_TS_\d+m_\d+m$', col):
            patterns['ts_double_period'] += 1
        elif re.search(r'_CS_\d+m_\d+m$', col):
            patterns['cs_double_period'] += 1
        elif re.search(r'_\d+m_\d+m$', col):
            patterns['double_period'] += 1
        elif re.search(r'_TS_\d+m$', col):
            patterns['with_ts'] += 1
        elif re.search(r'_CS_\d+m$', col):
            patterns['with_cs'] += 1
        elif re.search(r'_\d+m$', col) and not ('_TS_' in col or '_CS_' in col):
            patterns['standard'] += 1
        else:
            patterns['other'] += 1
    
    print("   - Column naming patterns:")
    for pattern, count in patterns.items():
        print(f"     - {pattern}: {count} columns")
    
    # Test some troublesome factor names
    test_factors = [
        "Gold_TS",               # Standard with TS suffix
        "Oil_TS_3m",             # Already has a time period suffix
        "Mcap Weights_CS_60m",   # Already has CS and time period suffix
        "10Yr Bond_TS",          # Standard with TS suffix and number
        "10Yr Bond 12_CS"        # Has CS suffix and number in name
    ]
    
    print("\n   - Testing specific factor patterns:")
    for factor in test_factors:
        print(f"\n     Factor: {factor}")
        
        # Find columns related to this factor
        base_name = factor
        if "_TS" in factor:
            base_name = factor.split("_TS")[0]
        elif "_CS" in factor:
            base_name = factor.split("_CS")[0]
            
        # Remove existing time period if any
        base_name = re.sub(r'_\d+m$', '', base_name)
        
        # Find related columns
        related_cols = [col for col in ma_df.columns if col.startswith(base_name)]
        # Filter out non-relevant columns (e.g., "Gold" vs "Golden")
        if "_TS" in factor:
            related_cols = [col for col in related_cols if "_TS" in col or col == factor]
        elif "_CS" in factor:
            related_cols = [col for col in related_cols if "_CS" in col or col == factor]
            
        print(f"     Related columns: {len(related_cols)}")
        for col in sorted(related_cols):
            print(f"     - {col}")
else:
    print(f"\n2. Moving averages file not found: {ma_file}")
    print("   - This file should be created by Step 2. Please run Step 2 first.")

# Step 3: Test the find_ma_columns and find_helper_ma_column functions from Step 8
print("\n3. Testing column finding functions from Step 8")

try:
    # Import necessary functions from Step 8
    from Step_8_Create_Feature_Sets import find_helper_ma_column
    
    print("   Successfully imported find_helper_ma_column function")
    
    # Test with a few test factors
    if os.path.exists(ma_file):
        ma_df = pd.read_excel(ma_file)
        test_helpers = [
            "Gold_TS",             # Standard with TS suffix
            "Oil_TS_3m",           # Already has a time period suffix
            "Mcap Weights_CS_60m"  # Already has CS and time period suffix
        ]
        
        print("\n   Testing find_helper_ma_column:")
        for helper in test_helpers:
            ma_col = find_helper_ma_column(helper, ma_df)
            print(f"     - Helper: {helper} → MA column: {ma_col}")
    else:
        print("   Cannot test functions without data file")
        
except ImportError:
    print("   Could not import functions from Step_8_Create_Feature_Sets.py")
    print("   The file may need to be modified to expose these functions for importing")
    
    # Define simplified test versions of the functions
    def test_find_ma_columns(factor, base_data):
        """Test version of find_ma_columns"""
        # Check for column mapping file
        mapping_file = "output/S2_Column_Mapping.xlsx"
        if os.path.exists(mapping_file):
            mapping_df = pd.read_excel(mapping_file)
            factor_row = mapping_df[mapping_df['original_column'] == factor]
            if not factor_row.empty:
                col_3m = factor_row['column_3m'].iloc[0]
                col_12m = factor_row['column_12m'].iloc[0]
                col_60m = factor_row['column_60m'].iloc[0]
                
                # Verify columns exist in data
                col_3m = col_3m if col_3m != "N/A" and col_3m in base_data.columns else factor
                col_12m = col_12m if col_12m != "N/A" and col_12m in base_data.columns else factor
                col_60m = col_60m if col_60m != "N/A" and col_60m in base_data.columns else factor
                
                return {
                    "1m": factor,
                    "3m": col_3m,
                    "12m": col_12m,
                    "60m": col_60m
                }
        
        # Try patterns manually
        result = {"1m": factor}
        for period in ["3m", "12m", "60m"]:
            # Try different patterns
            candidates = [
                f"{factor}_{period}",
                f"{factor}_TS_{period}",
                f"{factor}_CS_{period}"
            ]
            
            # If factor has CS or TS suffix
            if "_CS" in factor:
                base_name = factor.split("_CS")[0]
                candidates.extend([
                    f"{base_name}_CS_{period}",
                    f"{base_name}_{period}"
                ])
            elif "_TS" in factor:
                base_name = factor.split("_TS")[0]
                candidates.extend([
                    f"{base_name}_TS_{period}",
                    f"{base_name}_{period}"
                ])
                
            # If factor has a period suffix
            period_match = re.search(r'_(\d+)m$', factor)
            if period_match:
                candidates.append(f"{factor}_{period}")
                
                base_name = re.sub(r'_\d+m$', '', factor)
                candidates.extend([
                    f"{base_name}_{period}",
                    f"{base_name}_TS_{period}",
                    f"{base_name}_CS_{period}"
                ])
            
            # Check all candidates
            for col in candidates:
                if col in base_data.columns:
                    result[period] = col
                    break
            else:
                result[period] = factor
                print(f"Warning: No {period} MA column found for {factor}, using base factor")
                
        return result
    
    def test_find_helper_ma_column(helper, base_data):
        """Test version of find_helper_ma_column"""
        # Check for column mapping file
        mapping_file = "output/S2_Column_Mapping.xlsx"
        if os.path.exists(mapping_file):
            mapping_df = pd.read_excel(mapping_file)
            helper_row = mapping_df[mapping_df['original_column'] == helper]
            if not helper_row.empty:
                ma_column = helper_row['column_60m'].iloc[0]
                if ma_column != "N/A" and ma_column in base_data.columns:
                    return ma_column
        
        # Try different patterns
        ma_candidates = [
            f"{helper}_60m",
            f"{helper}_TS_60m",
            f"{helper}_CS_60m"
        ]
        
        # If helper has CS or TS suffix
        if "_CS" in helper:
            base_name = helper.split("_CS")[0]
            ma_candidates.extend([
                f"{base_name}_CS_60m",
                f"{base_name}_60m"
            ])
        elif "_TS" in helper:
            base_name = helper.split("_TS")[0]
            ma_candidates.extend([
                f"{base_name}_TS_60m",
                f"{base_name}_60m"
            ])
        
        # If helper has a period suffix
        period_match = re.search(r'_(\d+)m$', helper)
        if period_match:
            # Try adding 60m after existing period
            ma_candidates.append(f"{helper}_60m")
            
            # Try replacing period with 60m
            base_name = re.sub(r'_\d+m$', '', helper)
            ma_candidates.extend([
                f"{base_name}_60m",
                f"{base_name}_TS_60m",
                f"{base_name}_CS_60m"
            ])
        
        # Check all candidates
        for col in ma_candidates:
            if col in base_data.columns:
                return col
                
        # Use original helper if available
        if helper in base_data.columns:
            return helper
            
        return None
    
    # Test with sample factors
    if os.path.exists(ma_file):
        ma_df = pd.read_excel(ma_file)
        
        test_factors = [
            "Gold_TS",             # Standard with TS suffix
            "Oil_TS_3m",           # Already has a time period suffix
            "Mcap Weights_CS_60m"  # Already has CS and time period suffix
        ]
        
        print("\n   Testing test_find_ma_columns:")
        for factor in test_factors:
            ma_cols = test_find_ma_columns(factor, ma_df)
            print(f"     - Factor: {factor}")
            for period, col in ma_cols.items():
                print(f"       {period}: {col}")
                
        print("\n   Testing test_find_helper_ma_column:")
        for helper in test_factors:
            ma_col = test_find_helper_ma_column(helper, ma_df)
            print(f"     - Helper: {helper} → MA column: {ma_col}")
    else:
        print("   Cannot test functions without data file")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80) 