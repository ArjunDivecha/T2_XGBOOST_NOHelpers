import pandas as pd
import numpy as np
import h5py

print("Debug script for Step 8: Testing a single factor and window")
print("="*80)

# Load factor data
print("Loading data files...")
factor_data = pd.read_excel("output/S2_T2_Optimizer_with_MA.xlsx")
window_schedule = pd.read_excel("output/S4_Window_Schedule.xlsx")

# Get a sample window (e.g., window 100)
sample_window_id = 100
sample_window = window_schedule[window_schedule['Window_ID'] == sample_window_id].iloc[0]
print(f"\nSample window {sample_window_id}:")
print(f"  Training: {sample_window['Training_Start_Date']} to {sample_window['Training_End_Date']}")
print(f"  Validation: {sample_window['Training_End_Date']} to {sample_window['Validation_End_Date']}")
print(f"  Prediction: {sample_window['Prediction_Date']}")

# Get date ranges for this window
training_start = sample_window['Training_Start_Date']
training_end = sample_window['Training_End_Date']
validation_end = sample_window['Validation_End_Date']
prediction_date = sample_window['Prediction_Date']

# Get data for this window
training_data = factor_data[(factor_data['Date'] >= training_start) & (factor_data['Date'] <= training_end)]
validation_data = factor_data[(factor_data['Date'] > training_end) & (factor_data['Date'] <= validation_end)]
prediction_data = factor_data[factor_data['Date'] == prediction_date]
print(f"\nData subsets:")
print(f"  Training: {len(training_data)} rows")
print(f"  Validation: {len(validation_data)} rows")
print(f"  Prediction: {len(prediction_data)} rows")

# Load helper features
print("\nLoading helper features...")
with h5py.File("output/S7_Helper_Features.h5", "r") as hf:
    # Get factor columns
    factor_columns = [name.decode('utf-8') for name in hf['metadata']['factor_columns'][:]]
    print(f"Total factors in metadata: {len(factor_columns)}")
    
    # Get sample factor
    sample_factor = "Gold_TS"
    print(f"\nSample factor: {sample_factor}")
    
    if sample_factor not in factor_columns:
        similar_factors = [f for f in factor_columns if "Gold" in f]
        print(f"Factor not in list. Similar factors: {similar_factors[:5]}...")
    else:
        print(f"Factor found in metadata list")
    
    # Get helper features for this factor in this window
    helpers_group = hf['helper_features']
    window_group_name = f"window_{sample_window_id}"
    
    if window_group_name not in helpers_group:
        print(f"Window {sample_window_id} not found in helper features")
    else:
        window_group = helpers_group[window_group_name]
        
        factor_name = sample_factor.replace('/', '_')
        if factor_name not in window_group:
            print(f"Factor {sample_factor} not found in window {sample_window_id}")
            print(f"Available factors in window: {list(window_group.keys())[:5]}...")
        else:
            factor_group = window_group[factor_name]
            
            helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
            helper_corrs = factor_group['correlations'][:]
            
            print(f"Number of helpers: {len(helper_names)}")
            print(f"Helper names: {helper_names}")
            print(f"Correlations: {helper_corrs}")

# Check the factor in data file
print("\nChecking factor in data file:")
if sample_factor in factor_data.columns:
    print(f"Factor {sample_factor} found in data file")
else:
    print(f"Factor {sample_factor} NOT found in data file")
    similar_cols = [col for col in factor_data.columns if sample_factor in col]
    print(f"Similar columns: {similar_cols}")

# Check moving averages for the factor
print("\nChecking moving averages for the factor:")
ma_patterns = [
    f"{sample_factor}_3m",
    f"{sample_factor}_12m", 
    f"{sample_factor}_60m",
    f"{sample_factor}_TS_3m",
    f"{sample_factor}_TS_12m", 
    f"{sample_factor}_TS_60m"
]
for pattern in ma_patterns:
    if pattern in factor_data.columns:
        print(f"  ✓ Found {pattern}")
    else:
        print(f"  ✗ {pattern} not found")

# Direct check of the issue
print("\nDirect check of the key issue:")
print("Is the factor in helper_features[window_id]?")

with h5py.File("output/S7_Helper_Features.h5", "r") as hf:
    helpers_group = hf['helper_features']
    window_group_name = f"window_{sample_window_id}"
    window_group = helpers_group[window_group_name]
    
    # Check all factors in helper features
    for factor_key in window_group:
        # Convert key format to normal format
        factor = factor_key.replace('_', '/')
        
        # Check if this factor is in the data file
        if factor in factor_data.columns:
            print(f"Factor {factor} from helper features IS in data file")
            
            # Check MA columns
            ma_found = True
            ma_columns = [
                f"{factor}_3m",
                f"{factor}_12m",
                f"{factor}_60m"
            ]
            
            for ma_col in ma_columns:
                if ma_col not in factor_data.columns:
                    ma_found = False
                    print(f"  But MA column {ma_col} is missing")
            
            if ma_found:
                print(f"  All MA columns found for {factor}")
                
                # This should have been a valid factor - let's check what's happening in the loop
                print("\nSimulating the processing loop for this factor:")
                
                # Get helper features for this factor
                factor_group = window_group[factor_key]
                helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
                
                print(f"Factor {factor} has {len(helper_names)} helpers")
                
                # Define helper functions from Step_8
                def find_ma_columns(factor, base_data):
                    # Try different patterns
                    patterns = [
                        {
                            "1m": factor,
                            "3m": f"{factor}_3m",
                            "12m": f"{factor}_12m",
                            "60m": f"{factor}_60m"
                        },
                        {
                            "1m": factor,
                            "3m": f"{factor}_TS_3m",
                            "12m": f"{factor}_TS_12m",
                            "60m": f"{factor}_TS_60m"
                        }
                    ]
                    
                    for pattern in patterns:
                        if all(col in base_data.columns for col in pattern.values()):
                            return pattern
                    
                    result = {"1m": factor}
                    for period in ["3m", "12m", "60m"]:
                        ma_candidates = [f"{factor}_{period}", f"{factor}_TS_{period}"]
                        for col in ma_candidates:
                            if col in base_data.columns:
                                result[period] = col
                                break
                        else:
                            result[period] = factor
                    
                    return result
                
                def find_helper_ma_column(helper, base_data):
                    ma_candidates = [f"{helper}_60m", f"{helper}_TS_60m"]
                    for col in ma_candidates:
                        if col in base_data.columns:
                            return col
                    return helper if helper in base_data.columns else None
                
                # Try to process the factor
                factor_ma_cols = find_ma_columns(factor, training_data)
                print(f"MA columns found: {factor_ma_cols}")
                
                # Create factor's own features
                try:
                    X_train_own = pd.DataFrame({
                        f"{factor}_1m": training_data[factor_ma_cols["1m"]],
                        f"{factor}_3m": training_data[factor_ma_cols["3m"]],
                        f"{factor}_12m": training_data[factor_ma_cols["12m"]],
                        f"{factor}_60m": training_data[factor_ma_cols["60m"]]
                    })
                    print(f"Successfully created X_train_own with shape {X_train_own.shape}")
                except Exception as e:
                    print(f"Error creating X_train_own: {e}")
                
                # Create helper features
                try:
                    X_train_helpers = pd.DataFrame()
                    for i, helper in enumerate(helper_names):
                        helper_col = find_helper_ma_column(helper, training_data)
                        if helper_col is not None:
                            X_train_helpers[f"helper_{i+1}_{helper}"] = training_data[helper_col]
                        else:
                            print(f"Helper {helper} not found")
                    
                    if X_train_helpers.empty:
                        print("X_train_helpers is empty - no valid helpers found")
                    else:
                        print(f"Successfully created X_train_helpers with shape {X_train_helpers.shape}")
                except Exception as e:
                    print(f"Error creating X_train_helpers: {e}")
                
                # Create target
                try:
                    y_train = training_data[factor].shift(-1)
                    print(f"Successfully created y_train with shape {y_train.shape}")
                    
                    # Final check
                    if not X_train_helpers.empty:
                        X_train = pd.concat([X_train_own, X_train_helpers], axis=1)
                        
                        # Remove last row
                        X_train = X_train.iloc[:-1]
                        y_train = y_train.iloc[:-1]
                        
                        print(f"Final X_train shape: {X_train.shape}")
                        print(f"Final y_train shape: {y_train.shape}")
                        print(f"This factor should have been successful!")
                except Exception as e:
                    print(f"Error creating y_train: {e}")
                
                # Break after finding one valid factor
                break
        else:
            print(f"Factor {factor} from helper features is NOT in data file") 