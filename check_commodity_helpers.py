import h5py
import pandas as pd
import numpy as np
from collections import Counter

# Define the commodities we're looking for
commodities = ["Gold", "Copper", "Oil", "Agriculture"]

print(f"Checking how often {', '.join(commodities)} appear as helper features...")

# Open the HDF5 file
with h5py.File("output/S7_Helper_Features.h5", "r") as hf:
    # Create a counter for each commodity
    commodity_counts = {commodity: 0 for commodity in commodities}
    total_helpers = 0
    
    # Get the helper features group
    helpers_group = hf['helper_features']
    
    # Loop through all windows
    for window_name in helpers_group:
        window_group = helpers_group[window_name]
        
        # Loop through all factors in this window
        for factor_name in window_group:
            factor_group = window_group[factor_name]
            
            # Get helper names
            helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
            total_helpers += len(helper_names)
            
            # Check for each commodity
            for commodity in commodities:
                # Count occurrences where the commodity appears in the helper name
                for helper in helper_names:
                    if commodity in helper:
                        commodity_counts[commodity] += 1

    # Print results
    print("\nCommodity Occurrences as Helper Features:")
    print("----------------------------------------")
    for commodity, count in commodity_counts.items():
        percentage = (count / total_helpers) * 100
        print(f"{commodity}: {count} times ({percentage:.2f}% of all helper relationships)")
    
    print(f"\nTotal helper relationships analyzed: {total_helpers}")
    
    # Also check top 20 most common helpers
    print("\nChecking top 20 most common helper features for reference...")
    
    # Counter for all helper features
    all_helpers_counter = Counter()
    
    # Loop through all windows again
    for window_name in helpers_group:
        window_group = helpers_group[window_name]
        
        # Loop through all factors in this window
        for factor_name in window_group:
            factor_group = window_group[factor_name]
            
            # Get helper names
            helper_names = [name.decode('utf-8') for name in factor_group['names'][:]]
            
            # Count each helper
            for helper in helper_names:
                all_helpers_counter[helper] += 1
    
    # Print top 20 most common helpers
    print("\nTop 20 Most Common Helper Features:")
    print("----------------------------------")
    for helper, count in all_helpers_counter.most_common(20):
        percentage = (count / total_helpers) * 100
        print(f"{helper}: {count} times ({percentage:.2f}%)") 