import h5py
import pandas as pd
import numpy as np
from collections import Counter

# Define the commodities we're looking for
commodities = ["Gold", "Copper", "Oil", "Agriculture"]

print(f"Analyzing specific factor names containing {', '.join(commodities)}...")

# Open the HDF5 file
with h5py.File("output/S7_Helper_Features.h5", "r") as hf:
    # Create a counter for each specific factor containing our commodities
    commodity_factor_counter = Counter()
    
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
            
            # Count specific factor names containing our commodities
            for helper in helper_names:
                for commodity in commodities:
                    if commodity in helper:
                        commodity_factor_counter[helper] += 1

    # Print results for each commodity
    total_helpers = sum(commodity_factor_counter.values())
    
    for commodity in commodities:
        print(f"\n{commodity} Related Factors:")
        print("-" * (len(commodity) + 16))
        
        # Get all factors containing this commodity
        commodity_factors = [(factor, count) for factor, count in commodity_factor_counter.items() 
                            if commodity in factor]
        
        # Sort by count (descending)
        commodity_factors.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 10 factors for this commodity
        commodity_total = sum(count for _, count in commodity_factors)
        print(f"Top 10 most common {commodity} factors (out of {len(commodity_factors)} total {commodity} factors):")
        
        for i, (factor, count) in enumerate(commodity_factors[:10], 1):
            percentage = (count / commodity_total) * 100
            print(f"{i}. {factor}: {count} times ({percentage:.2f}% of {commodity} occurrences)")
            
        print(f"\nTotal {commodity} factor occurrences: {commodity_total}")
        
    print(f"\nOverall commodity factor occurrences: {total_helpers}") 