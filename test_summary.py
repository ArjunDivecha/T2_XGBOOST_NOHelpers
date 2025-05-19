import pandas as pd
import os
import json
import h5py

"""
Simple test script to verify window schedule column names.
This helps diagnose issues with column naming conventions.
"""

def main():
    """Main execution function."""
    print("=== Testing window schedule column names ===")
    
    # Load window schedule
    window_schedule_file = os.path.join("output", "S4_Window_Schedule.xlsx")
    if not os.path.exists(window_schedule_file):
        raise FileNotFoundError(f"Window schedule file not found: {window_schedule_file}")
    
    window_schedule = pd.read_excel(window_schedule_file)
    print(f"Loaded window schedule with {len(window_schedule)} windows")
    
    # Print column names
    print("\nColumn names (exact case):")
    for col in window_schedule.columns:
        print(f"  - {col}")
    
    # Print first row as a sample
    if len(window_schedule) > 0:
        print("\nSample first row:")
        first_row = window_schedule.iloc[0]
        for col in window_schedule.columns:
            print(f"  - {col}: {first_row[col]}")

# Check if the model file was created
h5_output_file = os.path.join("output", "S11B_Linear_Models.h5")
if os.path.exists(h5_output_file):
    print(f"Model file exists: {h5_output_file}")
    with h5py.File(h5_output_file, 'r') as hf:
        # Print the structure
        print("HDF5 file structure:")
        def print_structure(name, obj):
            print(f"  {name}")
        hf.visititems(print_structure)
else:
    print(f"Model file not found: {h5_output_file}")

if __name__ == "__main__":
    main() 