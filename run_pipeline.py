'''
run_pipeline.py

This script runs the XGBoost Factor Forecasting pipeline with the improved column naming conventions.
It executes Step 2 (Calculate Moving Averages) and Step 8 (Create Feature Sets) in sequence.

----------------------------------------------------------------------------------------------------
'''

import os
import sys
import subprocess
import time

def run_step(step_script, step_name):
    """Run a pipeline step and report on its success."""
    print(f"\n{'='*80}")
    print(f"EXECUTING: {step_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, step_script], capture_output=True, text=True)
    end_time = time.time()
    
    print(f"\nSTDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print(f"\nSTDERR:")
        print(result.stderr)
    
    success = result.returncode == 0
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"RESULT: {'SUCCESS' if success else 'FAILURE'} in {duration:.2f} seconds")
    print(f"{'='*80}")
    
    return success

def main():
    print(f"{'='*80}")
    print("XGBOOST FACTOR FORECASTING PIPELINE")
    print(f"{'='*80}")
    print("Running pipeline with improved column naming conventions")
    
    # Check that required Step 1 output exists
    step1_output = "output/S1_T2_Optimizer_cleaned.xlsx"
    if not os.path.exists(step1_output):
        print(f"ERROR: Required Step 1 output file not found: {step1_output}")
        print("Please run Step 1 first to generate the cleaned data file.")
        sys.exit(1)
    
    # Define steps to run
    steps = [
        {"script": "Step_2_Calculate_Moving_Averages.py", "name": "Step 2: Calculate Moving Averages"},
        {"script": "debug_naming_conventions.py", "name": "Debug: Test Naming Conventions"},
        {"script": "Step_8_Create_Feature_Sets.py", "name": "Step 8: Create Feature Sets"}
    ]
    
    # Track overall success
    all_steps_succeeded = True
    
    # Run each step
    for step in steps:
        success = run_step(step["script"], step["name"])
        if not success:
            print(f"\nERROR: {step['name']} failed. Stopping pipeline.")
            all_steps_succeeded = False
            break
    
    # Final status
    print(f"\n{'='*80}")
    if all_steps_succeeded:
        print("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("PIPELINE FAILED")
    print(f"{'='*80}")
    
    return 0 if all_steps_succeeded else 1

if __name__ == "__main__":
    sys.exit(main()) 