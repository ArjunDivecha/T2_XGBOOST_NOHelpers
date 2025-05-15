# File Documentation
'''
----------------------------------------------------------------------------------------------------
INPUT FILES:
- S3_Benchmark_Series.xlsx
  - Path: ./output/S3_Benchmark_Series.xlsx (output from Step 3)
  - Description: Excel file containing the benchmark series.
                 Used here only to obtain the date range of our dataset.
  - Format: Excel (.xlsx) with a header row. 'Date' column as dates.

OUTPUT FILES:
- S4_Window_Schedule.xlsx
  - Path: ./output/S4_Window_Schedule.xlsx
  - Description: Excel file containing the schedule of all 236 rolling windows.
                 Each window has training (60 months), validation (6 months), 
                 and prediction (1 month) periods.
  - Format: Excel (.xlsx) with a header row and one row per window (236 rows).

- S4_Window_Visualization.pdf
  - Path: ./output/S4_Window_Visualization.pdf
  - Description: PDF file with visualizations of the rolling window structure.
  - Format: PDF with example windows and timeline.

----------------------------------------------------------------------------------------------------
Purpose:
This script defines the rolling window structure for the factor return forecasting project.
Each window consists of 67 months of data:
- 60 months for training
- 6 months for validation
- 1 month for prediction
The script creates a schedule for all 236 windows with their respective date ranges.

Version: 1.0
Last Updated: Current Date
----------------------------------------------------------------------------------------------------
'''

# Section: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
import sys

# Section: Utility Functions
def create_window_schedule(dates, training_months=60, validation_months=6, prediction_months=1):
    """
    Create a schedule of rolling windows.
    
    Parameters:
    -----------
    dates : list or pd.Series
        List of dates in the dataset.
    training_months : int
        Number of months for training (default: 60).
    validation_months : int
        Number of months for validation (default: 6).
    prediction_months : int
        Number of months for prediction (default: 1).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the window schedule.
    """
    # Convert to pandas Series of dates if it's not already
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    # Calculate window size
    window_size = training_months + validation_months + prediction_months
    
    # Calculate maximum number of windows
    max_windows = len(dates) - window_size + 1
    
    # Initialize empty DataFrame for window schedule
    window_schedule = []
    
    for i in range(max_windows):
        # Define window boundaries
        training_start_idx = i
        training_end_idx = i + training_months - 1
        validation_start_idx = training_end_idx + 1
        validation_end_idx = validation_start_idx + validation_months - 1
        prediction_idx = validation_end_idx + 1
        
        # Get corresponding dates
        training_start_date = dates.iloc[training_start_idx]
        training_end_date = dates.iloc[training_end_idx]
        validation_start_date = dates.iloc[validation_start_idx]
        validation_end_date = dates.iloc[validation_end_idx]
        prediction_date = dates.iloc[prediction_idx]
        
        # Create window record
        window = {
            'Window_ID': i + 1,
            'Training_Start_Date': training_start_date,
            'Training_End_Date': training_end_date,
            'Validation_Start_Date': validation_start_date,
            'Validation_End_Date': validation_end_date,
            'Prediction_Date': prediction_date,
            'Training_Start_Index': training_start_idx,
            'Training_End_Index': training_end_idx,
            'Validation_Start_Index': validation_start_idx,
            'Validation_End_Index': validation_end_idx,
            'Prediction_Index': prediction_idx,
            'Training_Months': training_months,
            'Validation_Months': validation_months,
            'Total_Window_Months': window_size
        }
        
        window_schedule.append(window)
    
    # Convert to DataFrame
    window_schedule_df = pd.DataFrame(window_schedule)
    
    return window_schedule_df

def create_window_visualizations(window_schedule, output_file):
    """
    Create visualizations of the rolling window structure.
    
    Parameters:
    -----------
    window_schedule : pandas.DataFrame
        DataFrame containing the window schedule.
    output_file : str
        Path to output PDF file.
    """
    # Set plot style
    plt.style.use('ggplot')
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Rolling Window Structure\nFactor Return Forecasting Project',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20)
        plt.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        plt.text(0.5, 0.3, f'Total Windows: {len(window_schedule)}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        pdf.savefig()
        plt.close()
        
        # Create a visual of window structure
        plt.figure(figsize=(11, 8.5))
        plt.plot([0, 60, 60, 66, 66, 67], [0, 0, 0.5, 0.5, 1, 1], 'b-', linewidth=2)
        plt.text(30, 0.1, 'Training (60 months)', horizontalalignment='center')
        plt.text(63, 0.6, 'Validation\n(6 months)', horizontalalignment='center')
        plt.text(66.5, 1.1, 'Prediction\n(1 month)', horizontalalignment='center')
        plt.axvline(x=60, color='r', linestyle='--')
        plt.axvline(x=66, color='r', linestyle='--')
        plt.xlim(0, 70)
        plt.ylim(0, 1.3)
        plt.title('Window Structure (67 months total)')
        plt.xlabel('Month')
        plt.yticks([])
        pdf.savefig()
        plt.close()
        
        # Create a visualization of rolling windows
        plt.figure(figsize=(11, 8.5))
        for i in range(min(10, len(window_schedule))):  # Show first 10 windows
            plt.plot([0, 67], [i, i], 'k-', alpha=0.3)
            plt.plot([0, 60], [i, i], 'b-', linewidth=2)
            plt.plot([60, 66], [i, i], 'g-', linewidth=2)
            plt.plot([66, 67], [i, i], 'r-', linewidth=2)
            
            # Add window number
            plt.text(-3, i, f'W{i+1}', horizontalalignment='right')
        
        plt.axvline(x=0, color='k', linestyle='-')
        plt.axvline(x=60, color='k', linestyle='--')
        plt.axvline(x=66, color='k', linestyle='--')
        plt.axvline(x=67, color='k', linestyle='--')
        
        plt.text(30, 10.5, 'Training', horizontalalignment='center')
        plt.text(63, 10.5, 'Validation', horizontalalignment='center')
        plt.text(66.5, 10.5, 'Pred', horizontalalignment='center')
        
        plt.xlim(-5, 70)
        plt.ylim(-0.5, 11)
        plt.title('First 10 Rolling Windows (1-month step)')
        plt.xlabel('Month')
        plt.yticks([])
        plt.grid(False)
        pdf.savefig()
        plt.close()
        
        # Create a visualization of the date coverage
        first_window = window_schedule.iloc[0]
        last_window = window_schedule.iloc[-1]
        
        plt.figure(figsize=(11, 8.5))
        plt.plot([first_window['Training_Start_Date'], last_window['Prediction_Date']], [0, 0], 'k-', linewidth=2)
        
        # Mark first window
        plt.axvline(x=first_window['Training_Start_Date'], color='g', linestyle='-')
        plt.axvline(x=first_window['Prediction_Date'], color='r', linestyle='-')
        plt.text(first_window['Training_Start_Date'], 0.1, 'First Window Start', rotation=90)
        plt.text(first_window['Prediction_Date'], 0.1, 'First Prediction', rotation=90)
        
        # Mark last window
        plt.axvline(x=last_window['Training_Start_Date'], color='b', linestyle='-')
        plt.axvline(x=last_window['Prediction_Date'], color='r', linestyle='-')
        plt.text(last_window['Training_Start_Date'], 0.1, 'Last Window Start', rotation=90)
        plt.text(last_window['Prediction_Date'], 0.1, 'Last Prediction', rotation=90)
        
        plt.title('Full Date Coverage of All Windows')
        plt.ylabel('')
        plt.yticks([])
        plt.grid(True, axis='x')
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Add a summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Summary:\n\n'
                          f'Total Windows: {len(window_schedule)}\n\n'
                          f'First Window:\n'
                          f'  Training: {first_window["Training_Start_Date"].strftime("%Y-%m-%d")} to {first_window["Training_End_Date"].strftime("%Y-%m-%d")}\n'
                          f'  Validation: {first_window["Validation_Start_Date"].strftime("%Y-%m-%d")} to {first_window["Validation_End_Date"].strftime("%Y-%m-%d")}\n'
                          f'  Prediction: {first_window["Prediction_Date"].strftime("%Y-%m-%d")}\n\n'
                          f'Last Window:\n'
                          f'  Training: {last_window["Training_Start_Date"].strftime("%Y-%m-%d")} to {last_window["Training_End_Date"].strftime("%Y-%m-%d")}\n'
                          f'  Validation: {last_window["Validation_Start_Date"].strftime("%Y-%m-%d")} to {last_window["Validation_End_Date"].strftime("%Y-%m-%d")}\n'
                          f'  Prediction: {last_window["Prediction_Date"].strftime("%Y-%m-%d")}\n',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
        pdf.savefig()
        plt.close()

# Section: Main Script Logic
def main():
    print("=== Step 4: Define Window Structure ===")
    
    # --- Step 4.1: Set Window Parameters ---
    print("--- 4.1 Setting Window Parameters ---")
    
    training_months = 60  # 5 years
    validation_months = 6  # 6 months
    prediction_months = 1  # 1 month
    total_window_months = training_months + validation_months + prediction_months
    
    print(f"Window structure defined as:")
    print(f"  - Training: {training_months} months")
    print(f"  - Validation: {validation_months} months")
    print(f"  - Prediction: {prediction_months} month")
    print(f"  - Total window: {total_window_months} months")
    
    # --- Step 4.2: Get Date Range from Previous Step ---
    print("\n--- 4.2 Loading Date Range ---")
    
    # Define file paths
    input_file = os.path.join("output", "S3_Benchmark_Series.xlsx")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define output files with S4 prefix in output directory
    output_file = os.path.join(output_dir, "S4_Window_Schedule.xlsx")
    visualization_file = os.path.join(output_dir, "S4_Window_Visualization.pdf")
    
    try:
        # We only need the dates, not the full data
        df = pd.read_excel(input_file)
        print(f"Loaded data from {input_file}")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)
    
    # Identify the date column
    date_column_name = "Date"  # Assuming this is the name from previous steps
    
    if date_column_name not in df.columns:
        print(f"Error: Date column '{date_column_name}' not found in the file. Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    # Ensure date column is in datetime format
    df[date_column_name] = pd.to_datetime(df[date_column_name])
    
    # Sort by date
    df = df.sort_values(by=date_column_name)
    
    # Get date range
    first_date = df[date_column_name].min()
    last_date = df[date_column_name].max()
    total_months = len(df)
    
    print(f"Dataset spans from {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"Total months in dataset: {total_months}")
    
    # Check if we have enough data for at least one window
    if total_months < total_window_months:
        print(f"ERROR: Not enough data for even one window. Need at least {total_window_months} months, but only have {total_months}.")
        sys.exit(1)
    
    # --- Step 4.3: Create Window Schedule ---
    print("\n--- 4.3 Creating Window Schedule ---")
    
    window_schedule = create_window_schedule(
        df[date_column_name],
        training_months=training_months,
        validation_months=validation_months,
        prediction_months=prediction_months
    )
    
    num_windows = len(window_schedule)
    print(f"Created schedule with {num_windows} windows")
    
    first_prediction = window_schedule.iloc[0]['Prediction_Date']
    last_prediction = window_schedule.iloc[-1]['Prediction_Date']
    print(f"First prediction month: {first_prediction.strftime('%Y-%m-%d')}")
    print(f"Last prediction month: {last_prediction.strftime('%Y-%m-%d')}")
    
    # --- Step 4.4: Save Window Schedule ---
    print("\n--- 4.4 Saving Window Schedule ---")
    window_schedule.to_excel(output_file, index=False)
    print(f"Saved window schedule to {output_file}")
    
    # --- Step 4.5: Generate Visualizations ---
    print("\n--- 4.5 Generating Visualizations ---")
    create_window_visualizations(window_schedule, visualization_file)
    print(f"Generated window visualizations in {visualization_file}")
    
    # --- Step 4.6: Summary ---
    print("\n--- 4.6 Summary ---")
    print(f"Total windows created: {num_windows}")
    print(f"Each window consists of:")
    print(f"  - {training_months} months of training data")
    print(f"  - {validation_months} months of validation data")
    print(f"  - {prediction_months} month of prediction")
    print(f"First window:")
    print(f"  - Training: {window_schedule.iloc[0]['Training_Start_Date'].strftime('%Y-%m-%d')} to {window_schedule.iloc[0]['Training_End_Date'].strftime('%Y-%m-%d')}")
    print(f"  - Validation: {window_schedule.iloc[0]['Validation_Start_Date'].strftime('%Y-%m-%d')} to {window_schedule.iloc[0]['Validation_End_Date'].strftime('%Y-%m-%d')}")
    print(f"  - Prediction: {window_schedule.iloc[0]['Prediction_Date'].strftime('%Y-%m-%d')}")
    print(f"Last window:")
    print(f"  - Training: {window_schedule.iloc[-1]['Training_Start_Date'].strftime('%Y-%m-%d')} to {window_schedule.iloc[-1]['Training_End_Date'].strftime('%Y-%m-%d')}")
    print(f"  - Validation: {window_schedule.iloc[-1]['Validation_Start_Date'].strftime('%Y-%m-%d')} to {window_schedule.iloc[-1]['Validation_End_Date'].strftime('%Y-%m-%d')}")
    print(f"  - Prediction: {window_schedule.iloc[-1]['Prediction_Date'].strftime('%Y-%m-%d')}")
    
    print("\nStep 4 completed successfully!")

if __name__ == "__main__":
    main() 