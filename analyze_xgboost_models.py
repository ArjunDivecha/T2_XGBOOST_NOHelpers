#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_xgboost_models.py

This script analyzes multiple XGBoost models to identify patterns in feature importance
across different factors and windows.

INPUT FILES:
- ./output/S11A_XGBoost_Models/window_*/factor_*.joblib
  XGBoost model files saved in joblib format

OUTPUT FILES:
- ./output/xgboost_feature_importance_summary.xlsx
  Excel workbook with feature importance summaries
- ./output/xgboost_feature_importance_visualization.pdf
  PDF with visualizations of feature importance patterns

Usage:
    python analyze_xgboost_models.py [--window_list WINDOWS] [--factor_list FACTORS]

Example:
    python analyze_xgboost_models.py --window_list 1,2,3 --factor_list "12-1MTR_CS,1MTR_CS"
"""

import os
import sys
import glob
import re
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Default mapping for feature names based on factor naming conventions
def get_default_feature_names(factor_id):
    """Generate default feature names for a given factor."""
    # Clean up factor_id to get base name
    base_factor = factor_id
    for suffix in ['_CS', '_TS', '_3m', '_12m', '_60m']:
        base_factor = base_factor.replace(suffix, '')
    
    # Create feature names - only 4 dimensions (own MAs)
    feature_names = [
        f"{factor_id}_1m",
        f"{factor_id}_3m", 
        f"{factor_id}_12m",
        f"{factor_id}_60m"
    ]
    
    return feature_names

def extract_model_info(model_path):
    """Extract window ID and factor ID from model path."""
    window_id = None
    factor_id = None
    
    path_parts = model_path.split('/')
    for part in path_parts:
        if part.startswith('window_'):
            window_id = int(part.replace('window_', ''))
    
    match = re.search(r'factor_([^\.]+)\.joblib', model_path)
    if match:
        factor_id = match.group(1)
    
    return window_id, factor_id

def load_feature_names_from_h5(h5_path, window_id, factor_id):
    """Load feature names from feature sets H5 file if available."""
    import h5py
    
    feature_names = None
    
    if not os.path.exists(h5_path):
        return None
    
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Try different possible paths
            possible_paths = [
                f"feature_sets/window_{window_id}/training/{factor_id}/X/columns",
                f"window_{window_id}/training/{factor_id}/X/columns"
            ]
            
            for path in possible_paths:
                if path in h5f:
                    columns_data = h5f[path]
                    columns = columns_data[:]
                    
                    # If it's a byte string array, decode to UTF-8
                    if columns_data.dtype.kind in ['S', 'O']:
                        feature_names = [col.decode('utf-8') if isinstance(col, bytes) else col for col in columns]
                    else:
                        feature_names = list(columns)
                    
                    return feature_names
    except Exception as e:
        print(f"Error loading feature names from H5 file: {e}")
    
    return None

def analyze_model(model_path, feature_sets_path=None):
    """Load and analyze a single XGBoost model."""
    # Extract window ID and factor ID from path
    window_id, factor_id = extract_model_info(model_path)
    
    if not window_id or not factor_id:
        print(f"Warning: Could not extract window_id or factor_id from {model_path}")
        return None
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Get number of features
        n_features = len(model.feature_importances_)
        
        # Try to get feature names
        feature_names = None
        
        # First try from feature sets H5 file if provided
        if feature_sets_path:
            feature_names = load_feature_names_from_h5(feature_sets_path, window_id, factor_id)
        
        # If not found, use default generated names
        if not feature_names:
            feature_names = get_default_feature_names(factor_id)
            
            # Ensure we have the right number of features
            if len(feature_names) != n_features:
                # Adjust by adding or removing features
                if len(feature_names) < n_features:
                    for i in range(len(feature_names), n_features):
                        feature_names.append(f"Unknown_Feature_{i}")
                else:
                    feature_names = feature_names[:n_features]
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        df = pd.DataFrame({
            'window_id': window_id,
            'factor_id': factor_id,
            'feature_name': feature_names,
            'importance': importances,
            'importance_pct': importances / importances.sum() * 100,
            'feature_index': range(len(importances))
        })
        
        # Add tree count and other metadata
        df['n_trees'] = model.n_estimators
        df['best_iteration'] = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        df['actual_trees'] = df['best_iteration'] 
        
        # Extract feature type info
        df['is_target_feature'] = df['feature_name'].apply(lambda x: not x.startswith('helper_'))
        df['is_1m_MA'] = df['feature_name'].apply(lambda x: x.endswith('_1m'))
        df['is_3m_MA'] = df['feature_name'].apply(lambda x: x.endswith('_3m'))
        df['is_12m_MA'] = df['feature_name'].apply(lambda x: x.endswith('_12m'))
        df['is_60m_MA'] = df['feature_name'].apply(lambda x: x.endswith('_60m'))
        df['is_CS'] = df['feature_name'].apply(lambda x: '_CS' in x)
        df['is_TS'] = df['feature_name'].apply(lambda x: '_TS' in x)
        
        return df
    
    except Exception as e:
        print(f"Error analyzing model {model_path}: {e}")
        return None

def find_models(base_dir, window_list=None, factor_list=None):
    """Find joblib model files matching criteria."""
    pattern = os.path.join(base_dir, "window_*", "factor_*.joblib")
    
    all_models = glob.glob(pattern)
    print(f"Found {len(all_models)} model files total")
    
    # Filter by window if specified
    if window_list:
        window_patterns = [f"window_{w}" for w in window_list]
        filtered_models = []
        for model_path in all_models:
            if any(wp in model_path for wp in window_patterns):
                filtered_models.append(model_path)
        all_models = filtered_models
        print(f"After window filtering: {len(all_models)} models")
    
    # Filter by factor if specified
    if factor_list:
        factor_patterns = [f"factor_{f}.joblib" for f in factor_list]
        filtered_models = []
        for model_path in all_models:
            if any(fp in model_path for fp in factor_patterns):
                filtered_models.append(model_path)
        all_models = filtered_models
        print(f"After factor filtering: {len(all_models)} models")
    
    return all_models

def analyze_models(model_paths, feature_sets_path=None, max_models=None):
    """Analyze a list of model files."""
    # Limit the number of models to analyze if needed
    if max_models and len(model_paths) > max_models:
        print(f"Limiting analysis to {max_models} models (out of {len(model_paths)})")
        # Take a sample spread across the list
        step = len(model_paths) // max_models
        model_paths = model_paths[::step][:max_models]
    
    results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"Analyzing model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        result = analyze_model(model_path, feature_sets_path)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No models were successfully analyzed")
        return None
    
    # Combine all results
    df = pd.concat(results, ignore_index=True)
    return df

def summarize_feature_importance(df):
    """Create summary statistics for feature importance."""
    # Group by factor and feature
    factor_feature_summary = df.groupby(['factor_id', 'feature_name'])['importance_pct'].agg(
        ['mean', 'median', 'min', 'max', 'std', 'count']
    ).reset_index()
    
    # Pivot to get feature importance by factor
    pivot_df = df.pivot_table(
        index='factor_id',
        columns='feature_name',
        values='importance_pct',
        aggfunc='mean'
    ).fillna(0)
    
    # Overall feature importance
    overall_importance = df.groupby('feature_name')['importance_pct'].agg(
        ['mean', 'median', 'min', 'max', 'std', 'count']
    ).sort_values('mean', ascending=False).reset_index()
    
    # Feature type importance
    type_importance = pd.DataFrame({
        'feature_type': [
            'Target Features', 'Helper Features',
            '1m MA', '3m MA', '12m MA', '60m MA',
            'Cross-Sectional (CS)', 'Time-Series (TS)'
        ],
        'mean_importance': [
            df[df['is_target_feature']]['importance_pct'].mean(),
            df[~df['is_target_feature']]['importance_pct'].mean(),
            df[df['is_1m_MA']]['importance_pct'].mean(),
            df[df['is_3m_MA']]['importance_pct'].mean(),
            df[df['is_12m_MA']]['importance_pct'].mean(),
            df[df['is_60m_MA']]['importance_pct'].mean(),
            df[df['is_CS']]['importance_pct'].mean(),
            df[df['is_TS']]['importance_pct'].mean()
        ]
    })
    
    return {
        'factor_feature_summary': factor_feature_summary,
        'pivot': pivot_df,
        'overall': overall_importance,
        'type_importance': type_importance
    }

def visualize_feature_importance(df, summaries, output_path):
    """Create visualizations of feature importance patterns."""
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Create a multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(output_path) as pdf:
        # Page 1: Overall feature type importance
        plt.figure(figsize=(10, 6))
        type_df = summaries['type_importance'].sort_values('mean_importance', ascending=False)
        sns.barplot(x='feature_type', y='mean_importance', data=type_df)
        plt.title('Average Feature Importance by Feature Type', fontsize=14)
        plt.xlabel('')
        plt.ylabel('Mean Importance (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Page 2: Top 15 features overall
        plt.figure(figsize=(12, 8))
        top_features = summaries['overall'].head(15)
        sns.barplot(x='feature_name', y='mean', data=top_features)
        plt.title('Top 15 Features by Mean Importance', fontsize=14)
        plt.xlabel('')
        plt.ylabel('Mean Importance (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Page 3: Distribution of feature importance for target vs helper features
        plt.figure(figsize=(10, 6))
        target_data = df[df['is_target_feature']]['importance_pct']
        helper_data = df[~df['is_target_feature']]['importance_pct']
        
        plt.hist([target_data, helper_data], bins=20, alpha=0.7, 
                 label=['Target Features', 'Helper Features'])
        plt.title('Distribution of Feature Importance: Target vs Helper Features', fontsize=14)
        plt.xlabel('Importance (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Page 4: Heatmap of feature importance for top factors
        # Get top 20 factors with most models
        top_factors = df['factor_id'].value_counts().head(20).index.tolist()
        pivot_subset = summaries['pivot'].loc[top_factors]
        
        # Get top 15 features (columns) by mean importance
        top_features = summaries['overall'].head(15)['feature_name'].tolist()
        pivot_subset = pivot_subset[top_features]
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_subset, cmap='viridis', annot=False, fmt='.1f')
        plt.title('Feature Importance Heatmap: Top Factors x Top Features', fontsize=14)
        plt.xlabel('Feature')
        plt.ylabel('Factor')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Page 5: Box plot of MA timeframes
        ma_data = {
            '1m MA': df[df['is_1m_MA']]['importance_pct'],
            '3m MA': df[df['is_3m_MA']]['importance_pct'],
            '12m MA': df[df['is_12m_MA']]['importance_pct'],
            '60m MA': df[df['is_60m_MA']]['importance_pct']
        }
        
        plt.figure(figsize=(10, 6))
        plt.boxplot([ma_data[k] for k in ['1m MA', '3m MA', '12m MA', '60m MA']], 
                   labels=['1m MA', '3m MA', '12m MA', '60m MA'])
        plt.title('Distribution of Feature Importance by Moving Average Timeframe', fontsize=14)
        plt.xlabel('Moving Average Timeframe')
        plt.ylabel('Importance (%)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Page 6: Box plot of CS vs TS
        cs_ts_data = {
            'Cross-Sectional (CS)': df[df['is_CS']]['importance_pct'],
            'Time-Series (TS)': df[df['is_TS']]['importance_pct']
        }
        
        plt.figure(figsize=(10, 6))
        plt.boxplot([cs_ts_data[k] for k in ['Cross-Sectional (CS)', 'Time-Series (TS)']], 
                   labels=['Cross-Sectional (CS)', 'Time-Series (TS)'])
        plt.title('Distribution of Feature Importance: CS vs TS', fontsize=14)
        plt.xlabel('Feature Type')
        plt.ylabel('Importance (%)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def save_results_to_excel(df, summaries, output_path):
    """Save analysis results to Excel."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Raw data
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Feature importance by factor and feature
        summaries['factor_feature_summary'].to_excel(
            writer, sheet_name='Factor-Feature Summary', index=False)
        
        # Feature importance pivot table
        summaries['pivot'].to_excel(writer, sheet_name='Pivot Table')
        
        # Overall feature importance
        summaries['overall'].to_excel(
            writer, sheet_name='Overall Importance', index=False)
        
        # Feature type importance
        summaries['type_importance'].to_excel(
            writer, sheet_name='Type Importance', index=False)
        
        # Create a summary sheet
        summary_df = pd.DataFrame({
            'Metric': [
                'Analysis Date',
                'Number of Models Analyzed',
                'Number of Factors',
                'Number of Windows',
                'Most Important Feature Type',
                'Most Important Feature',
                'Target vs Helper Ratio',
                'CS vs TS Ratio'
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                df['factor_id'].count() // df['feature_name'].nunique(),
                df['factor_id'].nunique(),
                df['window_id'].nunique(),
                summaries['type_importance'].sort_values('mean_importance', ascending=False)['feature_type'].iloc[0],
                summaries['overall']['feature_name'].iloc[0],
                f"{df[df['is_target_feature']]['importance_pct'].mean():.2f} : {df[~df['is_target_feature']]['importance_pct'].mean():.2f}",
                f"{df[df['is_CS']]['importance_pct'].mean():.2f} : {df[df['is_TS']]['importance_pct'].mean():.2f}"
            ]
        })
        
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze XGBoost model feature importance.')
    
    parser.add_argument('--window_list', type=str, default=None,
                        help='Comma-separated list of window IDs to analyze')
    
    parser.add_argument('--factor_list', type=str, default=None,
                        help='Comma-separated list of factor IDs to analyze')
    
    parser.add_argument('--max_models', type=int, default=1000,
                        help='Maximum number of models to analyze')
    
    parser.add_argument('--models_dir', type=str, default='output/S11A_XGBoost_Models',
                        help='Directory containing model files')
    
    parser.add_argument('--feature_sets', type=str, default='output/S8_Feature_Sets.h5',
                        help='Path to feature sets H5 file')
    
    parser.add_argument('--output_prefix', type=str, default='output/xgboost_feature_importance',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Process window_list if provided
    if args.window_list:
        args.window_list = [w.strip() for w in args.window_list.split(',')]
    
    # Process factor_list if provided
    if args.factor_list:
        args.factor_list = [f.strip() for f in args.factor_list.split(',')]
    
    return args

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Find model files
    model_paths = find_models(args.models_dir, args.window_list, args.factor_list)
    
    if not model_paths:
        print("No model files found matching criteria")
        return
    
    # Analyze models
    df = analyze_models(model_paths, args.feature_sets, args.max_models)
    
    if df is None:
        print("Analysis failed")
        return
    
    print(f"Successfully analyzed {df['factor_id'].count() // df['feature_name'].nunique()} models")
    
    # Create summaries
    summaries = summarize_feature_importance(df)
    
    # Save results to Excel
    excel_path = f"{args.output_prefix}_summary.xlsx"
    save_results_to_excel(df, summaries, excel_path)
    print(f"Results saved to {excel_path}")
    
    # Create visualizations
    viz_path = f"{args.output_prefix}_visualization.pdf"
    visualize_feature_importance(df, summaries, viz_path)
    print(f"Visualizations saved to {viz_path}")

if __name__ == "__main__":
    main()
