# XGBoost Model Feature Analysis Tools

This repository contains a set of tools for analyzing and understanding XGBoost models and their features in the T2_XGBOOST_STANDALONE project.

## Overview

The tools in this collection help you:

1. **Inspect individual XGBoost models** - Examine parameters, feature importance, and tree structure
2. **Analyze feature importance across models** - Identify patterns in feature importance across different factors and windows
3. **Inspect H5 files** - Explore the structure of feature sets in H5 files

## Scripts

### 1. `inspect_xgboost_model.py`

A tool for inspecting a single XGBoost model saved in joblib format.

**Usage:**
```bash
python inspect_xgboost_model.py <path_to_model_file>
```

**Example:**
```bash
python inspect_xgboost_model.py output/S11A_XGBoost_Models/window_1/factor_12-1MTR_CS.joblib
```

**Output:**
- Prints detailed model information including parameters, feature importance, and tree structure
- Generates a feature importance visualization in `output/feature_importance.pdf`

### 2. `analyze_xgboost_models.py`

A tool for analyzing feature importance across multiple XGBoost models.

**Usage:**
```bash
python analyze_xgboost_models.py [--window_list WINDOWS] [--factor_list FACTORS] [--max_models MAX]
```

**Examples:**
```bash
# Analyze 100 random models
python analyze_xgboost_models.py --max_models 100

# Analyze specific windows and factors
python analyze_xgboost_models.py --window_list 1,2,3 --factor_list "12-1MTR_CS,1MTR_CS"
```

**Output:**
- Excel workbook with feature importance analysis: `output/xgboost_feature_importance_summary.xlsx`
- PDF with visualizations: `output/xgboost_feature_importance_visualization.pdf`

### 3. `inspect_h5.py` and `inspect_h5_deeper.py`

Tools for inspecting the structure of H5 files containing feature sets.

**Usage:**
```bash
python inspect_h5.py <path_to_h5_file>
python inspect_h5_deeper.py <h5_file_path> <window_id> <factor_id>
```

**Examples:**
```bash
python inspect_h5.py output/S8_Feature_Sets.h5
python inspect_h5_deeper.py output/S8_Feature_Sets.h5 1 "12-1MTR_CS"
```

## Feature Structure Summary

Each XGBoost model in this project uses **4 features** organized as follows:

1. **4 features from the target factor itself**:
   - 1-month moving average (`<factor>_1m`)
   - 3-month moving average (`<factor>_3m`)
   - 12-month moving average (`<factor>_12m`)
   - 60-month moving average (`<factor>_60m`)

## Feature Name Suffixes

Feature names include suffixes that indicate their type:

- **CS**: Cross-Sectional normalization (comparing across countries at a point in time)
- **TS**: Time-Series normalization (comparing across time for a single country)
- **_1m, _3m, _12m, _60m**: Moving average window lengths (1-month, 3-month, 12-month, 60-month)

## Getting Feature Names

When looking at feature importance, you can get the actual feature names using:

1. The `inspect_xgboost_model.py` script, which will automatically try to find feature names
2. The `inspect_h5_deeper.py` script which can extract feature names for a specific window and factor
3. The `xgboost_model_features_summary.md` document which explains the feature naming conventions

## Visualizing Feature Importance

To generate visualizations of feature importance:

1. For a single model:
   ```bash
   python inspect_xgboost_model.py output/S11A_XGBoost_Models/window_1/factor_12-1MTR_CS.joblib
   ```

2. For multiple models:
   ```bash
   python analyze_xgboost_models.py --max_models 100
   ```

The visualizations show the relative importance of different features, feature types, and patterns across factors. 