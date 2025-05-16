# Factor Return Forecasting with Dynamic Feature Selection - Project Checklist

This checklist outlines the main phases and steps for the project as described in the PRD.

## Phase1: Data Preparation
- [x] **Step1: Load and Validate Data**
- [x] Import `T2_Optimizer.xlsx` (or `.csv`) containing302 months ×106 factors
- [x] Verify data integrity and continuity
- [x] Handle any missing values appropriately (fill with mean of available countries, log replacements)
- [x] Sort by date in ascending order
- [x] Generate data quality/completeness report
- [x] **Step2: Calculate Moving Averages**
- For each factor:
  - [x]1-month MA (current value)
  - [x]3-month MA (short-term trend)
  - [x]12-month MA (medium-term trend)
  - [x]60-month MA (long-term trend)
- [x] **Step3: Create Benchmark Series**
- [x] Calculate equal-weighted average of all106 factors for each month
- [x] Store as separate series for easy comparison

## Phase2: Rolling Window Framework
- [x] **Step4: Define Window Structure**
- [x] Training:60 months of data
- [x] Validation:6 months following training
- [x] Prediction:1 month after validation
- [x] Total window:67 months
- [x] **Step5: Create Window Schedule**
- [x] First prediction: Month67
- [x] Last prediction: Month302
- [x] Total predictions:236 months (~20 years)
- [x] Windows advance by1 month each iteration

## Phase3: Dynamic Feature Engineering
- [x] **Step6: Calculate Rolling Correlations**
- For each window (using60-month training period):
  - [x] Calculate correlation matrix for all106 factors
  - [x] Store correlations for feature selection
- [x] **Step7: Select Helper Features**
- For each target factor (within each window):
  - [x] Rank all other factors by absolute correlation (based on training period)
  - [x] Select top10 most correlated factors
- [x] **Step8: Create Feature Sets**
- For each factor (within each window):
  - [x] Own factor:1,3,12,60-month MAs (4 features)
  - [x] Helper factors:60-month MAs of top10 correlated factors (10 features)
  - [x] Total14 features per factor model

## Phase4: Model Training and Prediction (Parallel Approaches)

### Fork A: XGBoost Models
- [x] **Step9A: Configure XGBoost**
- [x] Tree depth:4
- [x] Trees:500 with early stopping
- [x] Learning rate:0.01
- [x] Subsample:80%
- [x] Feature sample:70%
- [x] **Step10A: Tune XGBoost Hyperparameters**
- [x] Randomly select 25 windows from the entire timescale for tuning
- [x] Exclude these tuning windows from final performance evaluation
- [x] Test parameter combinations:
  - [x] max_depth: [3,4,5,6]
  - [x] learning_rate: [0.005,0.01,0.02]
  - [x] subsample: [0.7,0.8,0.9]
  - [x] colsample_bytree: [0.6,0.7,0.8]
  - [x] min_child_weight: [1,3,5]
- [x] Select optimal configuration based on validation performance
- [x] Document tuning results and configuration rationale
- [ ] **Step11A: Train XGBoost Models**
- [ ] Train106 factor-specific models for each window
- [ ] Use60-month training period
- [ ] Validate on6-month period
- [ ] Apply early stopping using validation loss
- [ ] Store trained model parameters
- [ ] **Step12A: Generate XGBoost Predictions**
- [ ] Predict next month return for all factors
- [ ] Store predictions for portfolio construction

## Hyperparameter Tuning Options

For Step10A, the following hyperparameter tuning options are available:
- max_depth: [3,4,5,6] (Tree depth)
- learning_rate: [0.005,0.01,0.02] (Step size shrinkage)
- subsample: [0.7,0.8,0.9] (Fraction of samples used for fitting trees)
- colsample_bytree: [0.6,0.7,0.8] (Fraction of features used for building each tree)
- min_child_weight: [1,3,5] (Minimum sum of instance weight in a child)

These parameters can be adjusted to optimize the XGBoost model's performance on the validation set.