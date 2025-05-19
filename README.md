# Factor Return Forecasting with Dynamic Feature Selection - Project Checklist

This checklist outlines the main phases and steps for the project as described in the PRD.

# Factor Return Forecasting with Dynamic Feature Selection - Project Checklist

This checklist outlines the main phases and steps for the project as described in the PRD.

## Phase 1: Data Preparation
- [x] **Step 1: Load and Validate Data**
  - [x] Import `T2_Optimizer.xlsx` (or `.csv`) containing 302 months × 106 factors
  - [x] Verify data integrity and continuity
  - [x] Handle any missing values appropriately (fill with cross-sectional mean of factors for each date, log replacements)
  - [x] Sort by date in ascending order
  - [x] Generate data quality/completeness report (summaries and logs produced by script)
- [x] **Step 2: Calculate Moving Averages**
  - For each factor:
    - [x] 1-month MA (current value - original series)
    - [x] 3-month MA (short-term trend)
    - [x] 12-month MA (medium-term trend)
    - [x] 60-month MA (long-term trend)
  - [x] Outputs: `S2_T2_Optimizer_with_MA.xlsx`, `S2_MA_Visualization.pdf`, `S2_Column_Mapping.xlsx`
- [x] **Step 3: Create Benchmark Series**
  - [x] Calculate equal-weighted average of all 106 factors (and their MAs) for each month
  - [x] Store as separate series for easy comparison in `S3_Benchmark_Series.xlsx`
  - [x] Visualize benchmarks in `S3_Benchmark_Visualization.pdf`

## Phase 2: Rolling Window Framework
- [x] **Step 4 & 5: Define Window Structure and Create Schedule** (`Step_4_Define_Window_Structure.py`)
  - [x] Window Definition:
    - Training: 60 months of data
    - Validation: 6 months following training
    - Prediction: 1 month after validation
    - Total window: 67 months
  - [x] Window Schedule Creation:
    - First prediction: Month 67 (based on 0-indexed data)
    - Last prediction: Month 302 (based on 0-indexed data, assuming 302 total data points)
    - Total prediction windows: 236
    - Windows advance by 1 month each iteration
  - [x] Input: `S3_Benchmark_Series.xlsx` (for date range)
  - [x] Outputs: `S4_Window_Schedule.xlsx`, `S4_Window_Visualization.pdf`

## Phase 3: Dynamic Feature Engineering
- [x] **Step 6: Calculate Rolling Correlations** (`Step_6_Calculate_Rolling_Correlations.py`)
  - For each window (using 60-month training period):
    - [x] Calculate correlation matrix for all 106 factors (using their original values, not MAs, for correlation calculation with each other).
    - [x] Store correlations for feature selection in `S6_Rolling_Correlations.h5`.
    - [x] Visualize sample correlations in `S6_Correlation_Sample_Visualizations.pdf`.
- [x] **Step 7: Select Helper Features** (`Step_7_Select_Helper_Features.py`)
  - For each target factor (within each window):
    - [x] Rank all other factors by absolute correlation with the target factor (based on training period correlations from Step 6).
    - [x] Select top 10 most correlated factors (specifically their 60-month MAs are intended as helpers, though this script just identifies the factors).
    - [x] Store selected helper factors and their correlations in `S7_Helper_Features.h5`.
    - [x] Outputs: `S7_Helper_Features.h5`, `S7_Helper_Features_Sample.xlsx`, `S7_Helper_Features_Visualization.pdf`.
- [x] **Step 8: Create Feature Sets** (`Step_8_Create_Feature_Sets.py`)
  - For each factor (within each window):
    - [x] **Own factor MAs**: 1-month, 3-month, 12-month, 60-month (4 features).
    - [ ] ~~Helper factors: 60-month MAs of top 10 correlated factors (10 features)~~ - **NOTE: Step 8 currently only implements the 4 own-factor MAs. Helper features from Step 7 are not yet incorporated into `S8_Feature_Sets.h5`.**
    - [x] Total 4 features per factor model (in current `S8_Feature_Sets.h5`).
  - [x] Inputs: `S2_T2_Optimizer_with_MA.xlsx`, `S4_Window_Schedule.xlsx`, `S2_Column_Mapping.xlsx`.
  - [x] Outputs: `S8_Feature_Sets.h5` (containing 4 features per factor), `S8_Feature_Sets_Sample.xlsx`, `S8_Feature_Sets_Visualization.pdf`.

## Phase 4: Model Training and Prediction (Parallel Approaches)

### Fork A: XGBoost Models
- [x] **Step 9A: Configure XGBoost** (`Step_9_Configure_XGBoost.py`)
  - [x] Script defines and saves base XGBoost parameters. Current base configuration (from script v2.0) is:
    - max_depth: 3
    - n_estimators: 500 (with early stopping)
    - learning_rate: 0.01
    - subsample: 0.8
    - colsample_bytree: 0.9
    - objective: 'reg:squarederror'
    - eval_metric: ['rmse', 'mae']
    - early_stopping_rounds: 50
    - min_child_weight: 1
    - gamma: 0
    - alpha (L1): 0.01
    - lambda (L2): 1
    - random_state: 42
  - [x] Note: Parameters optimized for 4-dimensional feature sets (helper features not used).
  - [x] Outputs: `S9_XGBoost_Config.xlsx`, `S9_XGBoost_Config.json`.
- [x] **Step 10A: Tune XGBoost Hyperparameters** (`Step_10A_Tune_XGBoost_Hyperparameters.py`)
  - [x] Uses `NUM_TUNING_WINDOWS = 25` randomly selected windows for tuning.
  - [x] Uses `NUM_FACTORS_FOR_TUNING = 10` for faster tuning.
  - [ ] Exclude these tuning windows from final performance evaluation (Standard practice, to be ensured in Phase 6/7).
  - [x] Tests various parameter combinations (specific grid defined in script).
  - [x] Selects optimal configuration based on validation performance.
  - [x] Outputs: `S10A_XGBoost_Tuning_Results.xlsx`, `S10A_XGBoost_Optimal_Params.json`, `S10A_Tuning_Windows.pkl`, `S10A_Tuning_Visualization.pdf`.
  - [x] Input `S8_Feature_Sets.h5` (Note: README.md was `S8_Feature_Sets.pkl`, corrected to `.h5` as per script).
- [x] **Step 11A: Train XGBoost Models** (`Step_11A_Train_XGBoost_Models.py`)
  - [x] Trains factor-specific models for each window using optimal parameters from Step 10A.
  - [x] Uses 60-month training period and 6-month validation period (defined by `S8_Feature_Sets.h5`).
  - [x] Applies early stopping using validation loss.
  - [x] Stores trained models in `./output/S11A_XGBoost_Models/window_<id>/factor_<id>.joblib`.
  - [x] Outputs: Models in `S11A_XGBoost_Models/` directory, `S11A_Training_Log.log`.
- [x] **Step 12A: Generate XGBoost Predictions** (`Step_12A_Generate_XGBoost_Predictions.py`)
  - [x] Predicts next month return for all factors using trained models.
  - [x] Uses features from `S8_Feature_Sets.h5` for prediction.
  - [x] Outputs: `S12A_XGBoost_Predictions.h5`, `S12A_XGBoost_Predictions_Matrix.xlsx`, `S12A_Prediction_Log.log`.

### Fork B: Linear Models
- [x] **Step 9B: Configure Linear Models**
  - Models configured and tuned in Step 10B are:
    - [x] Ordinary Least Squares (OLS)
    - [x] Ridge Regression (with alpha parameters)
    - [x] LASSO Regression (with alpha parameters)
    - [x] Non-Negative Least Squares (NNLS)
  - **NOTE**: The project initially aimed to use ElasticNet, but current scripts (`Step_10B`, `Step_11B`, `Step_12B`) implement NNLS instead. This section reflects the NNLS implementation.
- [x] **Step 10B: Tune Linear Models** (`Step_10B_Tune_Linear_Models.py` v3.0)
  - [x] Tunes OLS, Ridge, LASSO, and NNLS using `NUM_TUNING_WINDOWS = 10` and `NUM_FACTORS_FOR_TUNING = 15`.
  - [x] Optimizes alpha parameters for Ridge and LASSO.
  - [x] Saves optimal parameters for *each model type* in `S10B_Linear_Models_Optimal_Params.json`.
  - [x] Inputs: `S8_Feature_Sets.h5` (4-feature sets).
  - [x] Outputs: `S10B_Linear_Models_Tuning_Results.xlsx`, `S10B_Linear_Models_Optimal_Params.json`, `S10B_Linear_Models_Tuning_Visualization.pdf`.
- [x] **Step 11B: Train Linear Models** (`Step_11B_Train_Linear_Models.py` v3.0)
  - [x] Trains OLS, Ridge, LASSO, and NNLS models for each factor-window, using optimal params from Step 10B.
  - [x] Uses 4-feature sets from `S8_Feature_Sets.h5` (not 14 features).
  - [x] Validates on 6-month period.
  - [x] Inputs: `S8_Feature_Sets.h5`, `S4_Window_Schedule.xlsx`, `S10B_Linear_Models_Optimal_Params.json`.
  - [x] Outputs: `S11B_Linear_Models.h5`, `S11B_Linear_Models_Training_Summary.xlsx`, `S11B_Linear_Models_Training_Visualization.pdf`.
- [x] **Step 12B: Generate Linear Model Predictions** (`Step_12B_Generate_Linear_Model_Predictions.py` v3.0)
  - [x] Predicts returns using trained OLS, Ridge, LASSO, and NNLS models.
  - [x] Uses features from `S8_Feature_Sets.h5` for prediction.
  - [x] Inputs: `S11B_Linear_Models.h5`, `S4_Window_Schedule.xlsx`, `S8_Feature_Sets.h5`.
  - [x] Outputs: `S12B_Linear_Model_Predictions.h5`, `S12B_Linear_Model_Predictions_Matrix.xlsx`, `S12B_Prediction_Visualization.pdf`.

### Fork C: Time Series Models
- [ ] **Step 9C: Configure Time Series Models**
  - [ ] ARIMA model parameters
  - [ ] Vector Autoregression (VAR) setup
- [ ] **Step 10C: Tune Time Series Models**
  - [ ] Optimize p, d, q parameters for ARIMA
  - [ ] Select optimal lag structure for VAR
- [ ] **Step 11C: Train Time Series Models**
  - [ ] Train 106 factor-specific time series models
  - [ ] Incorporate exogenous variables from feature sets
- [ ] **Step 12C: Generate Time Series Predictions**
  - [ ] Predict returns using time series approaches
  - [ ] Store predictions for portfolio construction

### Fork D: LSTM Neural Network Models
- [ ] **Step 9D: Configure LSTM Neural Networks**
  - [ ] Define sequence length (lookback_months=12)
  - [ ] Specify LSTM architecture (64 → 32 nodes)
  - [ ] Set dense layers (16 → 8 nodes)
  - [ ] Configure dropout rate (0.2)
  - [ ] Set batch normalization parameters
  - [ ] Define optimizer settings (Adam with learning_rate=0.001)
  - [ ] Set loss function (MSE) and metrics (MAE, MAPE)
  - [ ] Create configuration file (lstm_config.yaml)
- [ ] **Step 10D: Tune LSTM Hyperparameters**
  - [ ] Designate early windows exclusively for tuning
  - [ ] Test parameter combinations:
    - [ ] lookback_months: [6, 12, 18, 24]
    - [ ] lstm_units: [[64, 32], [128, 64], [32, 16]]
    - [ ] dense_units: [[16, 8], [32, 16], [8, 4]]
    - [ ] dropout_rate: [0.1, 0.2, 0.3]
    - [ ] learning_rate: [0.001, 0.0005, 0.0001]
    - [ ] batch_size: [16, 32, 64]
  - [ ] Use Bayesian optimization with Optuna framework (n_trials=50)
  - [ ] Select optimal configuration based on validation performance
- [ ] **Step 11D: Develop Temporal Feature Engineering**
  - [ ] Create LSTMFeatureEngineer class inheriting from base FeatureEngineer
  - [ ] Implement sequence generation for each feature set
  - [ ] Ensure feature sequences respect temporal order
  - [ ] Create compatible feature format for LSTM input (3D tensors)
- [ ] **Step 12D: Train LSTM Models**
  - [ ] Implement LSTM training with early stopping and LR reduction
  - [ ] Set up TensorBoard monitoring and checkpoint saving
  - [ ] Train 106 factor-specific LSTM models for each window
  - [ ] Use 12-month sequences of the same 14 features as XGBoost
  - [ ] Implement proper callbacks for training optimization
- [ ] **Step 13D: Generate LSTM Predictions**
  - [ ] Create prediction pipeline for LSTM models
  - [ ] Generate next-month return forecasts for all factors
  - [ ] Store predictions alongside other model outputs

## Phase 5: Portfolio Construction

- [ ] **Step 13: Create Factor Weights / Portfolio Optimization** (`Step_13_Create_Factor_Weights.py`)
  - [x] This script implements portfolio optimization using externally provided expected returns and historical covariance.
  - [x] Inputs (example standalone mode): `Factor_Alpha.xlsx` (for expected returns), `T2_Optimizer.xlsx` (for historical returns for covariance).
  - [x] Method: Mean-variance optimization with constraints (long-only, fully invested) and penalties (HHI for concentration).
  - [x] Uses a hybrid window for covariance estimation (expanding then rolling).
  - [x] Outputs: `output/S13_rolling_window_weights.xlsx`, `output/S13_strategy_statistics.xlsx`, `output/S13_turnover_analysis.pdf`, `output/S13_strategy_performance.pdf`.
  - **NOTE**: This script provides a sophisticated optimization method. It does not directly implement the simpler "select top N and equal-weight" strategy described in README Steps 14 & 15. It would use the *output* of models (predicted returns from Step 12A/B or an ensemble) as its `expected_returns_file` input.

- [ ] **Step 14: Model-Specific Factor Selection**
  - For each model type and each prediction month:
    - [ ] Rank all 106 factors by predicted returns
    - [ ] Select top 5 factors
    - [ ] Record selections for analysis
  - **NOTE**: No specific script named `Step_14_...py` found. This logic would be a precursor to or part of a portfolio construction strategy. `Step_13` could use these selections if its input `expected_returns_file` was pre-filtered.
- [ ] **Step 15: Portfolio Formation**
  - [ ] Create separate portfolios for each model type:
    - [ ] XGBoost Portfolio
    - [ ] Linear Models Portfolio (for each variant)
    - [ ] Time Series Models Portfolio (for each variant)
    - [ ] LSTM Portfolio
  - [ ] Equal-weight the 5 selected factors (20% each)
  - **NOTE**: No specific script named `Step_15_...py` found. `Step_13_Create_Factor_Weights.py` implements a more complex optimization than simple equal weighting.
- [ ] **Step 16: Ensemble Portfolio Construction**
  - [ ] Create meta-predictions by combining model outputs
  - [ ] Build ensemble portfolio based on combined predictions
  - [ ] Implement fixed-weight ensemble (50/50)
  - [ ] Implement dynamic weighting based on recent performance
  - [ ] Create stacked ensemble (meta-model approach)
  - **NOTE**: No specific script named `Step_16_...py` found. `Step_13_Create_Factor_Weights.py` could be used to construct portfolios from ensemble predictions if the ensemble predictions are provided as its input.

## Phase 6: Performance Measurement

- [ ] **Step 17: Monthly Performance Metrics**
  - For each model type:
    - [ ] Calculate monthly returns
    - [ ] Compare vs benchmark returns
    - [ ] Calculate monthly alpha (excess return)
    - [ ] Track win/loss vs benchmark
- [ ] **Step 18: Cumulative Performance Analysis**
  - For each model type:
    - [ ] Calculate hit rate (% months beating benchmark)
    - [ ] Calculate cumulative excess return (alpha)
    - [ ] Calculate average monthly alpha
    - [ ] Calculate volatility of excess returns
    - [ ] Calculate information ratio
- [ ] **Step 19: Risk Metrics**
  - For each model type:
    - [ ] Calculate maximum drawdown
    - [ ] Calculate downside deviation
    - [ ] Calculate Sharpe ratio
    - [ ] Compare risk-adjusted returns

## Phase 7: Analysis and Reporting

- [ ] **Step 20: Model Comparison Analysis**
  - [ ] Compare performance across all model types
  - [ ] Identify market regimes where each model excels
  - [ ] Analyze correlation between model predictions
  - [ ] Evaluate ensemble benefits vs individual models
- [ ] **Step 21: Factor Selection Analysis**
  - [ ] Analyze which factors were most frequently selected by each model
  - [ ] Identify factors with highest contribution to alpha
  - [ ] Compare factor selection patterns across model types
  - [ ] Analyze stability of factor selection over time
- [ ] **Step 22: Feature Importance Analysis**
  - [ ] Analyze which features were most important for XGBoost
  - [ ] Compare feature importance patterns across model types
  - [ ] Analyze temporal stability of important features
- [ ] **Step 23: Market Regime Analysis**
  - [ ] Analyze model performance in different market regimes
  - [ ] Identify optimal model for each regime type
  - [ ] Analyze drawdown periods and recovery patterns
- [ ] **Step 24: Generate Final Report**
  - [ ] Document methodology for all model variants
  - [ ] Report comparative results with visualizations
  - [ ] Provide model-specific findings and recommendations
  - [ ] Include ensemble model performance analysis
  - [ ] Recommend optimal configuration for production

## Implementation Workflow

- [ ] **Step 25: Parallel Processing Framework**
  - [ ] Set up infrastructure to run all model forks concurrently
  - [ ] Ensure consistent data handling across model types
  - [ ] Implement unified reporting framework
  - [ ] Create common evaluation platform
- [ ] **Step 26: Window-by-Window Processing**
  - For each window:
    - [ ] Extract training, validation, and prediction data
    - [ ] Process all model types in parallel
    - [ ] Generate predictions for all factors
    - [ ] Create model-specific portfolios
    - [ ] Calculate performance metrics
    - [ ] Store results for analysis
- [ ] **Step 27: Results Aggregation**
  - [ ] Compile performance metrics across all windows
  - [ ] Calculate cumulative performance statistics
  - [ ] Generate performance visualizations
  - [ ] Create model comparison reports

## Documentation and Testing

- [ ] **Step 28: Technical Documentation**
  - [ ] Document system architecture for all model types
  - [ ] Create data flow diagrams
  - [ ] Document hyperparameter tuning results
  - [ ] Create model-specific API documentation
- [ ] **Step 29: Testing Framework**
  - [ ] Implement unit tests for all model types
  - [ ] Create regression tests for performance tracking
  - [ ] Validate model comparison methodology
  - [ ] Test for data leakage in temporal feature engineering
- [ ] **Step 30: Backtesting Validation**
  - [ ] Ensure no lookahead bias in any model
  - [ ] Confirm strict separation between tuning and evaluation
  - [ ] Verify all models use proper point-in-time data
  - [ ] Validate portfolio construction methodology

## Monitoring and Production

- [ ] **Step 31: Model Persistence Framework**
  - [ ] Create system for saving/loading all model types
  - [ ] Implement model versioning and metadata tracking
  - [ ] Develop efficient prediction pipelines
- [ ] **Step 32: Performance Monitoring Dashboard**
  - [ ] Create interactive dashboard for model tracking
  - [ ] Implement real-time performance visualization
  - [ ] Set up model comparison views
  - [ ] Create factor selection analysis tools
- [ ] **Step 33: Production Deployment Guide**
  - [ ] Document environment setup for all model types
  - [ ] Create execution scripts for training and prediction
  - [ ] Develop monitoring and maintenance procedures
  - [ ] Create user documentation for system operation

**NOTE ON REMAINING STEPS:** Forks C (Time Series) and D (LSTM), as well as Phases 6 (Performance Measurement), 7 (Analysis/Reporting), and detailed Implementation/Documentation/Monitoring steps do not have corresponding `Step_XX_...py` scripts in the current project structure. They are marked as `[ ]` (not yet implemented or status unknown based on available scripts). 