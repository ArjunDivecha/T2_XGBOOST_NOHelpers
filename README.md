# Factor Return Forecasting with Dynamic Feature Selection - Project Checklist

This checklist outlines the main phases and steps for the project as described in the PRD.

# Factor Return Forecasting with Dynamic Feature Selection - Project Checklist

This checklist outlines the main phases and steps for the project as described in the PRD.

## Phase 1: Data Preparation
- [x] **Step 1: Load and Validate Data**
  - [x] Import `T2_Optimizer.xlsx` (or `.csv`) containing 302 months × 106 factors
  - [x] Verify data integrity and continuity
  - [x] Handle any missing values appropriately (fill with mean of available countries, log replacements)
  - [x] Sort by date in ascending order
  - [x] Generate data quality/completeness report
- [x] **Step 2: Calculate Moving Averages**
  - For each factor:
    - [x] 1-month MA (current value)
    - [x] 3-month MA (short-term trend)
    - [x] 12-month MA (medium-term trend)
    - [x] 60-month MA (long-term trend)
- [x] **Step 3: Create Benchmark Series**
  - [x] Calculate equal-weighted average of all 106 factors for each month
  - [x] Store as separate series for easy comparison

## Phase 2: Rolling Window Framework
- [x] **Step 4: Define Window Structure**
  - [x] Training: 60 months of data
  - [x] Validation: 6 months following training
  - [x] Prediction: 1 month after validation
  - [x] Total window: 67 months
- [x] **Step 5: Create Window Schedule**
  - [x] First prediction: Month 67
  - [x] Last prediction: Month 302
  - [x] Total predictions: 236 months (~20 years)
  - [x] Windows advance by 1 month each iteration

## Phase 3: Dynamic Feature Engineering
- [x] **Step 6: Calculate Rolling Correlations**
  - For each window (using 60-month training period):
    - [x] Calculate correlation matrix for all 106 factors
    - [x] Store correlations for feature selection
- [x] **Step 7: Select Helper Features**
  - For each target factor (within each window):
    - [x] Rank all other factors by absolute correlation (based on training period)
    - [x] Select top 10 most correlated factors
- [x] **Step 8: Create Feature Sets**
  - For each factor (within each window):
    - [x] Own factor: 1, 3, 12, 60-month MAs (4 features)
    - [x] Helper factors: 60-month MAs of top 10 correlated factors (10 features)
    - [x] Total 14 features per factor model

## Phase 4: Model Training and Prediction (Parallel Approaches)

### Fork A: XGBoost Models
- [x] **Step 9A: Configure XGBoost**
  - [x] Tree depth: 4
  - [x] Trees: 500 with early stopping
  - [x] Learning rate: 0.01
  - [x] Subsample: 80%
  - [x] Feature sample: 70%
- [x] **Step 10A: Tune XGBoost Hyperparameters**
  - [x] Designate early windows (first 15-20) exclusively for tuning
  - [ ] Exclude these tuning windows from final performance evaluation
  - [x] Test parameter combinations:
    - [x] max_depth: [3, 4, 5, 6]
    - [x] learning_rate: [0.005, 0.01, 0.02]
    - [x] subsample: [0.7, 0.8, 0.9]
    - [x] colsample_bytree: [0.6, 0.7, 0.8]
    - [x] min_child_weight: [1, 3, 5]
  - [x] Select optimal configuration based on validation performance
  - [x] Document tuning results and configuration rationale
- [ ] **Step 11A: Train XGBoost Models**
  - [ ] Train 106 factor-specific models for each window
  - [ ] Use 60-month training period
  - [ ] Validate on 6-month period
  - [ ] Apply early stopping using validation loss
  - [ ] Store trained model parameters
- [ ] **Step 12A: Generate XGBoost Predictions**
  - [ ] Predict next month return for all factors
  - [ ] Store predictions for portfolio construction

### Fork B: Linear Models
- [ ] **Step 9B: Configure Linear Models**
  - [ ] Ordinary Least Squares (OLS)
  - [ ] Ridge Regression (with alpha parameters)
  - [ ] LASSO Regression (with alpha parameters)
  - [ ] Non-Negative Least Squares (NNLS)
- [x] **Step 10B: Tune Linear Models**
  - [x] Test different regularization strengths
  - [x] Optimize alpha parameters for Ridge and LASSO
  - [x] Select best linear model variant for each factor
- [x] **Step 11B: Train Linear Models**
  - [x] Train 106 factor-specific models of each type
  - [x] Use same 14-feature sets as XGBoost
  - [x] Validate on 6-month period
- [ ] **Step 12B: Generate Linear Model Predictions**
  - [ ] Predict returns using all linear model variants
  - [ ] Store predictions for portfolio construction

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

- [ ] **Step 14: Model-Specific Factor Selection**
  - For each model type and each prediction month:
    - [ ] Rank all 106 factors by predicted returns
    - [ ] Select top 5 factors
    - [ ] Record selections for analysis
- [ ] **Step 15: Portfolio Formation**
  - [ ] Create separate portfolios for each model type:
    - [ ] XGBoost Portfolio
    - [ ] Linear Models Portfolio (for each variant)
    - [ ] Time Series Models Portfolio (for each variant)
    - [ ] LSTM Portfolio
  - [ ] Equal-weight the 5 selected factors (20% each)
- [ ] **Step 16: Ensemble Portfolio Construction**
  - [ ] Create meta-predictions by combining model outputs
  - [ ] Build ensemble portfolio based on combined predictions
  - [ ] Implement fixed-weight ensemble (50/50)
  - [ ] Implement dynamic weighting based on recent performance
  - [ ] Create stacked ensemble (meta-model approach)

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