Product Requirements Document (PRD)
Factor Return Forecasting with Dynamic Feature Selection
1. Executive Summary
Build a factor selection system using XGBoost with correlation-based dynamic feature selection. The system will identify the top 5 factors each month that are expected to outperform the equal-weighted average of all 106 factors. Success is measured by beating the benchmark, not prediction accuracy.
2. Project Overview
2.1 Objective

Select 5 factors monthly that outperform the equal-weighted average of all factors
Use rolling 60-month training windows with 6-month validation
Implement dynamic feature selection based on rolling correlations
Generate 20+ years of out-of-sample performance versus benchmark
Achieve >55% hit rate in beating the average

2.2 Success Metrics

Primary: Beat the 106-factor average return >55% of months
Secondary: Positive cumulative excess return over full period
Risk-Adjusted: Information ratio > 0.5
Consistency: Lower maximum drawdown than benchmark

3. System Architecture
Phase 1: Data Preparation
Step 1: Load and Validate Data

Import T2_Optimizer.csv containing 302 months × 106 factors
Verify data integrity and continuity
Handle any missing values appropriately
Sort by date in ascending order

Step 2: Calculate Moving Averages
For each factor:

1-month MA (current value)
3-month MA (short-term trend)
12-month MA (medium-term trend)
60-month MA (long-term trend)

Step 3: Create Benchmark Series

Calculate equal-weighted average of all 106 factors for each month
This becomes the benchmark to beat
Store as separate series for easy comparison

Phase 2: Rolling Window Framework
Step 1: Define Window Structure

Training: 60 months of data
Validation: 6 months following training
Prediction: 1 month after validation
Total window: 67 months

Step 2: Create Window Schedule

First prediction: Month 67
Last prediction: Month 302
Total predictions: 236 months (~20 years)
Windows advance by 1 month each iteration

Phase 3: Dynamic Feature Engineering
Step 1: Calculate Rolling Correlations
For each window:

Use 60-month training period
Calculate correlation matrix for all 106 factors
Store correlations for feature selection

Step 2: Select Helper Features
For each target factor:

Rank all other factors by absolute correlation
Select top 10 most correlated factors
These become dynamic features for prediction

Step 3: Create Feature Sets
Each factor gets 14 features:

Own factor: 1, 3, 12, 60-month MAs (4 features)
Helper factors: 60-month MAs of top 10 correlated factors (10 features)

Phase 4: Model Training
Step 1: Configure XGBoost
Parameters optimized for financial data:

Tree depth: 4 (prevent overfitting)
Trees: 500 with early stopping
Learning rate: 0.01
Subsample: 80%
Feature sample: 70%

Step 2: Train Factor Models
For each window and factor:

Train on 60 months with 14 features
Validate on next 6 months
Use early stopping on validation loss
Store trained model

Step 3: Generate Predictions

Predict next month return for all 106 factors
No need to store exact predictions
Only need ranking for selection

Phase 5: Portfolio Construction
Step 1: Factor Selection
Each month:

Rank all 106 factors by predicted returns
Select top 5 factors
Record which factors selected

Step 2: Portfolio Formation

Equal-weight the 5 selected factors (20% each)
This is the "Model Portfolio"

Step 3: Calculate Returns

Model Portfolio: Average return of 5 selected factors
Benchmark: Average return of all 106 factors
Excess Return: Model Portfolio - Benchmark

Phase 6: Performance Measurement
Step 1: Monthly Metrics
For each prediction month:

Model portfolio return
Benchmark return
Excess return (alpha)
Win/Loss vs benchmark

Step 2: Cumulative Metrics
Track over full test period:

Hit rate (% months beating benchmark)
Cumulative excess return
Average monthly alpha
Volatility of excess returns

Step 3: Risk Metrics

Maximum drawdown (Model vs Benchmark)
Downside deviation
Information ratio (alpha/tracking error)
Sharpe ratio comparison

Phase 7: Analysis and Reporting
Step 1: Performance Attribution

Which factors most frequently selected?
Which factors contribute most to alpha?
Performance by market regime

Step 2: Feature Analysis

Which helper factors appear most often?
Stability of correlations over time
Feature importance patterns

Step 3: Time Series Analysis

Rolling hit rates
Regime-dependent performance
Drawdown analysis

4. Implementation Workflow
Month-by-Month Process

Data Window Setup

Extract months 1-60 for training
Extract months 61-66 for validation
Target: predict month 67


Feature Engineering

Calculate correlations using months 1-60
For each factor, identify top 10 correlated factors
Create 14-dimensional feature vectors


Model Training

Train 106 separate XGBoost models
Each model predicts one factor
Use validation for early stopping


Factor Selection

Get predictions for all 106 factors
Rank by predicted return
Select top 5


Performance Calculation

Model Portfolio = average of 5 selected factors' actual returns
Benchmark = average of all 106 factors' actual returns
Alpha = Model Portfolio - Benchmark


Record Results

Store selected factors
Store returns and alpha
Update cumulative metrics


Advance Window

Move all windows forward by 1 month
Repeat process



5. Success Criteria
Primary Metrics

Beat benchmark >55% of months
Positive cumulative excess return
Information ratio >0.5

Secondary Metrics

Lower maximum drawdown than benchmark
Consistent hit rate across market regimes
Reasonable factor turnover

Risk Controls

No factor concentration (max 5 factors)
Equal weighting prevents single-factor dominance
Dynamic features adapt to changing markets

6. Key Design Decisions
Why Top 5?

Sufficient diversification
Meaningful concentration for alpha
Manageable for real trading

Why 60-Month Windows?

Captures multiple market cycles
Stable correlation estimates
Balance between adaptability and stability

Why Dynamic Features?

Relationships between factors change
Adapts to market regimes
Captures current market structure

7. Monitoring and Maintenance
Daily Monitoring

Verify data quality
Check prediction pipeline
Monitor system performance

Monthly Reviews

Performance vs benchmark
Factor selection patterns
Feature importance analysis

Quarterly Analysis

Deep dive on alpha sources
Correlation stability check
Model retraining decisions

8. Risk Management
Model Risks

Overfitting: Controlled by validation and simple models
Regime change: Handled by rolling windows
Correlation breakdown: Monitored monthly

Implementation Risks

Data quality: Daily validation checks
Calculation errors: Comprehensive testing
System failures: Backup procedures

9. Testing Protocol
Backtesting

Full historical test (236 months)
No lookahead bias
Proper point-in-time data

Validation Tests

Compare to simple benchmarks
Test parameter sensitivity
Verify feature importance

Stress Testing

Performance in crisis periods
Behavior with missing data
Correlation breakdown scenarios

10. Documentation Requirements
Technical Documentation

Complete system architecture
Data flow diagrams
API specifications

User Documentation

Operating procedures
Troubleshooting guide
Performance reports

Compliance Documentation

Audit trail
Decision rationale
Risk controls

This PRD focuses entirely on beating the benchmark average rather than prediction accuracy. The success of the system is measured by consistent outperformance of the equal-weighted portfolio of all factors.