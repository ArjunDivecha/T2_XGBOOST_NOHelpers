# XGBoost Model Feature Analysis

## Summary of Features in XGBoost Models

This document explains the structure and naming of features used in the XGBoost models found in the `output/S11A_XGBoost_Models/` directory.

## Feature Structure

Each XGBoost model is trained with **14 features** organized in the following way:

1. **4 features from the target factor itself**:
   - 1-month moving average (`<factor>_1m`)
   - 3-month moving average (`<factor>_3m`)
   - 12-month moving average (`<factor>_12m`)
   - 60-month moving average (`<factor>_60m`)

2. **10 features from helper factors** (the most correlated factors):
   - 60-month moving averages of the top 10 correlated factors (`helper_1_<factor>_60m` through `helper_10_<factor>_60m`)

## Example: 12-1MTR_CS Factor

For the 12-1MTR_CS factor (12-1 Month Trailing Return Cross-Section), the features are:

| Index | Feature Name | Description |
|-------|-------------|-------------|
| 0 | 12-1MTR_CS_1m | 1-month MA of 12-1 Month Trailing Return (Cross-Section) |
| 1 | 12-1MTR_CS_3m | 3-month MA of 12-1 Month Trailing Return (Cross-Section) |
| 2 | 12-1MTR_CS_12m | 12-month MA of 12-1 Month Trailing Return (Cross-Section) |
| 3 | 12-1MTR_CS_60m | 60-month MA of 12-1 Month Trailing Return (Cross-Section) |
| 4 | helper_1_12-1MTR_CS_60m | 60-month MA of the #1 helper factor (same as target) |
| 5 | helper_2_12MTR_CS_60m | 60-month MA of the #2 helper factor (12-Month Trailing Return CS) |
| 6 | helper_3_12-1MTR_TS_60m | 60-month MA of the #3 helper factor (12-1 Month Trailing Return Time-Series) |
| 7 | helper_4_12MTR_TS_60m | 60-month MA of the #4 helper factor (12-Month Trailing Return Time-Series) |
| 8 | helper_5_Current Account_CS_60m | 60-month MA of the #5 helper factor (Current Account CS) |
| 9 | helper_6_Current Account_TS_60m | 60-month MA of the #6 helper factor (Current Account TS) |
| 10 | helper_7_Earnings Yield_TS_60m | 60-month MA of the #7 helper factor (Earnings Yield TS) |
| 11 | helper_8_1MTR_CS_60m | 60-month MA of the #8 helper factor (1-Month Trailing Return CS) |
| 12 | helper_9_Trailing PE_CS_60m | 60-month MA of the #9 helper factor (Trailing PE CS) |
| 13 | helper_10_Trailing EPS_TS_60m | 60-month MA of the #10 helper factor (Trailing EPS TS) |

## Feature Importance Analysis

For the 12-1MTR_CS factor in window 1, the most important features in descending order are:

1. **helper_10_Trailing EPS_TS_60m (13.17%)**  
   The 60-month moving average of Trailing EPS (Time-Series) is the most important predictor.

2. **helper_6_Current Account_TS_60m (11.41%)**  
   The 60-month moving average of Current Account (Time-Series) is the second most important.

3. **helper_9_Trailing PE_CS_60m (9.56%)**  
   The 60-month moving average of Trailing PE (Cross-Section) ranks third in importance.

4. **12-1MTR_CS_60m (9.32%)**  
   The long-term (60-month) moving average of the target factor itself is the fourth most important.

This analysis shows that the most important predictors for this factor are a mix of earnings-related metrics (Trailing EPS, Trailing PE) and other financial indicators (Current Account). Interestingly, the 60-month moving average of the target factor is more important than its shorter-term averages.

## Understanding Feature Suffix Naming

The features have suffixes that indicate their type:

- **CS**: Cross-Sectional normalization (comparing across countries at a point in time)
- **TS**: Time-Series normalization (comparing across time for a single country)
- **_1m, _3m, _12m, _60m**: Moving average window lengths (1-month, 3-month, 12-month, 60-month)

## How to Use This Information

When analyzing model predictions and performance, consider:

1. Which helper features are most important for each factor
2. Whether the target factor's own history (MAs) or helper factors drive predictions
3. The relative importance of short-term vs. long-term trends
4. The balance of cross-sectional (CS) vs. time-series (TS) factors

For future models, you might want to focus on including the most important helper features identified here. 