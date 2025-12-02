# EDA Summary

# Dataset Overview
- Total samples: 1,567
- Original features: 590 sensor measurements
- After cleaning: [446] features
- Failure rate: [6.64%] (imbalanced dataset)

# Data Cleaning
- Dropped [28] features with >50% missing values
- Dropped [116] features with zero variance
- Filled remaining missing values with median
- **Final feature count: [446]

# Key Findings

# Most Predictive Features
Based on statistical analysis (t-test), these features show significant differences between Pass and Fail:
# Top 3 Features by Mean Difference:

1.Feature 161: Largest difference, but NOT statistically significant (p=0.42) - High variance/unreliable
2.Feature 159: Second largest difference, statistically significant (p=0.002) - Reliable predictor  
3.Feature 21: Third largest difference, highly significant (p<0.001) - Highly reliable predictor

Conclusion: Features 159 and 21 are reliable predictors. Feature 161 excluded due to statistical insignificance.

# Box Plot Analysis: Mean Difference vs Statistical Significance

# Key Observations

Feature 161 (Largest Mean Difference, NOT Significant):
- Shows the largest difference between Pass and Fail medians
- However, both groups show **high variance** (wide boxes with significant overlap)
- T-test result: p-value = 0.42 (NOT significant)
- Interpretation:The large difference is likely due to sensor noise/inconsistency rather than a true predictive pattern
- Conclusion: Despite ranking #1 by mean difference, this feature is **unreliable** for prediction

Feature 21 (3rd Largest Difference, VERY Significant):
- Shows moderate difference between Pass and Fail medians
- Both groups show low variance (narrow boxes with clear separation)
- T-test result: p-value = 0.000017 (highly significant)
- Interpretation:The difference is consistent and repeatable
- Conclusion: This feature is a reliable predictor of defects

Feature 159 (2nd Largest Difference, Significant):
- Similar pattern to Feature 21: clear separation with manageable variance
- T-test result: p-value = 0.002 (significant)
- Conclusion: Another reliable predictor

# Important Insight

This analysis demonstrates why statistical significance is crucial alongside mean difference:
- Mean difference alone can be misleading (Feature 161 ranks #1 but is unreliable)
- Tight distributions (low variance) are more valuable than large differences with high variance
- For predictive modeling, we prioritize consistency over magnitude

Recommendation: Prioritize features with **both** substantial differences AND statistical significance (p < 0.05) for model building.

[List your actual top 5]

# Statistical Significance
- [9] out of top 20 features are statistically significant (p < 0.05)
- This indicates these sensors are truly predictive of defects
- Strong candidates for predictive model

# Multicollinearity
- [Found / Did not find] highly correlated feature pairs
- [If found: Consider removing redundant features in modeling phase]

# Next Steps
1. Feature engineering (create interaction features)
2. Build baseline classification model
3. Try multiple algorithms (Logistic Regression, Random Forest, XGBoost)
4. Evaluate using recall (prioritize catching defects!)
