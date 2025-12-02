import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setting visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', 100)

# Loading the  data
df = pd.read_csv(r"C:\Users\shree\Desktop\SECOM Project\uci-secom.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()[:10]}...")  # Show first 10
print(f"\nFirst few rows:")
print(df.head())

# Separating features and target
X = df.drop(['Time', 'Pass/Fail'], axis=1)
y = df['Pass/Fail']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns (sensor readings): {X.columns.tolist()[:20]}...")

# Check target distribution
print("\nTarget Distribution:")
print(y.value_counts())
print("\nPercentages:")
print(y.value_counts(normalize=True) * 100)

# Visualize
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar')
plt.title('Class Distribution: Pass vs Fail')
plt.xlabel('Class (-1=Pass, 1=Fail)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\shree\Desktop\SECOM Project\class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()  

# Calculate imbalance
fail_rate = (y == 1).sum() / len(y) * 100
print(f"\nFailure rate: {fail_rate:.2f}%")
print(f"This is an imbalanced dataset - we'll handle this later with SMOTE")


# Check missing values
missing = X.isnull().sum()
missing_pct = (missing / len(X)) * 100

missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_percentage': missing_pct
}).sort_values('missing_percentage', ascending=False)

print("Features with most missing values:")
print(missing_df.head(20))

# Count how many features have >50% missing
high_missing = (missing_pct > 50).sum()
print(f"\nFeatures with >50% missing: {high_missing}")

# Visualize missing values (top 30)
top_missing = missing_df[missing_df['missing_count'] > 0].head(30)

plt.figure(figsize=(12, 6))
top_missing['missing_percentage'].plot(kind='bar')
plt.title('Top 30 Features with Missing Values')
plt.xlabel('Feature')
plt.ylabel('Percentage Missing')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"C:\Users\shree\Desktop\SECOM Project\missing_values.png", dpi=300, bbox_inches='tight')
plt.close()  

# Basic statistics for first 10 features
print("\nBasic statistics (first 10 features):")
print(X.iloc[:, :10].describe())

# Decision: Drop columns with >50% missing (too little information)
threshold = 0.5
cols_to_drop_missing = missing_df[missing_df['missing_percentage'] > threshold*100].index.tolist()

print(f"Columns to drop (>{threshold*100}% missing): {len(cols_to_drop_missing)}")
print(f"Examples: {cols_to_drop_missing[:10]}")

# Also drop columns with zero variance (all same value = useless)
variances = X.var()
zero_var_cols = variances[variances == 0].index.tolist()

print(f"Columns with zero variance: {len(zero_var_cols)}")
if len(zero_var_cols) > 0:
    print(f"Examples: {zero_var_cols[:10]}")


# Combine all columns to drop
all_cols_to_drop = list(set(cols_to_drop_missing + zero_var_cols))

print(f"Original features: {X.shape[1]}")
print(f"Dropping {len(all_cols_to_drop)} features")

# Drop them
X_cleaned = X.drop(columns=all_cols_to_drop)

print(f"Remaining features: {X_cleaned.shape[1]}")

# Fill remaining missing values with median (robust to outliers)
print(f"Missing values before filling: {X_cleaned.isnull().sum().sum()}")

X_cleaned = X_cleaned.fillna(X_cleaned.median())

print(f"Missing values after filling: {X_cleaned.isnull().sum().sum()}")

# Verify
print(f"\nFinal cleaned dataset shape: {X_cleaned.shape}")
print(f"Target shape (unchanged): {y.shape}")

# Create dataset with target for comparison
X_with_target = X_cleaned.copy()
X_with_target['target'] = y.values

# Separate Pass (-1) and Fail (1) samples
pass_samples = X_with_target[X_with_target['target'] == -1].drop('target', axis=1)
fail_samples = X_with_target[X_with_target['target'] == 1].drop('target', axis=1)

print(f"Pass samples: {len(pass_samples)}")
print(f"Fail samples: {len(fail_samples)}")

# Calculate absolute mean difference between Pass and Fail
mean_pass = pass_samples.mean()
mean_fail = fail_samples.mean()
mean_diff = (mean_fail - mean_pass).abs().sort_values(ascending=False)

print("Top 20 features with largest mean difference:")
print(mean_diff.head(20))

# Save for later use
top_features = mean_diff.head(20).index.tolist()
print(f"\nTop features saved: {top_features[:10]}...")

# Plot top 10 features with biggest differences
plt.figure(figsize=(12, 6))
mean_diff.head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Features: Largest Difference Between Pass and Fail', fontsize=14, fontweight='bold')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Absolute Mean Difference', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\shree\Desktop\SECOM Project\feature_mean_differences.png", dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved!")

from scipy import stats
print("Statistical Significance Testing (t-test):")

significant_features = []

for feature in top_features:
    pass_values = pass_samples[feature].dropna()
    fail_values = fail_samples[feature].dropna()
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(pass_values, fail_values)
    
    # Check if significant (p < 0.05)
    is_significant = p_value < 0.05
    
    if is_significant:
        significant_features.append(feature)
    
    significance_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
    
    print(f"Feature {feature}: t-stat={t_stat:8.3f}, p-value={p_value:.6f} {significance_marker}")

print(f"\n{len(significant_features)} out of {len(top_features)} features are statistically significant (p < 0.05)")


# Visualize the top 3 most different features
top_3_features = mean_diff.head(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(top_3_features):
    # Prepare data for box plot
    pass_data = pass_samples[feature].dropna()
    fail_data = fail_samples[feature].dropna()
    
    # Create box plot
    bp = axes[idx].boxplot([pass_data, fail_data], 
                            labels=['Pass', 'Fail'],
                            patch_artist=True,
                            boxprops=dict(facecolor='lightblue'),
                            medianprops=dict(color='red', linewidth=2))
    
    axes[idx].set_title(f'Feature {feature}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Sensor Value', fontsize=10)
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Top 3 Predictive Features: Pass vs Fail Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r"C:\Users\shree\Desktop\SECOM Project\top_features_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Box plots saved!")

# Calculate correlation matrix for top 30 features
top_30_features = mean_diff.head(30).index.tolist()
correlation_matrix = X_cleaned[top_30_features].corr()

# Visualize
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            center=0, 
            square=True, 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8},
            fmt='.2f')
plt.title('Correlation Matrix: Top 30 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r"C:\Users\shree\Desktop\SECOM Project\correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print("Correlation matrix saved!")


# Find pairs with correlation > 0.8 (highly correlated)
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.8:
            high_corr_pairs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': corr_value
            })

if len(high_corr_pairs) > 0:
    print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
    for pair in high_corr_pairs[:10]:  # Show first 10
        print(f"  {pair['feature1']}  {pair['feature2']}: {pair['correlation']:.3f}")
else:
    print("No highly correlated pairs found (|r| > 0.8)")
    print("This is GOOD - features are relatively independent!")


#EDA SUMMARY

with open("EDA_summary.md", "w") as f:
    f.write("""# EDA Summary

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
""")
# Save cleaned data
X_cleaned.to_csv(r"C:\Users\shree\Desktop\SECOM Project\x_cleaned.csv", index=False)
y.to_csv(r"C:\Users\shree\Desktop\SECOM Project\y.csv", index=False)
print("Cleaned data saved!")
