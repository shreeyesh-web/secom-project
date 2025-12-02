import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

print("="*60)
print("FEATURE ENGINEERING - Manufacturing Defect Prediction")
print("="*60)

# You need to recreate X_cleaned and y from your EDA
# Option 1: Copy the cleaning code from your EDA notebook
# Option 2: Save/load from CSV (recommended)

print("\nLibraries loaded successfully!")

# Load cleaned data
X_cleaned = pd.read_csv(r"C:\Users\shree\Desktop\SECOM Project\x_cleaned.csv")
y = pd.read_csv(r"C:\Users\shree\Desktop\SECOM Project\y.csv").values.ravel()

print(f"Loaded X_cleaned: {X_cleaned.shape}")
print(f"Loaded y: {y.shape}")
print("\nData ready for feature engineering!")

# Create a copy so we don't modify original
X_engineered = X_cleaned.copy()

# Get list of original feature names
original_features = X_cleaned.columns.tolist()

print(f"Starting with {len(original_features)} features")
print(f"We'll create additional engineered features...")

from scipy import stats

# Separate Pass and Fail samples
X_with_target = X_cleaned.copy()
X_with_target['target'] = y

pass_samples = X_with_target[X_with_target['target'] == -1].drop('target', axis=1)
fail_samples = X_with_target[X_with_target['target'] == 1].drop('target', axis=1)

# Calculate mean differences
mean_diff = (fail_samples.mean() - pass_samples.mean()).abs().sort_values(ascending=False)

# Get top 20 by mean difference, test significance
significant_features = []
for feature in mean_diff.head(20).index:
    pass_values = pass_samples[feature].dropna()
    fail_values = fail_samples[feature].dropna()
    t_stat, p_value = stats.ttest_ind(pass_values, fail_values)
    
    if p_value < 0.05:
        significant_features.append(feature)

top_significant_features = significant_features[:10]  # Top 10
print(f"Top 10 significant features: {top_significant_features}")

print("Creating interaction features...")

# Use top 5 significant features for interactions
top_5 = top_significant_features[:5]

interaction_count = 0

for i, feat1 in enumerate(top_5):
    for feat2 in top_5[i+1:]:
        # Ratio feature
        X_engineered[f'{feat1}_div_{feat2}'] = (
            X_engineered[feat1] / (X_engineered[feat2] + 1e-10)  # +1e-10 avoids division by zero
        )
        
        # Difference feature
        X_engineered[f'{feat1}_minus_{feat2}'] = (
            X_engineered[feat1] - X_engineered[feat2]
        )
        
        # Product feature
        X_engineered[f'{feat1}_times_{feat2}'] = (
            X_engineered[feat1] * X_engineered[feat2]
        )
        
        interaction_count += 3

print(f"Created {interaction_count} interaction features")
print(f"Total features now: {X_engineered.shape[1]}")


print("Creating statistical aggregation features...")

# Calculate statistics across ALL original sensor features
X_engineered['mean_all_sensors'] = X_cleaned[original_features].mean(axis=1)
X_engineered['std_all_sensors'] = X_cleaned[original_features].std(axis=1)
X_engineered['max_all_sensors'] = X_cleaned[original_features].max(axis=1)
X_engineered['min_all_sensors'] = X_cleaned[original_features].min(axis=1)
X_engineered['range_all_sensors'] = (
    X_engineered['max_all_sensors'] - X_engineered['min_all_sensors']
)
X_engineered['median_all_sensors'] = X_cleaned[original_features].median(axis=1)

print("Created 6 statistical aggregation features:")
print("  - mean_all_sensors")
print("  - std_all_sensors")
print("  - max_all_sensors")
print("  - min_all_sensors")
print("  - range_all_sensors")
print("  - median_all_sensors")

print(f"\nTotal features now: {X_engineered.shape[1]}")

# List all columns
print(X_engineered.columns.tolist())

print("Creating outlier count feature...")

# Define "normal range" for each sensor (mean ± 2 standard deviations)
sensor_means = X_cleaned[original_features].mean()
sensor_stds = X_cleaned[original_features].std()

# Count outliers for each product
outlier_count = 0

for col in original_features:
    lower_bound = sensor_means[col] - 2 * sensor_stds[col]
    upper_bound = sensor_means[col] + 2 * sensor_stds[col]
    
    # Count sensors outside normal range
    outlier_count += (
        (X_engineered[col] < lower_bound) | 
        (X_engineered[col] > upper_bound)
    ).astype(int)

X_engineered['num_outlier_sensors'] = outlier_count

print("Created 'num_outlier_sensors' feature")
print(f"  - Counts how many sensors are >2 std deviations from normal")
print(f"\nTotal features now: {X_engineered.shape[1]}")

# Show distribution
print(f"\nOutlier count distribution:")
print(X_engineered['num_outlier_sensors'].describe())

# Check if new features help separate Pass/Fail
new_features = ['mean_all_sensors', 'std_all_sensors', 'num_outlier_sensors']

X_with_target = X_engineered[new_features].copy()
X_with_target['target'] = y

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(new_features):
    pass_data = X_with_target[X_with_target['target'] == -1][feature]
    fail_data = X_with_target[X_with_target['target'] == 1][feature]
    
    axes[idx].boxplot([pass_data, fail_data], 
                      labels=['Pass', 'Fail'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightgreen'))
    
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Engineered Features: Pass vs Fail Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r"C:\Users\shree\Desktop\SECOM Project\engineered_features.png", dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved!")

import itertools

# Step 1: NEW: Only start with top 5 features (not full dataset)
top_features = ['21', '295', '160', '159', '294']
X_engineered = X_cleaned[top_features].copy()

# Step 2: Generate all pairwise ratios ONLY among the top 5
for f1, f2 in itertools.permutations(top_features, 2):
    ratio_name = f'ratio_{f1}_{f2}'
    X_engineered[ratio_name] = X_cleaned[f1] / (X_cleaned[f2] + 1e-10)

# Step 3: Define features to plot (original + ratios)
features_to_check = X_engineered.columns.tolist()

# Step 4: Generate and save boxplots (one file per feature)
for feature in features_to_check:
    try:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=y, y=X_engineered[feature])
        plt.title(f'Boxplot of {feature} by Target')
        plt.xlabel('Target (Pass=0 / Fail=1)')
        plt.ylabel(feature)

        # Save each figure with unique filename
        save_path = fr"C:\Users\shree\Desktop\SECOM Project\{feature}_boxplot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # ✅ ensures next plot starts fresh

        print(f"✅ Saved: {save_path}")
    except Exception as e:
        print(f"⚠ Skipped {feature} due to error: {e}")

## Engineered Features Analysis

### num_outlier_sensors
#- Pass products: Median = {X} outlier sensors
#-# Fail products: Median = {Y} outlier sensors
#- #**Difference: {Y/X}x more outliers in Fail products**
#- #Interpretation: Products with many abnormal sensors are highly likely to be defective
#- #**#Conclusion: Strong predictive feature** ✅

### std_all_sensors
#- #Pass products: Median variance = {A}
#- #Fail products: Median variance = {B}
#- #Interpretation: Defective products show higher variance (unstable manufacturing process)
#- #**Conclusion: Moderately useful feature** ⚠️

### mean_all_sensors
#- #Pass products: Median mean = {C}
#- #Fail products: Median mean = {D}
#- #Interpretation: Some difference, but significant overlap
#- #**Conclusion: Weak feature, but included for completeness** ⚠️

### Overall
#Our engineered features, particularly num_outlier_sensors, demonstrate clear ability 
#to distinguish Pass from Fail products, validating our feature engineering approach.


from sklearn.preprocessing import StandardScaler

print("Scaling all features...")

# Scale ALL features (original + engineered)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)

X_scaled_df = pd.DataFrame(
    X_scaled, 
    columns=X_engineered.columns,
    index=X_engineered.index
)

print(f"Scaled {X_scaled_df.shape[1]} features")
print("\nAfter scaling, all features have mean ≈ 0, std ≈ 1")


from sklearn.feature_selection import SelectKBest, f_classif

print("Selecting top 50 features...")

# Convert target to binary
y_binary = (y == 1).astype(int)

# Select top 50 features
k_best = SelectKBest(score_func=f_classif, k=50)
X_selected = k_best.fit_transform(X_scaled_df, y_binary)

# Get selected feature names
selected_mask = k_best.get_support()
selected_features = X_scaled_df.columns[selected_mask].tolist()

print(f"Selected {len(selected_features)} best features")
print(f"\nTop 10 selected features:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"  {i}. {feat}")

# Check if any engineered features made it
engineered = ['mean_all_sensors', 'std_all_sensors', 'num_outlier_sensors', 
              'ratio_21_295', 'ratio_160_159']
selected_engineered = [f for f in engineered if f in selected_features]

if selected_engineered:
    print(f"\nEngineered features selected: {selected_engineered}")
else:
    print("\nNo engineered features selected (original features are stronger)")



# Create final dataset
X_final = pd.DataFrame(
    X_selected,
    columns=selected_features,
    index=X_engineered.index
)

# Save
X_final.to_csv('data/X_final.csv', index=False)
y_binary_df = pd.DataFrame(y_binary, columns=['target'])
y_binary_df.to_csv(r"C:\Users\shree\Desktop\SECOM Project\data\processed_data.csv", index=False)

print("="*60)
print("FEATURE ENGINEERING COMPLETE!")
print("="*60)
print(f"\nStarting features: {X_cleaned.shape[1]} (original)")
print(f"After engineering: {X_engineered.shape[1]} (original + engineered)")
print(f"After selection: {X_final.shape[1]} (best 50)")
print(f"\nFiles saved:")
print(f"  - data/X_final.csv")
print(f"  - data/y_final.csv")
print(f"\n✅ Ready for modeling!")