import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import xgboost as xgb

print("="*60)
print("PHASE 4: MODEL BUILDING")
print("Manufacturing Defect Prediction")
print("="*60)
print("\nAll libraries loaded successfully!")

# Load the final feature set from feature engineering phase
X = pd.read_csv(r"C:\Users\shree\Desktop\SECOM Project\data\X_final.csv")
y = pd.read_csv(r"C:\Users\shree\Desktop\SECOM Project\data\processed_data.csv")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns (first 10): {X.columns.tolist()[:10]}")
print(f"\nTarget distribution:")
pass_count = int((y == 0).sum().values[0])
fail_count = int((y == 1).sum().values[0])

print(f"  Pass (0): {pass_count} samples ({pass_count/len(y)*100:.1f}%)")
print(f"  Fail (1): {fail_count} samples ({fail_count/len(y)*100:.1f}%)")

print(f"\n Data loaded successfully!")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% for testing
    random_state=42,      # Reproducible results
    stratify=y            # Keep same Pass/Fail ratio in both sets
)

print("Train/Test Split Complete!")
print("="*60)

# Count samples
train_pass = int((y_train == 0).sum())
train_fail = int((y_train == 1).sum())
test_pass = int((y_test == 0).sum())
test_fail = int((y_test == 1).sum())

# Print stats
print(f"Training set: {X_train.shape[0]} samples")
print(f"  - Pass (0): {train_pass} ({train_pass/len(y_train)*100:.1f}%)")
print(f"  - Fail (1): {train_fail} ({train_fail/len(y_train)*100:.1f}%)")

print(f"\nTest set: {X_test.shape[0]} samples")
print(f"  - Pass (0): {test_pass} ({test_pass/len(y_test)*100:.1f}%)")
print(f"  - Fail (1): {test_fail} ({test_fail/len(y_test)*100:.1f}%)")

print("\n Note: Training set is imbalanced (~7% failures)")
print("   We'll fix this with SMOTE in next step")


from imblearn.over_sampling import SMOTE

print("Applying SMOTE to balance training data...")
print("="*60)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("BEFORE SMOTE:")
print(f"  Pass (0): {(y_train == 0).sum()} samples")
print(f"  Fail (1): {(y_train == 1).sum()} samples")
print(f"  Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAFTER SMOTE:")
print(f"  Pass (0): {(y_train_balanced == 0).sum()} samples")
print(f"  Fail (1): {(y_train_balanced == 1).sum()} samples")
print(f"  Balanced ratio: {(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.1f}:1")

print(f"\n Training set is now balanced!")
print(f"   Added {len(y_train_balanced) - len(y_train)} synthetic Fail samples")
print("\n IMPORTANT: Test set remains unchanged (original imbalance)")

print("="*60)
print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
print("="*60)

# Logistic Regression needs scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train_balanced)
print("✅ Training complete!")

# Predict on test set
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(classification_report(y_test, y_pred_lr, target_names=['Pass', 'Fail']))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
TN, FP, FN, TP = cm_lr.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN): {TN} - Correctly predicted Pass")
print(f"  False Positives (FP): {FP} - False alarms (predicted Fail, was Pass)")
print(f"  False Negatives (FN): {FN} - MISSED defects (predicted Pass, was Fail) ")
print(f"  True Positives (TP): {TP} - Correctly detected defects ")

# Key metrics
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\nKey Metrics:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f} (Of predicted failures, {precision*100:.1f}% were actually failures)")
print(f"  Recall:    {recall:.3f} (Caught {recall*100:.1f}% of all failures) ")
print(f"  F1-Score:  {f1:.3f}")
print(f"  ROC AUC:   {roc_auc:.3f}")

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pass', 'Fail'], 
            yticklabels=['Pass', 'Fail'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix: Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\shree\Desktop\SECOM Project\data\confusion_image.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n Confusion matrix saved!")

print("="*60)
print("MODEL 2: RANDOM FOREST")
print("="*60)

# Train model (tree models don't need scaling!)
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,       # 100 trees
    max_depth=10,          # Limit depth to prevent overfitting
    min_samples_split=10,
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
rf_model.fit(X_train_balanced, y_train_balanced)
print(" Training complete!")

# Predict
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(classification_report(y_test, y_pred_rf, target_names=['Pass', 'Fail']))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
TN, FP, FN, TP = cm_rf.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN): {TN}")
print(f"  False Positives (FP): {FP}")
print(f"  False Negatives (FN): {FN} ")
print(f"  True Positives (TP): {TP} ")

# Key metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\nKey Metrics:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f} ⭐")
print(f"  F1-Score:  {f1:.3f}")
print(f"  ROC AUC:   {roc_auc:.3f}")

# Visualize
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Pass', 'Fail'], 
            yticklabels=['Pass', 'Fail'])
plt.title('Confusion Matrix: Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\shree\Desktop\SECOM Project\data\confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['importance'], color='steelblue')
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest at top
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\shree\Desktop\SECOM Project\data\feature_importance_rf.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n Confusion matrix and feature importance saved!")

print("="*60)
print("MODEL 3: XGBOOST (Advanced)")
print("="*60)

# Train model
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)
print(" Training complete!")

# Predict
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(classification_report(y_test, y_pred_xgb, target_names=['Pass', 'Fail']))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
TN, FP, FN, TP = cm_xgb.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN): {TN}")
print(f"  False Positives (FP): {FP}")
print(f"  False Negatives (FN): {FN} ")
print(f"  True Positives (TP): {TP} ")

# Key metrics
accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb)
recall = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)
roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"\nKey Metrics:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f} ⭐")
print(f"  F1-Score:  {f1:.3f}")
print(f"  ROC AUC:   {roc_auc:.3f}")

# Visualize
plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Pass', 'Fail'], 
            yticklabels=['Pass', 'Fail'])
plt.title('Confusion Matrix: XGBoost', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\shree\Desktop\SECOM Project\data\confusion_matrix_xgb.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n Confusion matrix saved!")

print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Create comparison dataframe
models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print("\n Performance Comparison:")
print("="*60)
print(models_comparison.to_string(index=False))

# Find best model by recall
best_recall_idx = models_comparison['Recall'].idxmax()
best_model = models_comparison.loc[best_recall_idx, 'Model']

print(f"\n BEST MODEL (by Recall): {best_model}")
print(f"   Recall: {models_comparison.loc[best_recall_idx, 'Recall']:.3f}")
print(f"   → Catches {models_comparison.loc[best_recall_idx, 'Recall']*100:.1f}% of defects!")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Metrics comparison
models_comparison.set_index('Model')[['Precision', 'Recall', 'F1-Score', 'ROC AUC']].plot(
    kind='bar', ax=axes[0], width=0.8
)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_xticklabels(models_comparison['Model'], rotation=45, ha='right')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: ROC Curves
# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
axes[1].plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC={auc_lr:.3f})', linewidth=2)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
axes[1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', linewidth=2)

# XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
axes[1].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})', linewidth=2)

# Random baseline
axes[1].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)

axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate (Recall)', fontsize=12)
axes[1].set_title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\shree\Desktop\SECOM Project\data\model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n Comparison visualizations saved!")

print("="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

# Use best model's confusion matrix (let's assume XGBoost for now)
# YOU SHOULD USE YOUR ACTUAL BEST MODEL!
cm = confusion_matrix(y_test, y_pred_xgb)
TN, FP, FN, TP = cm.ravel()

print("\nConfusion Matrix Values:")
print(f"  TN (Correct Pass): {TN}")
print(f"  FP (False Alarm): {FP}")
print(f"  FN (Missed Defect): {FN}")
print(f"  TP (Caught Defect): {TP}")

# Calculate detection rate
total_defects = TP + FN
detection_rate = TP / total_defects if total_defects > 0 else 0
false_alarm_rate = FP / (TN + FP) if (TN + FP) > 0 else 0

print(f"\n Model Performance:")
print(f"  Detection Rate: {detection_rate*100:.1f}% ({TP} out of {total_defects} defects caught)")
print(f"  False Alarm Rate: {false_alarm_rate*100:.1f}% ({FP} false alarms)")

# Business assumptions (research industry averages)
print("\n" + "="*60)
print("COST CALCULATION")
print("="*60)

defect_cost = 500  # € per defect reaching customer
inspection_cost = 10  # € per manual inspection
production_volume_per_year = 100000  # products/year

print("\nAssumptions:")
print(f"  Cost per defect (to customer): €{defect_cost}")
print(f"  Cost per inspection: €{inspection_cost}")
print(f"  Annual production: {production_volume_per_year:,} products")

# BASELINE (No ML model)
baseline_defect_rate = 0.07  # 7% from data
baseline_defects_per_year = production_volume_per_year * baseline_defect_rate
baseline_cost = baseline_defects_per_year * defect_cost

print("\n BASELINE (No ML Model):")
print(f"  Defects per year: {baseline_defects_per_year:,.0f}")
print(f"  All defects reach customers")
print(f"  Total annual cost: €{baseline_cost:,.0f}")

# WITH ML MODEL
detected_defects = baseline_defects_per_year * detection_rate
missed_defects = baseline_defects_per_year * (1 - detection_rate)
false_alarms = production_volume_per_year * (1 - baseline_defect_rate) * false_alarm_rate

cost_missed_defects = missed_defects * defect_cost
cost_inspections = (detected_defects + false_alarms) * inspection_cost
total_cost_with_ml = cost_missed_defects + cost_inspections

print("\n WITH ML MODEL:")
print(f"  Defects detected: {detected_defects:,.0f} (inspected and fixed)")
print(f"  Defects missed: {missed_defects:,.0f} (reach customers)")
print(f"  False alarms: {false_alarms:,.0f} (unnecessary inspections)")
print(f"\n  Cost of missed defects: €{cost_missed_defects:,.0f}")
print(f"  Cost of inspections: €{cost_inspections:,.0f}")
print(f"  Total annual cost: €{total_cost_with_ml:,.0f}")

# SAVINGS
savings = baseline_cost - total_cost_with_ml
savings_percentage = (savings / baseline_cost) * 100

print("\n" + "="*60)
print(" ANNUAL SAVINGS")
print("="*60)
print(f"  Savings: €{savings:,.0f}")
print(f"  Cost reduction: {savings_percentage:.1f}%")
print(f"  ROI: {(savings/17000)*100:.0f}% (assuming €17k implementation cost)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cost comparison
costs_df = pd.DataFrame({
    'Scenario': ['Baseline\n(No ML)', 'With ML\nModel'],
    'Cost': [baseline_cost, total_cost_with_ml]
})

bars = axes[0].bar(costs_df['Scenario'], costs_df['Cost'], 
                   color=['#FF6B6B', '#4ECDC4'], width=0.6)
axes[0].set_title('Annual Cost Compar,.ison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Annual Cost (€)', fontsize=12)
axes[0].set_xlabel('')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'€{height:,.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add savings annotation
axes[0].annotate(f'Savings:\n€{savings:,.0f}\n({savings_percentage:.1f}%)',
                xy=(0.5, max(costs_df['Cost'])*0.6),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: Defect breakdown
defect_data = pd.DataFrame({
    'Category': ['Detected\n(Fixed)', 'Missed\n(Cost)', 'False\nAlarms'],
    'Count': [detected_defects, missed_defects, false_alarms],
    'Color': ['#2ECC71', '#E74C3C', '#F39C12']
})

bars2 = axes[1].bar(defect_data['Category'], defect_data['Count'], 
                    color=defect_data['Color'], width=0.6)
axes[1].set_title('ML Model Impact Breakdown', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Products', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, defect_data['Count']):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(r'C:\Users\shree\Desktop\SECOM Project\data\business_impact.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n Business impact visualization saved!")


# Modeling Results Summary

## Models Tested
#1. Logistic Regression (Baseline)
#2. Random Forest (Ensemble)
#3. XGBoost (Advanced Gradient Boosting)

## Performance Comparison

#| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
#|-------|----------|-----------|--------|----------|---------|
#| Logistic Regression | [fill] | [fill] | [fill] | [fill] | [fill] |
#| Random Forest | [fill] | [fill] | [fill] | [fill] | [fill] |
#| XGBoost | [fill] | [fill] | [fill] | [fill] | [fill] |

## Best Model: [Model Name]

#**Selected based on RECALL** (most important metric for defect detection)

#- **Recall: X%** - Catches X% of all defects
#- **Precision: Y%** - Y% of flagged products are actually defective
#- **ROC AUC: Z** - Excellent discrimination ability

## Business Impact

### Annual Savings: €[X] (Y% cost reduction)

#**Baseline (No ML):**
#- 7,000 defects/year reach customers
#- Cost: €350,000/year

#**With ML Model:**
#- [X] defects detected and fixed
#- [Y] defects missed (reach customers)
#- [Z] false alarms (acceptable)
#- Cost: €[W]/year

#**ROI:** [Calculate]% return on €17,000 implementation

## Key Insights

#1. **SMOTE was critical** - Balanced training data improved recall significantly
#2. **Feature selection worked** - 50 features sufficient for prediction
#3. **[Best model] performed best** - Achieved highest recall with acceptable precision
#4. **Original sensors > engineered features** - Domain-specific patterns matter

## Recommendations

### For Deployment:
#1. Deploy [best model] at quality control checkpoint
#2. Flag products with >50% defect probability for manual inspection
#3. Expected to reduce defect costs by [Y]%

### Critical Sensors (from feature importance):
#1. Sensor [X] - Most important
#2. Sensor [Y] - Second most important
#3. Sensor [Z] - Third most important

#**Recommendation:** Prioritize calibration and maintenance of these sensors

### Next Steps:
#1. Pilot deployment on one production line (1-3 months)
#2. Collect real-world feedback and adjust threshold
#3. Roll out to all lines (3-6 months)
#4. Quarterly model retraining with new data

## Conclusion

#Machine learning successfully improves defect detection from baseline to [X]% recall,
#resulting in €[Y] annual savings. Model is ready for pilot deployment.