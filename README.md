# Manufacturing Defect Detection Using Machine Learning
A predictive model to identify manufacturing defects using sensor data from semiconductor production.
In semiconductor manufacturing the products which are defctive when reached to customers causes significant losses. This project builds machine learning system which detects the defect early in the production process

# The challenges:
- Only 7% of products have defects (highly imbalanced dataset)
- 591 sensor readings per product (high-dimensional data)
- Need to balance catching defects vs. false alarms

# Results & Model Selection
Model                 Accuracy   Recall   Precision   ROC AUC
--------------------------------------------------------------
Logistic Regression     0.564     0.381     0.061      0.487
Random Forest           0.787     0.190     0.074      0.528
XGBoost                 0.844     0.190     0.111      0.567

# Why I Chose Logistic Regression:
Despite lower accuracy, Logistic Regression is the best model for this business problem because:
- Recall leader: 38.1%, nearly 2 times better at catching defects.
- Optimized for business losses: False negatives cost more vs false positives.
- Explains itself clearly: shows which sensor features matter most.
- Model is robust, less likely to behave unpredictably.

Key Insights : In manufacturing, catching defects (recall) matters more than overall accuracy. Initially, I was impressed by XGBoost's 84% accuracy. But then I realized - that's not what matters for defect detection
Logistic Regression only had 56% accuracy but caught twice as many defects 38%. In manufacturing, missing a defect costs more while a false alarm costs very less for inspection.
This taught me an important lesson: always optimize for the business metric that matters, not just accuracy.

# What I Learned
1. Handling Severely Imbalanced Data
   - Implemented SMOTE to balance training data.
   - Learned why accuracy alone is misleading for imbalanced datasets.

2. Feature Engineering with High-Dimensional Data
   - Worked with 591 sensor features.
   - Applied feature selection to reduce dimensionality.
   - Created statistical features (mean, std, range).

3. Real-World Model Evaluation
   - Focused on recall (catching defects) over accuracy.
   - Understood business trade-offs between false alarms and missed defects.
  
# Business Impact (Example)
-Defect rate: Out of 10,000 wafers, 200 are FAIL.
-Logistic Regression recall = 38.1% → catches 76 defects, misses 124.
-Cost of missing a defect = €500 → 124 × €500 = €62,000 loss.
-Model flags extra 400 PASS wafers as FAIL (false alarms).
-Cost per false alarm = €10 → 400 × €10 = €4,000 cost.
-Total impact: €62,000 (missed defects) + €4,000 (false alarms) = €66,000.
-Compared to XGBoost missing 162 defects → €81,000 loss, Logistic Regression saves €15,000.

#Technologies Used
- Python: Core programming language
- Pandas & NumPy: Data manipulation and analysis
- Scikit-learn: Machine learning models and preprocessing
- XGBoost: Gradient boosting for best performance
- Imbalanced-learn: SMOTE for handling class imbalance
- Matplotlib & Seaborn: Data visualization

# Next Steps for Improvement

1. Hyperparameter Optimization
   - Use GridSearchCV to find optimal XGBoost parameters.
   - Focus on maximizing recall while maintaining precision.

2. Threshold Tuning
   - Adjust decision threshold from 0.5 to increase recall.
   - Find optimal balance for business needs.

3. Advanced Techniques
   - Try ensemble methods combining multiple models.
   - Experiment with cost-sensitive learning.
  
# Dataset
- Source: SECOM Dataset from UCI Machine Learning Repository.
- Size: 1,567 observations with 591 features.
- Target: Pass (-1) or Fail (1) classification.

# Acknowledgments
- Dataset from UCI Machine Learning Repository.
- Inspired by real semiconductor manufacturing challenges.

