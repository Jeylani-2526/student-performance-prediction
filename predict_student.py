import numpy as np
import pandas as pd
import joblib

# Load models and scalers
svm = joblib.load("models\svm_classifier.joblib")
lr_reg = joblib.load("\models\linear_regression.joblib")
scaler_cls = joblib.load("models\scaler_cls.joblib")
scaler_reg = joblib.load("models\scaler_reg.joblib")

# The exact order and feature names you used for training
important_features = [
    'G1', 'G2', 'failures', 'age',
    'romantic_yes', 'goout',
    'traveltime', 'total_alcohol', 'parent_edu', 'high_absence'
]

# Example: new student data (edit these values for your case)
# Order:       G1, G2, failures, age, romantic_yes, goout, traveltime, total_alcohol, parent_edu, high_absence
new_student = [12, 13, 0, 17, 0, 3, 1, 2, 8, 0]  

# Convert input to a DataFrame with correct feature names
X_new_df = pd.DataFrame([new_student], columns=important_features)

# Scale features using the original scalers
X_new_cls_scaled = scaler_cls.transform(X_new_df)
X_new_reg_scaled = scaler_reg.transform(X_new_df)

# Classification prediction (pass/fail) with SVM
pass_prob = svm.predict_proba(X_new_cls_scaled)[0, 1]
pass_pred = svm.predict(X_new_cls_scaled)[0]

# Regression prediction (final grade) with Linear Regression
grade_pred = lr_reg.predict(X_new_reg_scaled)[0]

# Print results
print("Predicted Pass/Fail: ", "PASS" if pass_pred == 1 else "FAIL")
print(f"Probability of Passing: {pass_prob:.2f}")
print(f"Predicted Final Grade: {grade_pred:.1f}")

# Grade range interpretation
if grade_pred >= 18:
    grade_range = "Excellent (18-20)"
elif grade_pred >= 14:
    grade_range = "Good (14-17)"
elif grade_pred >= 10:
    grade_range = "Pass (10-13)"
else:
    grade_range = "Fail (<10)"

print(f"Grade Range: {grade_range}")
