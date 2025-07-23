import pandas as pd
import joblib

# Load models and scalers
svm = joblib.load("models/svm_classifier.joblib")
lr_reg = joblib.load("models/linear_regression.joblib")
scaler_cls = joblib.load("models/scaler_cls.joblib")
scaler_reg = joblib.load("models/scaler_reg.joblib")

# The exact order and feature names you used for training
important_features = [
    'G1', 'G2', 'failures', 'age',
    'romantic_yes', 'goout',
    'traveltime', 'total_alcohol', 'parent_edu', 'high_absence'
]

def get_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = int(input(prompt))
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                print(f"Please enter a value between {min_val} and {max_val}.")
            else:
                return value
        except ValueError:
            print("Please enter an integer.")

def get_binary(prompt):
    while True:
        value = input(prompt + " (yes=1, no=0): ").strip().lower()
        if value in ['1', 'yes', 'y']:
            return 1
        elif value in ['0', 'no', 'n']:
            return 0
        else:
            print("Please enter 1 for yes or 0 for no.")

print("Enter new student information as prompted:")

G1 = get_int("G1 (First period grade, 0-20): ", 0, 20)
G2 = get_int("G2 (Second period grade, 0-20): ", 0, 20)
failures = get_int("Number of past class failures (0-3): ", 0, 3)
age = get_int("Age: ", 15, 22)
romantic_yes = get_binary("In a romantic relationship?")
goout = get_int("Going out with friends (1=very low to 5=very high): ", 1, 5)
traveltime = get_int("Home to school travel time (1=<15min, 2=15-30min, 3=30-60min, 4=>1h): ", 1, 4)
total_alcohol = get_int("Alcohol consumption (1=very low to 5=very high): ", 1, 5)
parent_edu = get_int("Parental education level (0=none to 8=highest): ", 0, 8)
high_absence = get_binary("High absences?")

new_student = [G1, G2, failures, age, romantic_yes, goout, traveltime, total_alcohol, parent_edu, high_absence]


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
print(f"Probability of Passing: {pass_prob * 100:.2f} %")
print(f"Predicted Final Grade: {grade_pred:.1f}")

# Grade range interpretation
if grade_pred >= 18:
    grade_range = "Excellent (18-20)"
elif grade_pred >= 14:
    grade_range = "Good (14-17)"
elif grade_pred > 10:
    grade_range = "Pass (11-13)"
else:
    grade_range = "Fail (<=10)"

print(f"Grade Range: {grade_range}")
