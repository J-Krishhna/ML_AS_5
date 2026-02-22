# 🌳 Loan Approval Prediction using Decision Tree
print("Jaya Krishna G - 24BAD042")
# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 2. Load Dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

# 3. Preprocessing
print(df.head())
print(df.isnull().sum())

# Handle Missing Values
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].median(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].median(),
    'Credit_History': df['Credit_History'].mode()[0]
}, inplace=True)

# Select Features
features = ['ApplicantIncome','LoanAmount','Credit_History','Education','Property_Area']
X = df[features]
y = df['Loan_Status']

# Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y = le.fit_transform(y)   # Y=1, N=0

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Decision Tree (Default Depth)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# 6. Experiment with Tree Depth
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)

dt_deep = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_deep.fit(X_train, y_train)

# 7. Predict Loan Status
y_pred = dt.predict(X_test)

# 8. Evaluate Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. Feature Importance
importances = dt.feature_importances_
feature_names = X.columns

# 10. Detect Overfitting
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# 11. Compare Shallow vs Deep Trees
print("Shallow Tree Accuracy:", accuracy_score(y_test, dt_shallow.predict(X_test)))
print("Deep Tree Accuracy:", accuracy_score(y_test, dt_deep.predict(X_test)))

# 📊 Visualization

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Tree Structure Plot (Limited Depth for Visibility)
plt.figure(figsize=(15,8))
plot_tree(dt_shallow, feature_names=X.columns, class_names=["Rejected","Approved"], filled=True)
plt.title("Decision Tree Structure (Depth=3)")
plt.show()

# Feature Importance Plot
plt.figure()
plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()