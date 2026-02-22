# 🧪 Breast Cancer Diagnosis using KNN
print("Jaya Krishna G - 24BAD042")
# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# 2. Load Dataset
df = pd.read_csv("breast-cancer.csv")

# 3. Data Inspection & Preprocessing
print(df.head())
print(df.info())
print(df.isnull().sum())

features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean']
X = df[features]
y = df['diagnosis']

# 4. Encode Target Labels (M=1, B=0)
le = LabelEncoder()
y = le.fit_transform(y)

# 5. Feature Scaling (Important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 8. Experiment with Different K Values
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_k))

best_k = k_values[np.argmax(accuracies)]
print("Best K:", best_k)

# 9. Predict Diagnosis Labels
y_pred = knn.predict(X_test)

# 10. Evaluate Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 11. Identify Misclassified Cases
misclassified = np.where(y_test != y_pred)
print("Number of Misclassified Samples:", len(misclassified[0]))

# 12. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Accuracy vs K Plot
plt.plot(k_values, accuracies)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

# Decision Boundary (Using Two Features: radius_mean & texture_mean)
X2 = df[['radius_mean','texture_mean']]
X2_scaled = scaler.fit_transform(X2)
y2 = le.fit_transform(df['diagnosis'])

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2_scaled, y2, test_size=0.2, random_state=42
)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train2, y_train2)

x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X2_scaled[:,0], X2_scaled[:,1], c=y2)
plt.xlabel("Radius")
plt.ylabel("Texture")
plt.title("Decision Boundary (K=5)")
plt.show()