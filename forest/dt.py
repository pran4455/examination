import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

correlation = df.corr()
# print(correlation)
sns.heatmap(correlation, annot = True, linewidths = 0.5)
plt.show()
df
# Drop non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)  # Remove text-based columns

# Handle missing values (only for numeric columns)
df.fillna(df.mean(numeric_only=True), inplace=True)
y = (df['num'] > 0).astype(int).values
y
sns.countplot(x=y, palette="coolwarm")
plt.title("Distribution of Heart Disease (1: Disease, 0: No Disease)")
plt.show()

# Define Features (X) and Target (y)
X = df.drop("num", axis=1)  # Remove target column
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
dt_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)

dt_model.fit(X_train, y_train)

plt.figure(figsize=(12, 6))
plot_tree(dt_model, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.show()

y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred))

# New patient input with correct number of features (age, trestbps, chol, thalch, oldpeak, ca)
new_patient = np.array([921, 55, 1, 140, 230, 0, 1, 150, 0, 2.5, 0])   # Adjust values as needed
new_patient = new_patient.reshape(1, -1)  # Reshape for model input

# Predict for new patient
prediction = dt_model.predict(new_patient)

# Print result
if prediction[0] == 1:
    print("Prediction: Patient is at risk of Heart Disease.")
else:
    print("Prediction: No Heart Disease detected.")
