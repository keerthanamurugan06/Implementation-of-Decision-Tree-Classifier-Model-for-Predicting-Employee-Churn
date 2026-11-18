# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KEERTHANA M
RegisterNumber: 212225040185
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 1: Load Employee Dataset
data = pd.read_csv("Employee.csv")
data.columns = data.columns.str.strip()

print("Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)

# Step 2: Preprocess Categorical Data
data = pd.get_dummies(
    data,
    columns=['Departments', 'salary'],
    drop_first=True
)

# Step 3: Separate Input and Output
X = data.drop('left', axis=1)
y = data['left']

# Step 4: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Create Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Predict Employee Churn
y_pred = model.predict(X_test)

# Step 8: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Classifier Model")
print("Predicted Values:", y_pred[:10])
print("Actual Values:", y_test.values[:10])
print("Accuracy:", accuracy)

# Step 9: Display Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Stayed', 'Left']
))

# Step 10: Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Stayed', 'Left']
)

display.plot(cmap='Blues')
plt.title("Employee Churn - Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

# Step 11: Visualize Decision Tree
plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classifier - Employee Churn")
plt.savefig("sam.png", dpi=200, bbox_inches='tight')
plt.show()

print("\nDecision Tree diagram saved as sam.png")
print("Confusion Matrix saved as confusion_matrix.png")

# Display Decision Tree
plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classifier for Employee Churn")
plt.savefig("sam.png", dpi=200, bbox_inches='tight')
plt.show()
```

## Output:
<img width="645" height="428" alt="image" src="https://github.com/user-attachments/assets/0800c43d-315f-4c7e-a7d5-bfa06717f809" />
<img width="1280" height="960" alt="image" src="https://github.com/user-attachments/assets/cc324abd-7eff-46d1-adda-84273bd998c4" />

<img width="3760" height="1929" alt="image" src="https://github.com/user-attachments/assets/df84282f-eb35-4870-a1ed-f019a3a62d12" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
