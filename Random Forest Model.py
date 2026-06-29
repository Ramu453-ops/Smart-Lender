# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Salary': [30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1]
})

# Features (X) and Target (y)
X = data[['Age', 'Salary']]
y = data['Purchased']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    criterion='gini',      # or 'entropy'
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict for a new customer
new_data = [[38, 80000]]  # Age=38, Salary=80000
prediction = model.predict(new_data)

print("\nPrediction for New Customer:")
if prediction[0] == 1:
    print("Purchased")
else:
    print("Not Purchased")