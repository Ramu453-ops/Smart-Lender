# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset (you can replace with real loan dataset)
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Income': [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000],
    'Loan_Amount': [50000, 80000, 100000, 120000, 150000, 180000, 200000, 220000],
    'Approved': [0, 0, 0, 1, 1, 1, 1, 1]
})

# Features and Target
X = data[['Age', 'Income', 'Loan_Amount']]
y = data['Approved']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "smart_lender_model.pkl")
print("Model saved successfully!")