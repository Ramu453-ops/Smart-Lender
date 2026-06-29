# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display results
print("Model Performance")
print("----------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Import joblib
import joblib

# Save the trained model
joblib.dump(model, "trained_model.pkl")

print("Model saved successfully!")
import joblib

# Load the saved model
loaded_model = joblib.load("trained_model.pkl")

print("Model loaded successfully!")

# Make predictions using the loaded model
prediction = loaded_model.predict(X_test)

print("Predictions:")
print(prediction)