from sklearn.model_selection import train_test_split
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Salary': [30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1]
})

# Features (X) and Target (y)
X = data[['Age', 'Salary']]
y = data['Purchased']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the results
print("Training Features:")
print(X_train)

print("\nTesting Features:")
print(X_test)

print("\nTraining Labels:")
print(y_train)

print("\nTesting Labels:")
print(y_test)