from sklearn.preprocessing import StandardScaler
import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45],
    'Salary': [30000, 50000, 70000, 90000, 110000]
})

# Initialize scaler
scaler = StandardScaler()

# Scale the data
scaled_data = scaler.fit_transform(data)

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

print(scaled_df)