import pandas as pd

# ---------- Import and read the dataset ----------
file_path = "loan_data.csv"  # update with your actual file name/path

# Choose the loader based on file type
if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
elif file_path.endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path)
elif file_path.endswith(".json"):
    df = pd.read_json(file_path)
elif file_path.endswith(".txt"):
    df = pd.read_csv(file_path, delimiter="\t")  # adjust delimiter as needed
else:
    raise ValueError("Unsupported file format")

# ---------- Confirm the dataset loaded correctly ----------
print("Dataset loaded successfully!")
print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())