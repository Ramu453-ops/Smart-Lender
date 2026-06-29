import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load dataset ----------
file_path = "loan_data.csv"  # update with your actual file
df = pd.read_csv(file_path)

# ---------- 2. Separate numerical and categorical columns ----------
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# ---------- 3. Univariate analysis for numerical columns ----------
for col in numerical_cols:
    print(f"\n--- {col} ---")
    print(df[col].describe())

    plt.figure(figsize=(12, 4))

    # Histogram with KDE
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, color="#3B8BD4")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    # Boxplot to detect outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color="#E8593C")
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

# ---------- 4. Univariate analysis for categorical columns ----------
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())
    print(f"Unique categories: {df[col].nunique()}")

    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette="viridis")
    plt.title(f"Count plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------- 5. Skewness check for numerical columns ----------
print("\n--- Skewness of numerical columns ---")
print(df[numerical_cols].skew())