import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load dataset ----------
file_path = "loan_data.csv"  # update with your actual file
df = pd.read_csv(file_path)

# ---------- 2. Define target variable and column types ----------
target_col = "loan_status"  # update with your actual target column name

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Remove target from feature lists if present, to analyze it separately
if target_col in numerical_cols:
    numerical_cols.remove(target_col)
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

# ---------- 3. Numerical vs Numerical: Correlation heatmap ----------
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation heatmap (numerical features)")
plt.tight_layout()
plt.show()

# Pairplot for a closer look (optional, can be slow for many columns)
sns.pairplot(df[numerical_cols].sample(min(500, len(df))))
plt.suptitle("Pairwise relationships (numerical features)", y=1.02)
plt.show()

# ---------- 4. Numerical vs Categorical Target: Boxplots ----------
if target_col in df.columns:
    for col in numerical_cols:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=target_col, y=col, data=df, palette="Set2")
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.show()

# ---------- 5. Categorical vs Categorical Target: Stacked bar / count plots ----------
if target_col in df.columns:
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue=target_col, data=df, palette="Set1")
        plt.title(f"{col} vs {target_col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.show()

# ---------- 6. Categorical vs Numerical Target (if target is numerical): Groupby means ----------
if target_col in numerical_cols + [target_col] and df[target_col].dtype != "object":
    for col in categorical_cols:
        print(f"\nAverage {target_col} by {col}:")
        print(df.groupby(col)[target_col].mean().sort_values(ascending=False))