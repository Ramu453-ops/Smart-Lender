"""
Checking and Handling Values (Missing Data, Duplicates, Outliers)
===================================================================
A complete, runnable walkthrough of the standard data-cleaning checks that
normally happen BEFORE any multivariate analysis. Demonstrated on the Iris
dataset with artificially injected missing values & duplicate rows so every
technique below actually has something to act on.

Sections:
    1. Load data (+ inject NaNs/duplicates for demo purposes)
    2. CHECK missing values      -> counts, %, heatmap, matrix
    3. HANDLE missing values     -> drop / mean / median / mode / KNN impute
    4. CHECK & HANDLE duplicates -> detect, inspect, drop
    5. CHECK outliers            -> IQR method, Z-score method, boxplots
    6. HANDLE outliers           -> drop / cap (winsorize) / impute

Requirements: numpy, pandas, matplotlib, seaborn, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer, KNNImputer

sns.set_style("whitegrid")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. LOAD DATA (replace with pd.read_csv("your_file.csv") for real use)
#    -> Artificially inject missing values + duplicate rows so the checks
#       below have something real to find. Skip this block with your own data.
# ---------------------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Inject ~8% missing values at random into the numeric columns
numeric_cols = iris.feature_names
for col in numeric_cols:
    missing_idx = df.sample(frac=0.08, random_state=np.random.randint(0, 1000)).index
    df.loc[missing_idx, col] = np.nan

# Inject 5 duplicate rows
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)

print("=" * 70)
print("1. RAW DATA PREVIEW")
print("=" * 70)
print(df.head(), "\n")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ---------------------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ---------------------------------------------------------------------------
print("=" * 70)
print("2. CHECKING MISSING VALUES")
print("=" * 70)

missing_count = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing_count, "missing_%": missing_pct})
missing_summary = missing_summary[missing_summary["missing_count"] > 0]
print("Columns with missing values:\n", missing_summary, "\n")

print(f"Total missing cells: {df.isnull().sum().sum()}")
print(f"Rows with at least one missing value: {df.isnull().any(axis=1).sum()}\n")

# Visual 1: missingness heatmap (yellow = missing)
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Value Map (bright = missing)")
plt.tight_layout()
plt.savefig("01_missing_value_map.png", dpi=150)
plt.close()

# Visual 2: missing % bar chart
if not missing_summary.empty:
    plt.figure(figsize=(7, 4))
    missing_summary["missing_%"].plot(kind="bar", color="indianred")
    plt.ylabel("% Missing")
    plt.title("Missing Value Percentage by Column")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("02_missing_value_pct.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# 3. HANDLE MISSING VALUES
#    Several strategies shown side by side — pick the one that fits your case.
# ---------------------------------------------------------------------------
print("=" * 70)
print("3. HANDLING MISSING VALUES")
print("=" * 70)

# --- Strategy A: Drop rows with ANY missing value (simple, but loses data) ---
df_dropped = df.dropna()
print(f"[Drop rows]      shape after dropna(): {df_dropped.shape}  "
      f"(removed {df.shape[0] - df_dropped.shape[0]} rows)")

# --- Strategy B: Drop columns with too much missingness (e.g. > 50%) ---
thresh = 0.5
df_drop_cols = df.loc[:, df.isnull().mean() < thresh]
print(f"[Drop columns]   shape after dropping cols >{thresh:.0%} missing: {df_drop_cols.shape}")

# --- Strategy C: Mean / Median imputation (numeric columns) ---
df_mean_imputed = df.copy()
mean_imputer = SimpleImputer(strategy="mean")
df_mean_imputed[numeric_cols] = mean_imputer.fit_transform(df_mean_imputed[numeric_cols])

df_median_imputed = df.copy()
median_imputer = SimpleImputer(strategy="median")
df_median_imputed[numeric_cols] = median_imputer.fit_transform(df_median_imputed[numeric_cols])

print(f"[Mean impute]    remaining NaNs: {df_mean_imputed[numeric_cols].isnull().sum().sum()}")
print(f"[Median impute]  remaining NaNs: {df_median_imputed[numeric_cols].isnull().sum().sum()}")

# --- Strategy D: Mode imputation (works for categorical columns too) ---
df_mode_imputed = df.copy()
mode_imputer = SimpleImputer(strategy="most_frequent")
df_mode_imputed[numeric_cols] = mode_imputer.fit_transform(df_mode_imputed[numeric_cols])
print(f"[Mode impute]    remaining NaNs: {df_mode_imputed[numeric_cols].isnull().sum().sum()}")

# --- Strategy E: KNN imputation (uses similarity across rows — usually best for MV stats) ---
df_knn_imputed = df.copy()
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed[numeric_cols] = knn_imputer.fit_transform(df_knn_imputed[numeric_cols])
print(f"[KNN impute]     remaining NaNs: {df_knn_imputed[numeric_cols].isnull().sum().sum()}\n")

print("Guidance:")
print("  - Drop rows/cols  : only when missingness is small (<5%) and random")
print("  - Mean/median     : quick, distorts variance/correlations if missing % is high")
print("  - Mode            : best for categorical columns")
print("  - KNN / iterative : preserves multivariate relationships best — preferred")
print("                      before PCA/MANOVA/clustering on the cleaned data\n")

# From here on, continue with the KNN-imputed version (best for multivariate work)
df_clean = df_knn_imputed.copy()

# ---------------------------------------------------------------------------
# 4. CHECK & HANDLE DUPLICATES
# ---------------------------------------------------------------------------
print("=" * 70)
print("4. CHECKING & HANDLING DUPLICATES")
print("=" * 70)

dup_mask = df_clean.duplicated()
print(f"Duplicate rows found: {dup_mask.sum()}")
if dup_mask.sum() > 0:
    print("Example duplicate rows:\n", df_clean[dup_mask].head(), "\n")

df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Shape after dropping duplicates: {df_clean.shape}\n")

# ---------------------------------------------------------------------------
# 5. CHECK OUTLIERS
#    Two standard methods: IQR (robust) and Z-score (assumes ~normal data)
# ---------------------------------------------------------------------------
print("=" * 70)
print("5. CHECKING OUTLIERS")
print("=" * 70)


def iqr_outlier_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


outlier_report = {}
for col in numeric_cols:
    lower, upper = iqr_outlier_bounds(df_clean[col])
    mask = (df_clean[col] < lower) | (df_clean[col] > upper)
    outlier_report[col] = {"lower": round(lower, 2), "upper": round(upper, 2),
                            "n_outliers": int(mask.sum())}

print("IQR method (1.5*IQR rule):")
print(pd.DataFrame(outlier_report).T, "\n")

z_scores = (df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std()
z_outliers = (z_scores.abs() > 3).sum()
print("Z-score method (|z| > 3):")
print(z_outliers, "\n")

# Boxplots — visually flag outliers (points beyond the whiskers)
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_clean[numeric_cols], palette="Set2")
plt.title("Boxplots — Points Beyond Whiskers Are Outliers (IQR Rule)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("03_outlier_boxplots.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. HANDLE OUTLIERS
#    Strategy shown: cap/winsorize at the IQR bounds (keeps all rows, limits
#    extreme influence). Alternatives: drop rows, or log-transform skewed data.
# ---------------------------------------------------------------------------
print("=" * 70)
print("6. HANDLING OUTLIERS (capping at IQR bounds)")
print("=" * 70)

df_capped = df_clean.copy()
for col in numeric_cols:
    lower, upper = iqr_outlier_bounds(df_clean[col])
    df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)

n_capped = (df_clean[numeric_cols] != df_capped[numeric_cols]).sum()
print("Values capped per column:\n", n_capped, "\n")

# Before/after comparison plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=df_clean[numeric_cols], ax=axes[0], palette="Set2")
axes[0].set_title("Before Capping")
axes[0].tick_params(axis="x", rotation=20)
sns.boxplot(data=df_capped[numeric_cols], ax=axes[1], palette="Set2")
axes[1].set_title("After Capping (Winsorized)")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("04_outliers_before_after.png", dpi=150)
plt.close()

print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Original shape : {df.shape}")
print(f"Final clean shape : {df_capped.shape}")
print("Figures saved:")
print("  01_missing_value_map.png")
print("  02_missing_value_pct.png")
print("  03_outlier_boxplots.png")
print("  04_outliers_before_after.png")
print("=" * 70)