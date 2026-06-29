"""
Balancing the Dataset (Handling Class Imbalance)
===================================================================
A complete, runnable walkthrough of the standard techniques for dealing
with imbalanced classification data. `imbalanced-learn` isn't assumed to
be installed, so Random Under/Oversampling and SMOTE are implemented
manually with numpy/sklearn — drop in `from imblearn.over_sampling import
SMOTE` instead if you have the package available.

Sections:
    1. Create an imbalanced dataset (demo) + check the imbalance
    2. Random Undersampling      -> remove majority-class rows
    3. Random Oversampling       -> duplicate minority-class rows
    4. SMOTE                     -> synthesize new minority-class rows
    5. Class weights             -> keep all data, reweight the loss function
    6. Before/after comparison + effect on a classifier

Requirements: numpy, pandas, matplotlib, seaborn, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style("whitegrid")
RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. CREATE AN IMBALANCED DATASET (replace with your own X, y)
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=1000, n_features=6, n_informative=4, n_redundant=0,
    n_clusters_per_class=1, weights=[0.92, 0.08],   # 92% class 0, 8% class 1
    flip_y=0.01, random_state=RANDOM_STATE,
)
feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_cols)
df["target"] = y

print("=" * 70)
print("1. CHECKING CLASS IMBALANCE")
print("=" * 70)
counts = df["target"].value_counts().sort_index()
pct = (df["target"].value_counts(normalize=True).sort_index() * 100).round(2)
print("Class counts:\n", counts)
print("\nClass percentages:\n", pct)
print(f"\nImbalance ratio (majority:minority) = {counts.max() / counts.min():.1f} : 1\n")

plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df, palette="Set2")
plt.title("Original Class Distribution")
plt.tight_layout()
plt.savefig("01_original_distribution.png", dpi=150)
plt.close()

majority_class = counts.idxmax()
minority_class = counts.idxmin()
df_majority = df[df["target"] == majority_class]
df_minority = df[df["target"] == minority_class]

# ---------------------------------------------------------------------------
# 2. RANDOM UNDERSAMPLING — randomly drop majority-class rows to match minority
#    Pros: fast, no synthetic data.  Cons: throws away majority-class info.
# ---------------------------------------------------------------------------
print("=" * 70)
print("2. RANDOM UNDERSAMPLING")
print("=" * 70)

df_majority_under = df_majority.sample(n=len(df_minority), random_state=RANDOM_STATE)
df_undersampled = pd.concat([df_majority_under, df_minority]).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("New class counts:\n", df_undersampled["target"].value_counts().sort_index(), "\n")

# ---------------------------------------------------------------------------
# 3. RANDOM OVERSAMPLING — duplicate minority-class rows (with replacement)
#    Pros: keeps all majority info.  Cons: exact duplicates can cause overfitting.
# ---------------------------------------------------------------------------
print("=" * 70)
print("3. RANDOM OVERSAMPLING")
print("=" * 70)

df_minority_over = df_minority.sample(n=len(df_majority), replace=True, random_state=RANDOM_STATE)
df_oversampled = pd.concat([df_majority, df_minority_over]).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("New class counts:\n", df_oversampled["target"].value_counts().sort_index(), "\n")

# ---------------------------------------------------------------------------
# 4. SMOTE (Synthetic Minority Oversampling Technique)
#    Generates NEW synthetic points along the line between a minority sample
#    and one of its k nearest minority neighbors — avoids exact duplicates.
#    (Manual implementation below; equivalent to imblearn.over_sampling.SMOTE)
# ---------------------------------------------------------------------------
print("=" * 70)
print("4. SMOTE")
print("=" * 70)


def smote(X_minority, n_samples, k_neighbors=5, random_state=0):
    """Generate `n_samples` synthetic points for a minority class via SMOTE."""
    rs = np.random.RandomState(random_state)
    k = min(k_neighbors, len(X_minority) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)

    synthetic = np.zeros((n_samples, X_minority.shape[1]))
    for i in range(n_samples):
        base_idx = rs.randint(0, len(X_minority))
        neighbor = neighbor_idx[base_idx, rs.randint(1, k + 1)]  # skip self (col 0)
        gap = rs.rand()
        synthetic[i] = X_minority[base_idx] + gap * (X_minority[neighbor] - X_minority[base_idx])
    return synthetic


X_minority_arr = df_minority[feature_cols].values
n_to_generate = len(df_majority) - len(df_minority)
synthetic_points = smote(X_minority_arr, n_to_generate, k_neighbors=5, random_state=RANDOM_STATE)

df_synthetic = pd.DataFrame(synthetic_points, columns=feature_cols)
df_synthetic["target"] = minority_class

df_smote = pd.concat([df_majority, df_minority, df_synthetic]).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Generated {n_to_generate} synthetic minority samples")
print("New class counts:\n", df_smote["target"].value_counts().sort_index(), "\n")

# ---------------------------------------------------------------------------
# 5. CLASS WEIGHTS — keep the original (imbalanced) data, but tell the model
#    to penalize minority-class mistakes more heavily during training.
#    Best choice when you don't want to alter the data distribution at all.
# ---------------------------------------------------------------------------
print("=" * 70)
print("5. CLASS WEIGHTS (alternative to resampling)")
print("=" * 70)

classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))
print(f"Computed balanced class weights: {class_weight_dict}")
print("-> Pass this directly to most sklearn classifiers, e.g.:")
print("   LogisticRegression(class_weight=class_weight_dict)\n")

# ---------------------------------------------------------------------------
# 6. BEFORE/AFTER COMPARISON
# ---------------------------------------------------------------------------
print("=" * 70)
print("6. DISTRIBUTION COMPARISON")
print("=" * 70)

versions = {
    "Original (imbalanced)": df,
    "Undersampled": df_undersampled,
    "Oversampled": df_oversampled,
    "SMOTE": df_smote,
}

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (name, data) in zip(axes, versions.items()):
    sns.countplot(x="target", data=data, palette="Set2", ax=ax)
    ax.set_title(f"{name}\n(n={len(data)})")
plt.tight_layout()
plt.savefig("02_all_methods_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Bonus: does balancing actually help the classifier on the minority class?
# Train/test split FIRST, then balance only the TRAINING set (never touch
# the test set — that has to reflect the real, imbalanced world).
# ---------------------------------------------------------------------------
print("=" * 70)
print("BONUS: EFFECT ON A CLASSIFIER (minority class = class of interest)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

results = {}

# Baseline: trained on raw imbalanced data
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_train, y_train)
results["Imbalanced (baseline)"] = clf.predict(X_test)

# Class-weighted: same data, reweighted loss
clf_w = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_STATE).fit(X_train, y_train)
results["Class-weighted"] = clf_w.predict(X_test)

# SMOTE on the training set only
train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df["target"] = y_train
maj = train_df[train_df.target == majority_class]
minr = train_df[train_df.target == minority_class]
synth = smote(minr[feature_cols].values, len(maj) - len(minr), random_state=RANDOM_STATE)
synth_df = pd.DataFrame(synth, columns=feature_cols)
synth_df["target"] = minority_class
train_balanced = pd.concat([maj, minr, synth_df]).sample(frac=1, random_state=RANDOM_STATE)

clf_smote = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(
    train_balanced[feature_cols], train_balanced["target"])
results["SMOTE-balanced training set"] = clf_smote.predict(X_test)

for name, y_pred in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["class 0", "class 1"], digits=3))

print("Look at RECALL for class 1 (the minority class) across the three models —")
print("class weighting and SMOTE typically recover far more minority-class cases")
print("than the imbalanced baseline, usually at some cost to overall precision.")

print("\n" + "=" * 70)
print("DONE — figures saved:")
print("  01_original_distribution.png")
print("  02_all_methods_comparison.png")
print("=" * 70)