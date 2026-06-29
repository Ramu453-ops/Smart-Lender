"""
Multivariate Analysis in Python
================================
A complete, runnable walkthrough of the core multivariate statistics
techniques, demonstrated on the classic Iris dataset (4 numeric variables,
3 groups). Swap in your own DataFrame wherever `df` and `target` are used.

Sections:
    1. Load data
    2. Descriptive statistics (means, std, per-group)
    3. Covariance & correlation matrices (+ heatmap)
    4. Principal Component Analysis (PCA)        -> dimensionality reduction
    5. MANOVA (Wilks' Lambda)                    -> do groups differ on ALL vars at once?
    6. K-Means clustering                        -> unsupervised grouping
    7. Linear Discriminant Analysis (LDA)         -> supervised separation/classification

Requirements: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
(All are standard with Anaconda; no statsmodels dependency required.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, adjusted_rand_score

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATA  (replace this block with pd.read_csv("your_file.csv") etc.)
# ---------------------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

numeric_cols = iris.feature_names          # the 4 continuous variables
target_col = "species"                    # the grouping variable

print("=" * 70)
print("1. DATA PREVIEW")
print("=" * 70)
print(df.head(), "\n")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ---------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
print("=" * 70)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 70)
print("Overall summary:\n", df[numeric_cols].describe().round(2), "\n")

print("Per-group means:\n", df.groupby(target_col)[numeric_cols].mean().round(2), "\n")
print("Per-group std dev:\n", df.groupby(target_col)[numeric_cols].std().round(2), "\n")

# ---------------------------------------------------------------------------
# 3. COVARIANCE & CORRELATION MATRICES
# ---------------------------------------------------------------------------
print("=" * 70)
print("3. COVARIANCE & CORRELATION")
print("=" * 70)
cov_matrix = df[numeric_cols].cov()
corr_matrix = df[numeric_cols].corr()
print("Covariance matrix:\n", cov_matrix.round(2), "\n")
print("Correlation matrix:\n", corr_matrix.round(2), "\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cov_matrix, annot=True, fmt=".1f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Covariance Matrix")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Correlation Matrix")
plt.tight_layout()
plt.savefig("01_covariance_correlation.png", dpi=150)
plt.close()

# Pairwise scatter plots colored by group — a quick "eyeball" multivariate view
sns.pairplot(df, hue=target_col, diag_kind="kde", palette="Set2", height=2)
plt.savefig("02_pairplot.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ---------------------------------------------------------------------------
print("=" * 70)
print("4. PRINCIPAL COMPONENT ANALYSIS")
print("=" * 70)

X = df[numeric_cols].values
X_scaled = StandardScaler().fit_transform(X)  # PCA is scale-sensitive — always standardize

pca = PCA()
scores = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

print("Explained variance ratio per component:")
for i, var in enumerate(explained, 1):
    print(f"  PC{i}: {var:.3f}  (cumulative: {explained[:i].sum():.3f})")
print()

print("Component loadings (how each original variable contributes to each PC):")
loadings = pd.DataFrame(pca.components_.T, index=numeric_cols,
                         columns=[f"PC{i+1}" for i in range(len(numeric_cols))])
print(loadings.round(3), "\n")

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(range(1, len(explained) + 1), explained, "o-", color="steelblue")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot")
axes[0].set_xticks(range(1, len(explained) + 1))

# Biplot: scores on PC1/PC2 colored by group, with loading vectors overlaid
colors = {sp: c for sp, c in zip(iris.target_names, sns.color_palette("Set2", 3))}
for sp in iris.target_names:
    mask = df[target_col] == sp
    axes[1].scatter(scores[mask, 0], scores[mask, 1], label=sp, alpha=0.7, color=colors[sp])
for i, var_name in enumerate(numeric_cols):
    axes[1].arrow(0, 0, loadings.iloc[i, 0] * 3, loadings.iloc[i, 1] * 3,
                  color="black", alpha=0.6, head_width=0.08)
    axes[1].text(loadings.iloc[i, 0] * 3.3, loadings.iloc[i, 1] * 3.3,
                 var_name, fontsize=8)
axes[1].set_xlabel(f"PC1 ({explained[0]:.1%} var)")
axes[1].set_ylabel(f"PC2 ({explained[1]:.1%} var)")
axes[1].set_title("PCA Biplot")
axes[1].legend()
axes[1].axhline(0, color="grey", lw=0.5)
axes[1].axvline(0, color="grey", lw=0.5)
plt.tight_layout()
plt.savefig("03_pca_scree_biplot.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. MANOVA — does the GROUP affect ALL variables jointly?
#    (Wilks' Lambda computed manually so no statsmodels dependency is needed.
#     If you have statsmodels installed, MANOVA.from_formula(...) gives the
#     same result plus Pillai's trace, Roy's root, etc.)
# ---------------------------------------------------------------------------
print("=" * 70)
print("5. MANOVA (Wilks' Lambda)")
print("=" * 70)

groups = df[target_col].unique()
n_total, p = X.shape
k = len(groups)

grand_mean = X.mean(axis=0)
SSCP_total = (X - grand_mean).T @ (X - grand_mean)        # total scatter

SSCP_within = np.zeros((p, p))
for g in groups:
    Xg = df.loc[df[target_col] == g, numeric_cols].values
    SSCP_within += (Xg - Xg.mean(axis=0)).T @ (Xg - Xg.mean(axis=0))

SSCP_between = SSCP_total - SSCP_within

wilks_lambda = np.linalg.det(SSCP_within) / np.linalg.det(SSCP_total)

# Rao's F-approximation for Wilks' Lambda significance
df1 = p * (k - 1)
s = np.sqrt((p**2 * (k - 1)**2 - 4) / (p**2 + (k - 1)**2 - 5)) if (p**2 + (k-1)**2 - 5) != 0 else 1
df2 = s * (n_total - 1 - (p + k) / 2) - (df1 - 2) / 2
exponent = 1 / s
F_stat = ((1 - wilks_lambda**exponent) / wilks_lambda**exponent) * (df2 / df1)
p_value = 1 - stats.f.cdf(F_stat, df1, df2)

print(f"Groups compared: {list(groups)}")
print(f"Wilks' Lambda = {wilks_lambda:.5f}")
print(f"Approx. F({df1:.1f}, {df2:.1f}) = {F_stat:.3f},  p-value = {p_value:.3e}")
print("-> Lambda close to 0 (and tiny p-value) means the groups differ strongly")
print("   on the combination of variables considered jointly.\n")

# ---------------------------------------------------------------------------
# 6. K-MEANS CLUSTERING — unsupervised: do natural clusters match the labels?
# ---------------------------------------------------------------------------
print("=" * 70)
print("6. K-MEANS CLUSTERING")
print("=" * 70)

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster"] = cluster_labels

ari = adjusted_rand_score(df[target_col].cat.codes, cluster_labels)
print(f"Adjusted Rand Index vs. true species labels: {ari:.3f}  (1.0 = perfect match)\n")
print("Cross-tab of true species vs. assigned cluster:")
print(pd.crosstab(df[target_col], df["cluster"]), "\n")

plt.figure(figsize=(6, 5))
plt.scatter(scores[:, 0], scores[:, 1], c=cluster_labels, cmap="Set2", s=40, edgecolor="k")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (plotted on PCA axes)")
plt.tight_layout()
plt.savefig("04_kmeans_clusters.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. LINEAR DISCRIMINANT ANALYSIS (LDA) — supervised: best axes to SEPARATE groups
# ---------------------------------------------------------------------------
print("=" * 70)
print("7. LINEAR DISCRIMINANT ANALYSIS")
print("=" * 70)

y = df[target_col].cat.codes.values
lda = LinearDiscriminantAnalysis()
lda_scores = lda.fit_transform(X_scaled, y)
y_pred = lda.predict(X_scaled)

print(f"Training accuracy: {(y_pred == y).mean():.3%}")
print("Confusion matrix:\n", confusion_matrix(y, y_pred), "\n")

plt.figure(figsize=(6, 5))
for i, sp in enumerate(iris.target_names):
    mask = y == i
    plt.scatter(lda_scores[mask, 0], lda_scores[mask, 1], label=sp,
                alpha=0.7, color=colors[sp])
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA: Maximally Separated Groups")
plt.legend()
plt.tight_layout()
plt.savefig("05_lda_separation.png", dpi=150)
plt.close()

print("=" * 70)
print("DONE — 5 figures saved as PNG files in the working directory:")
print("  01_covariance_correlation.png")
print("  02_pairplot.png")
print("  03_pca_scree_biplot.png")
print("  04_kmeans_clusters.png")
print("  05_lda_separation.png")
print("=" * 70)