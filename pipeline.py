"""
Unsupervised Learning Pipeline: K-Means + PCA on the UCI Wine dataset
=======================================================================
Dataset: 178 wine samples, 13 numeric chemical/physical measurements,
derived from 3 different cultivars grown in the same region of Italy.
The cultivar label is DROPPED before clustering (treated as unlabeled)
and only reattached at the very end to sanity-check what the clusters found.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

plt.rcParams["figure.dpi"] = 120
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data (labels dropped for the unsupervised exercise)
# ---------------------------------------------------------------------
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
true_labels = data.target          # kept aside only for final interpretation
true_label_names = data.target_names

print("Dataset shape:", X.shape)
print("Features:", list(X.columns))

# ---------------------------------------------------------------------
# 2. Standardize features (K-Means is distance-based)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# 3. Elbow method + silhouette score across a range of k
# ---------------------------------------------------------------------
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), sil_scores, marker="o", color="#DD8452")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average silhouette score")
axes[1].grid(alpha=0.3)

best_k_by_sil = list(k_range)[int(np.argmax(sil_scores))]
axes[1].axvline(best_k_by_sil, color="green", linestyle="--", alpha=0.6,
                 label=f"best k = {best_k_by_sil}")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/01_elbow_silhouette.png", bbox_inches="tight")
plt.close()

print("\nInertia by k:", dict(zip(k_range, np.round(inertias, 1))))
print("Silhouette by k:", dict(zip(k_range, np.round(sil_scores, 3))))
print(f"Silhouette-optimal k = {best_k_by_sil}")

# ---------------------------------------------------------------------
# 4. Fit final K-Means with chosen k
# ---------------------------------------------------------------------
K = best_k_by_sil  # chosen from silhouette score (elbow agrees - see plot)
kmeans_final = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
cluster_labels = kmeans_final.fit_predict(X_scaled)

X_with_clusters = X.copy()
X_with_clusters["cluster"] = cluster_labels
X_with_clusters["true_cultivar"] = true_labels

# ---------------------------------------------------------------------
# 5. PCA to 2 components for visualization
# ---------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
print(f"\nExplained variance ratio: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
print(f"Total variance captured by 2D projection: {explained_var.sum():.3f} ({explained_var.sum()*100:.1f}%)")

# Also fit 3-component PCA to report cumulative variance for reference
pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
pca3.fit(X_scaled)
print(f"Explained variance ratio (3 comps): {np.round(pca3.explained_variance_ratio_, 3)}")
print(f"Cumulative variance (3 comps): {pca3.explained_variance_ratio_.sum():.3f}")

# ---------------------------------------------------------------------
# 6. Scatter plot: PCA-reduced data colored by cluster
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for c in range(K):
    mask = cluster_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=45, alpha=0.75, color=palette[c % len(palette)],
               label=f"Cluster {c} (n={mask.sum()})", edgecolor="white", linewidth=0.4)

# Plot centroids projected into PCA space
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           marker="X", s=250, color="black", label="Centroids", zorder=5)

ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
ax.set_title(f"K-Means Clusters (k={K}) Visualized via PCA")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/02_pca_clusters_scatter.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 7. Sanity check against real cultivar labels (not used in fitting)
# ---------------------------------------------------------------------
crosstab = pd.crosstab(X_with_clusters["cluster"], 
                        X_with_clusters["true_cultivar"].map(dict(enumerate(true_label_names))))
print("\nCluster vs. true cultivar crosstab (labels were hidden during fitting):")
print(crosstab)

ari = adjusted_rand_score(true_labels, cluster_labels)
print(f"\nAdjusted Rand Index vs true cultivar labels: {ari:.3f}")

# ---------------------------------------------------------------------
# 8. Cluster profiles (mean feature values per cluster, in original units)
# ---------------------------------------------------------------------
profile = X_with_clusters.groupby("cluster")[X.columns].mean().round(2)
print("\nCluster feature-mean profiles (original units):")
print(profile.T)

profile.T.to_csv("cluster_profiles.csv")
crosstab.to_csv("cluster_vs_true_labels.csv")
X_with_clusters.to_csv("wine_with_clusters.csv", index=False)

print("\nDone. Plots saved to plots/, tables saved as CSV.")
