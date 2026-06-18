# ============================================================
#  Phase 3 — EDA & PCA  |  Music Genre Classifier
# ============================================================
#  Run this after feature_extraction.py has produced features.csv
#  pip install pandas numpy matplotlib seaborn scikit-learn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 0. Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10")
GENRE_COLORS = {
    "blues":     "#1f77b4",
    "classical": "#ff7f0e",
    "country":   "#2ca02c",
    "disco":     "#d62728",
    "hiphop":    "#9467bd",
    "jazz":      "#8c564b",
    "metal":     "#e377c2",
    "pop":       "#7f7f7f",
    "reggae":    "#bcbd22",
    "rock":      "#17becf",
}

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")
print(f"Dataset shape: {df.shape}")
print(f"Genres: {sorted(df['label'].unique())}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
print(df.describe().round(2))

feature_cols = [c for c in df.columns if c not in ("filename", "label")]
X = df[feature_cols].values
y = df["label"].values


# ── 2. Class balance ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
counts = df["label"].value_counts().sort_index()
bars = ax.bar(counts.index, counts.values,
              color=[GENRE_COLORS[g] for g in counts.index], edgecolor="white")
ax.bar_label(bars, padding=3, fontsize=10)
ax.set_title("Track count per genre (should be 100 each)", fontsize=13)
ax.set_xlabel("Genre")
ax.set_ylabel("Count")
ax.set_ylim(0, 120)
plt.tight_layout()
plt.savefig("plot1_class_balance.png", dpi=150)
plt.show()
# INTERPRETATION: All bars should be ~100. If any genre is lower, a file was
# corrupted or skipped during extraction — re-check that genre's folder.


# ── 3. Feature distributions — key features per genre ─────────────────────────
key_features = [
    "mfcc_1_mean", "spectral_centroid_mean",
    "zcr_mean",    "tempo",
    "chroma_mean", "rms_mean",
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    for genre in sorted(df["label"].unique()):
        vals = df[df["label"] == genre][feat]
        ax.hist(vals, bins=20, alpha=0.45,
                label=genre, color=GENRE_COLORS[genre], edgecolor="none")
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

handles = [mpatches.Patch(color=GENRE_COLORS[g], label=g)
           for g in sorted(df["label"].unique())]
fig.legend(handles=handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)
fig.suptitle("Feature distributions by genre", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plot2_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
# INTERPRETATION:
#  mfcc_1_mean    — classical tends very negative; hip-hop/metal cluster higher
#  spectral_centroid — metal/rock shift right (bright); blues/jazz shift left
#  zcr_mean       — metal is high (noisy guitars); classical is low (tonal)
#  tempo          — disco/hiphop cluster around 100–130 BPM
#  chroma_mean    — classical/jazz show distinct pitch content
#  rms_mean       — metal/rock will be louder (higher RMS)


# ── 4. Boxplots — genre spread for top 4 discriminating features ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
box_features = ["mfcc_1_mean", "spectral_centroid_mean", "zcr_mean", "tempo"]
genres_sorted = sorted(df["label"].unique())

for ax, feat in zip(axes.flatten(), box_features):
    data_by_genre = [df[df["label"] == g][feat].values for g in genres_sorted]
    bp = ax.boxplot(data_by_genre, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, genre in zip(bp["boxes"], genres_sorted):
        patch.set_facecolor(GENRE_COLORS[genre])
        patch.set_alpha(0.8)
    ax.set_xticklabels(genres_sorted, rotation=35, ha="right", fontsize=9)
    ax.set_title(feat, fontsize=11)
    ax.set_ylabel("Value")

fig.suptitle("Genre spread — key features", fontsize=14)
plt.tight_layout()
plt.savefig("plot3_boxplots.png", dpi=150)
plt.show()
# INTERPRETATION: Non-overlapping boxes = feature separates those genres well.
# Heavily overlapping boxes = feature alone won't distinguish those genres —
# but combining features (what SVM/RF do) still works.


# ── 5. Correlation heatmap ─────────────────────────────────────────────────────
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, linewidths=0.3,
            annot=False, ax=ax)
ax.set_title("Feature correlation matrix", fontsize=13)
plt.tight_layout()
plt.savefig("plot4_correlation.png", dpi=150)
plt.show()
# INTERPRETATION:
#  Deep red blocks  = highly correlated features — carry redundant info.
#  MFCC coefficients are often correlated with each other (expected).
#  PCA (next step) will collapse this redundancy into orthogonal components.


# ── 6. PCA ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # CRITICAL: PCA needs scaled data

pca_full = PCA()
pca_full.fit(X_scaled)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# 6a. Scree plot — how many components do we need?
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

n_show = min(20, len(explained))
ax1.bar(range(1, n_show + 1), explained[:n_show] * 100,
        color="#378ADD", edgecolor="white")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Variance explained (%)")
ax1.set_title("Scree plot")

ax2.plot(range(1, len(cumulative) + 1), cumulative * 100,
         marker="o", markersize=4, color="#378ADD", linewidth=1.5)
ax2.axhline(90, color="#E24B4A", linestyle="--", linewidth=1, label="90% threshold")
ax2.axhline(95, color="#EF9F27", linestyle="--", linewidth=1, label="95% threshold")
ax2.set_xlabel("Number of components")
ax2.set_ylabel("Cumulative variance (%)")
ax2.set_title("Cumulative explained variance")
ax2.legend()

n90 = int(np.searchsorted(cumulative, 0.90)) + 1
n95 = int(np.searchsorted(cumulative, 0.95)) + 1
print(f"\nPCA: {n90} components explain 90% of variance")
print(f"PCA: {n95} components explain 95% of variance")

plt.tight_layout()
plt.savefig("plot5_scree.png", dpi=150)
plt.show()
# INTERPRETATION: The "elbow" in the scree plot shows diminishing returns.
# Use however many components reach 90–95% — typically 8–12 for GTZAN.
# For the 2-D scatter below we always use just PC1 & PC2 for visualisation.


# 6b. 2-D PCA scatter — do genres cluster?
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
for genre in genres_sorted:
    mask = y == genre
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=GENRE_COLORS[genre], label=genre,
               alpha=0.6, s=30, edgecolors="none")

ax.set_xlabel(f"PC1  ({explained[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2  ({explained[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("PCA scatter — PC1 vs PC2 (coloured by genre)", fontsize=13)
ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig("plot6_pca_scatter.png", dpi=150)
plt.show()
# INTERPRETATION:
#  Tight, well-separated clusters → genres are linearly distinguishable.
#  Expect: classical isolated (very different timbre), metal/rock overlapping,
#  blues/jazz overlapping. Messy overlap = need non-linear model (SVM RBF / RF).
#  This plot is your EDA notebook's hero figure — include it in your report!


# 6c. PCA loading plot — what does PC1/PC2 actually represent?
loadings = pd.DataFrame(
    pca2.components_.T,
    index=feature_cols,
    columns=["PC1", "PC2"]
)
top_n = 10
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, pc in zip(axes, ["PC1", "PC2"]):
    top = loadings[pc].abs().nlargest(top_n).index
    vals = loadings.loc[top, pc].sort_values()
    colors = ["#E24B4A" if v < 0 else "#378ADD" for v in vals]
    ax.barh(top, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{pc} — top {top_n} feature loadings", fontsize=11)
    ax.set_xlabel("Loading weight")

plt.tight_layout()
plt.savefig("plot7_pca_loadings.png", dpi=150)
plt.show()
# INTERPRETATION: Large positive/negative loadings tell you which features
# drive each principal component. If mfcc_1_mean dominates PC1, that axis
# is essentially a "timbre axis". Useful for your report's discussion section.


# ── 7. Genre mean feature heatmap ─────────────────────────────────────────────
# Normalise so each feature is on the same 0-1 scale for visual comparison
genre_means = df.groupby("label")[feature_cols].mean()
genre_means_norm = (genre_means - genre_means.min()) / \
                   (genre_means.max() - genre_means.min() + 1e-9)

# Keep only the 15 most variable features to keep the plot readable
top_feats = genre_means.std().nlargest(15).index
genre_means_norm = genre_means_norm[top_feats]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(genre_means_norm.T, cmap="YlOrRd",
            linewidths=0.4, annot=True, fmt=".2f",
            cbar_kws={"label": "Normalised mean"}, ax=ax)
ax.set_title("Genre mean feature heatmap (top 15 most variable features)", fontsize=13)
ax.set_xlabel("Genre")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("plot8_genre_heatmap.png", dpi=150)
plt.show()
# INTERPRETATION: Bright cells = genre has distinctively high value for that
# feature. This is your most "report-ready" table — it shows at a glance why
# classical looks different from metal, for example.

print("\nAll plots saved! Files: plot1_class_balance.png … plot8_genre_heatmap.png")
