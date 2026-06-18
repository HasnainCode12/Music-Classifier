
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score,
    ConfusionMatrixDisplay
)

CSV_PATH   = "features.csv"
MODELS_DIR = "saved_models"
OUT_DIR    = "evaluation_plots"
CV_FOLDS   = 5
RANDOM_SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)

GENRE_COLORS = {
    "blues":     "#1f77b4", "classical": "#ff7f0e", "country":  "#2ca02c",
    "disco":     "#d62728", "hiphop":    "#9467bd", "jazz":     "#8c564b",
    "metal":     "#e377c2", "pop":       "#7f7f7f", "reggae":   "#bcbd22",
    "rock":      "#17becf",
}
MODEL_COLORS = {"knn": "#7F77DD", "svm": "#1D9E75", "random_forest": "#BA7517"}
MODEL_LABELS = {"knn": "k-NN", "svm": "SVM (RBF)", "random_forest": "Random Forest"}


df = pd.read_csv(CSV_PATH)
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

X    = df[feature_cols].values
y    = le.transform(df["label"].values) 
genres = list(le.classes_)  

pipelines = {
    name: joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))
    for name in ("knn", "svm", "random_forest")
}

print(f"Genres : {genres}")
print(f"Samples: {len(X)}\n")


skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

cv_summary = {} 

print("=" * 62)
print(f"{'Model':<16} {'Acc':>6}  {'±':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
print("=" * 62)

for name, pipeline in pipelines.items():
    results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        n_jobs=-1,
        return_train_score=False,
    )
    cv_summary[name] = {
        "acc_mean":  results["test_accuracy"].mean(),
        "acc_std":   results["test_accuracy"].std(),
        "prec_mean": results["test_precision_macro"].mean(),
        "rec_mean":  results["test_recall_macro"].mean(),
        "f1_mean":   results["test_f1_macro"].mean(),
        "fold_accs": results["test_accuracy"],
    }
    r = cv_summary[name]
    print(f"{MODEL_LABELS[name]:<16} {r['acc_mean']:.4f}  "
          f"±{r['acc_std']:.3f}  {r['prec_mean']:.4f}  "
          f"{r['rec_mean']:.4f}  {r['f1_mean']:.4f}")

print("=" * 62)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Confusion matrices — test set (20%)", fontsize=14, y=1.02)

test_preds = {}

for ax, (name, pipeline) in zip(axes, pipelines.items()):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_preds[name] = y_pred

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                  display_labels=genres)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(f"{MODEL_LABELS[name]}\n"
                 f"Acc: {accuracy_score(y_test, y_pred):.3f}", fontsize=11)
    ax.set_xticklabels(genres, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(genres, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150,
            bbox_inches="tight")
plt.show()

best_name = max(cv_summary, key=lambda n: cv_summary[n]["f1_mean"])
print(f"\nBest model by CV macro-F1: {MODEL_LABELS[best_name]}")

y_pred_best = test_preds[best_name]

report = classification_report(
    y_test, y_pred_best,
    target_names=genres,
    output_dict=True
)

report_df = pd.DataFrame(report).T.loc[genres, ["precision", "recall", "f1-score"]]

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    report_df.astype(float),
    annot=True, fmt=".2f",
    cmap="YlGn", vmin=0, vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Score"},
    ax=ax
)
ax.set_title(f"Per-class precision / recall / F1  —  {MODEL_LABELS[best_name]}",
             fontsize=13)
ax.set_xlabel("Metric")
ax.set_ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_metrics.png"), dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(CV_FOLDS)
width = 0.25

for i, (name, res) in enumerate(cv_summary.items()):
    bars = ax.bar(
        x + i * width,
        res["fold_accs"] * 100,
        width=width,
        label=MODEL_LABELS[name],
        color=MODEL_COLORS[name],
        edgecolor="white",
        alpha=0.85
    )

ax.set_xticks(x + width)
ax.set_xticklabels([f"Fold {i+1}" for i in range(CV_FOLDS)])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 100)
ax.set_title("Accuracy per CV fold — all models", fontsize=13)
ax.legend()
ax.axhline(
    cv_summary["svm"]["acc_mean"] * 100,
    color=MODEL_COLORS["svm"], linestyle="--", linewidth=1, alpha=0.5
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cv_folds.png"), dpi=150)
plt.show()

summary_rows = []
for name in ("knn", "svm", "random_forest"):
    r = cv_summary[name]
    summary_rows.append({
        "Model":         MODEL_LABELS[name],
        "CV Acc (mean)": f"{r['acc_mean']:.4f}",
        "CV Acc (±std)": f"±{r['acc_std']:.4f}",
        "Macro Prec":    f"{r['prec_mean']:.4f}",
        "Macro Recall":  f"{r['rec_mean']:.4f}",
        "Macro F1":      f"{r['f1_mean']:.4f}",
    })
summary_df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(11, 2.2))
ax.axis("off")
tbl = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)

best_row = list(MODEL_LABELS.keys()).index(best_name) + 1
for col in range(len(summary_df.columns)):
    tbl[0, col].set_facecolor("#2C2C2A")
    tbl[0, col].set_text_props(color="white")
    tbl[best_row, col].set_facecolor("#E1F5EE")

ax.set_title("Model comparison — stratified 5-fold CV", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison_table.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print("\nModel comparison:")
print(summary_df.to_string(index=False))

rf_pipeline = pipelines["random_forest"]
rf_clf      = rf_pipeline.named_steps["clf"]
importances = rf_clf.feature_importances_

if "pca" in rf_pipeline.named_steps:
    print("\nNote: PCA is enabled — feature importances reflect PCA components,")
    print("not original features. Set USE_PCA=False in train_models.py to get")
    print("per-feature importances from Random Forest.")
else:
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    top15    = feat_imp.head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top15.index[::-1], top15.values[::-1],
                   color="#BA7517", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title("Top 15 features — Random Forest", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.show()

print(f"\n{'='*55}")
print(f"Full classification report — {MODEL_LABELS[best_name]}")
print(f"{'='*55}")
print(classification_report(y_test, y_pred_best, target_names=genres))

print(f"\nAll plots saved to: {OUT_DIR}/")
print("  confusion_matrices.png")
print("  per_class_metrics.png")
print("  cv_folds.png")
print("  model_comparison_table.png")
print("  feature_importance.png  (if USE_PCA=False in train_models.py)")
