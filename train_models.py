# ============================================================
#  Phase 4 — Model Training  |  Music Genre Classifier
# ============================================================
#  Trains k-NN, SVM (RBF), and Random Forest on features.csv
#  Saves each trained model to disk for Phase 5 evaluation
#
#  pip install scikit-learn pandas numpy joblib
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── 0. Config ─────────────────────────────────────────────────────────────────
CSV_PATH    = "features.csv"
MODELS_DIR  = "saved_models"
TEST_SIZE   = 0.20       # 80/20 split
RANDOM_SEED = 42
USE_PCA     = True       # set False to train on raw features
PCA_VARIANCE= 0.95       # keep components that explain 95% of variance
CV_FOLDS    = 5          # stratified k-fold for cross-validation

os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. Load & prepare data ────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

feature_cols = [c for c in df.columns if c not in ("filename", "label")]
X = df[feature_cols].values
y = df["label"].values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Genres : {sorted(set(y))}\n")

# Encode string labels → integers (needed for some sklearn internals)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Stratified train/test split — preserves genre proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_enc          # <-- critical: ensures each genre appears in both sets
)
print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")


# ── 2. Build preprocessing steps ─────────────────────────────────────────────
# StandardScaler: zero-mean, unit-variance — mandatory before SVM and k-NN
# PCA: optional dimensionality reduction (helps SVM speed, minor accuracy gain)

scaler = StandardScaler()

if USE_PCA:
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
    preprocessor = [("scaler", scaler), ("pca", pca)]
else:
    preprocessor = [("scaler", scaler)]


# ── 3. Define models ──────────────────────────────────────────────────────────
# Each model is wrapped in a Pipeline so preprocessing is always applied
# consistently — no risk of forgetting to scale before predicting.

models = {
    "knn": Pipeline(preprocessor + [
        ("clf", KNeighborsClassifier(
            n_neighbors=5,          # start with 5; tune in Phase 5
            metric="euclidean",     # alternatives: "manhattan", "minkowski"
            weights="uniform",      # "distance" weights closer neighbours more
        ))
    ]),

    "svm": Pipeline(preprocessor + [
        ("clf", SVC(
            kernel="rbf",           # RBF handles non-linear boundaries
            C=10,                   # regularisation — higher = tighter fit
            gamma="scale",          # "scale" = 1/(n_features * X.var()) — good default
            probability=True,       # enables predict_proba() for confidence scores
            random_state=RANDOM_SEED,
        ))
    ]),

    "random_forest": Pipeline(preprocessor + [
        ("clf", RandomForestClassifier(
            n_estimators=200,       # number of trees — more = better, slower
            max_depth=None,         # let trees grow fully
            min_samples_split=2,
            max_features="sqrt",    # sqrt(n_features) per split — standard for clf
            random_state=RANDOM_SEED,
            n_jobs=-1,              # use all CPU cores
        ))
    ]),
}


# ── 4. Train + quick cross-validate ──────────────────────────────────────────
print("=" * 55)
print(f"{'Model':<18} {'CV Acc (mean)':<16} {'CV Acc (std)'}")
print("=" * 55)

cv_results = {}

for name, pipeline in models.items():
    # 5-fold stratified cross-validation on TRAINING set only
    # (test set stays completely unseen until Phase 5)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_results[name] = cv_scores

    print(f"{name:<18} {cv_scores.mean():.4f}           ± {cv_scores.std():.4f}")

    # Refit on full training set and save
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, os.path.join(MODELS_DIR, f"{name}.pkl"))

print("=" * 55)


# ── 5. Quick test-set accuracy (preview — full eval is Phase 5) ───────────────
print("\nTest-set accuracy (preview):")
print("-" * 35)
for name, pipeline in models.items():
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"  {name:<18} {acc:.4f}  ({acc*100:.1f}%)")


# ── 6. Save encoder + column list (needed for Phase 5 & inference) ────────────
joblib.dump(le,           os.path.join(MODELS_DIR, "label_encoder.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

print(f"\nModels saved to: {MODELS_DIR}/")
print("  knn.pkl  |  svm.pkl  |  random_forest.pkl")
print("  label_encoder.pkl  |  feature_cols.pkl")


# ── 7. Predict a single new track (inference demo) ────────────────────────────
def predict_genre(wav_path: str, model_name: str = "svm") -> str:
    """
    Load a .wav file, extract features, and predict its genre.
    Uses the saved pipeline — no manual scaling needed.

    Usage:
        genre = predict_genre("my_song.wav", model_name="svm")
        print(genre)  # e.g. "jazz"
    """
    import librosa

    SR, DURATION, N_MFCC = 22050, 30, 13

    y_audio, sr = librosa.load(wav_path, sr=SR, duration=DURATION, mono=True)

    feats = {}
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfccs[i]))
    chroma   = librosa.feature.chroma_stft(y=y_audio, sr=sr)
    feats["chroma_mean"]            = float(np.mean(chroma))
    feats["chroma_std"]             = float(np.std(chroma))
    centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    feats["spectral_centroid_std"]  = float(np.std(centroid))
    rolloff  = librosa.feature.spectral_rolloff(y=y_audio, sr=sr)
    feats["spectral_rolloff_mean"]  = float(np.mean(rolloff))
    feats["spectral_rolloff_std"]   = float(np.std(rolloff))
    zcr      = librosa.feature.zero_crossing_rate(y_audio)
    feats["zcr_mean"]               = float(np.mean(zcr))
    feats["zcr_std"]                = float(np.std(zcr))
    rms      = librosa.feature.rms(y=y_audio)
    feats["rms_mean"]               = float(np.mean(rms))
    feats["rms_std"]                = float(np.std(rms))
    tempo, _ = librosa.beat.beat_track(y=y_audio, sr=sr)
    feats["tempo"]                  = float(np.asarray(tempo).flat[0])

    saved_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    saved_le   = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    X_new      = np.array([[feats[c] for c in saved_cols]])

    pipeline   = joblib.load(os.path.join(MODELS_DIR, f"{model_name}.pkl"))
    pred_idx   = pipeline.predict(X_new)[0]
    return saved_le.inverse_transform([pred_idx])[0]


# Uncomment to test on a file:
# print(predict_genre("Data/genres_original/jazz/jazz.00042.wav", "svm"))
