import os
import librosa
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH = "Data/genres_original"   # adjust to your GTZAN folder
CSV_OUTPUT   = "features.csv"
SR           = 22050                    # sample rate librosa resamples to
DURATION     = 30                       # seconds per track
N_MFCC       = 13                       # number of MFCC coefficients


# ── Feature extractor ──────────────────────────────────────────────────────────
def extract_features(file_path):
    """
    Load one .wav file and return a flat dict of audio features.
    Returns None if the file cannot be loaded.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)
    except Exception as e:
        print(f"  [skip] {file_path}: {e}")
        return None

    features = {}

    # 1. MFCCs — capture timbre / tonal texture
    #    Shape: (n_mfcc, T) → we take the mean & std across time frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i+1}_std"]  = float(np.std(mfccs[i]))

    # 2. Chroma — capture harmonic / pitch class content (12 bins, one per semitone)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"]  = float(np.std(chroma))

    # 3. Spectral centroid — "brightness" of the sound (Hz)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"]  = float(np.std(centroid))

    # 4. Spectral rolloff — frequency below which 85 % of energy falls
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"]  = float(np.std(rolloff))

    # 5. Zero-crossing rate — how often signal crosses zero (proxy for noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    # 6. RMS energy — loudness / dynamics
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"]  = float(np.std(rms))

    # 7. Tempo (BPM) — rhythmic speed
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).flat[0])

    return features


# ── Main loop ─────────────────────────────────────────────────────────────────
def build_dataset(dataset_path, output_csv):
    rows = []
    genres = sorted(os.listdir(dataset_path))

    for genre in genres:
        genre_dir = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_dir):
            continue

        files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        print(f"\nProcessing genre: {genre} ({len(files)} files)")

        for fname in files:
            fpath = os.path.join(genre_dir, fname)
            feats = extract_features(fpath)
            if feats is not None:
                feats["filename"] = fname
                feats["label"]    = genre
                rows.append(feats)

    df = pd.DataFrame(rows)

    # Reorder: filename and label at the front
    cols = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    df = df[cols]

    df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved {len(df)} rows → {output_csv}")
    print(f"Feature columns: {len(df.columns) - 2}")
    return df


# ── Quick sanity check (single file) ─────────────────────────────────────────
def preview_one(file_path):
    """Test on a single file before running the full dataset."""
    print(f"Extracting features from: {file_path}")
    feats = extract_features(file_path)
    if feats:
        for k, v in feats.items():
            print(f"  {k:35s} {v:.4f}")
    return feats


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Option A: test on one file first ---
    preview_one("Data/genres_original/blues/blues.00000.wav")

    # --- Option B: run full extraction ---
    df = build_dataset(DATASET_PATH, CSV_OUTPUT)
    print(df.head())