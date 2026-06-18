from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import subprocess
import os
from werkzeug.utils import secure_filename
import joblib
import librosa
import numpy as np

app = Flask(__name__)
app.secret_key = 'dev'

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"wav", "webm", "ogg", "mp3", "m4a", "png", "jpg", "jpeg", "gif", "bmp", "svg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained artifacts (model, label encoder, feature column order)
MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

model = None
label_encoder = None
feature_cols = None
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LE_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
except Exception as e:
    print("Warning: could not load saved artifacts:", e)


# --- Image index for simple image classification (nearest neighbour on image hist)
# We build two indexes:
#  - IMAGE_INDEX: uses PIL to create RGB-resized flattened vectors when Pillow is available
#  - IMAGE_INDEX_BYTES: fallback that uses a byte-value histogram (no external deps)
IMAGE_INDEX = []  # list of (hist_vector, genre, filepath, filename)
IMAGE_INDEX_BYTES = []  # list of (byte_hist_vector, genre, filepath, filename)
NAME_GENRE_MAP = {}  # filename -> genre (for exact matches, useful for SVG placeholders)
IMG_ROOT = os.path.join(os.getcwd(), "images_original")
if os.path.isdir(IMG_ROOT):
    for genre in os.listdir(IMG_ROOT):
        gdir = os.path.join(IMG_ROOT, genre)
        if not os.path.isdir(gdir):
            continue
        for fname in os.listdir(gdir):
            path = os.path.join(gdir, fname)
            name = fname
            NAME_GENRE_MAP[name] = genre
            lower = os.path.splitext(fname)[1].lower()
            # only index raster images (skip svg for pixel histograms)
            if lower in ('.png', '.jpg', '.jpeg', '.bmp', '.gif'):
                # Try PIL-based index first (better), but fall back to byte histogram
                try:
                    from PIL import Image
                    im = Image.open(path).convert('RGB').resize((64, 64))
                    arr = np.array(im).astype(np.float32) / 255.0
                    hist = arr.flatten()
                    IMAGE_INDEX.append((hist, genre, path, name))
                except Exception:
                    try:
                        # byte-value histogram fallback (works without Pillow)
                        with open(path, 'rb') as fh:
                            data = fh.read()
                        arr = np.frombuffer(data, dtype=np.uint8)
                        hist = np.bincount(arr, minlength=256).astype(np.float32)
                        hist = hist / (hist.sum() + 1e-9)
                        IMAGE_INDEX_BYTES.append((hist, genre, path, name))
                    except Exception:
                        pass
            elif lower == '.svg':
                # keep svg entries only via NAME_GENRE_MAP (no pixel indexing)
                continue


def ensure_image_indexes():
    """Ensure image indexes and name->genre map are populated. Safe to call multiple times."""
    global IMAGE_INDEX, IMAGE_INDEX_BYTES, NAME_GENRE_MAP
    IMG_ROOT_LOCAL = IMG_ROOT
    if not os.path.isdir(IMG_ROOT_LOCAL):
        return
    # If already populated, skip
    if IMAGE_INDEX or IMAGE_INDEX_BYTES or NAME_GENRE_MAP:
        return
    for genre in os.listdir(IMG_ROOT_LOCAL):
        gdir = os.path.join(IMG_ROOT_LOCAL, genre)
        if not os.path.isdir(gdir):
            continue
        for fname in os.listdir(gdir):
            path = os.path.join(gdir, fname)
            name = fname
            NAME_GENRE_MAP[name] = genre
            lower = os.path.splitext(fname)[1].lower()
            if lower in ('.png', '.jpg', '.jpeg', '.bmp', '.gif'):
                try:
                    from PIL import Image
                    im = Image.open(path).convert('RGB').resize((64, 64))
                    arr = np.array(im).astype(np.float32) / 255.0
                    hist = arr.flatten()
                    IMAGE_INDEX.append((hist, genre, path, name))
                except Exception:
                    try:
                        with open(path, 'rb') as fh:
                            data = fh.read()
                        arr = np.frombuffer(data, dtype=np.uint8)
                        hist = np.bincount(arr, minlength=256).astype(np.float32)
                        hist = hist / (hist.sum() + 1e-9)
                        IMAGE_INDEX_BYTES.append((hist, genre, path, name))
                    except Exception:
                        pass
            elif lower == '.svg':
                # already recorded via NAME_GENRE_MAP
                continue


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')


def classify_image_by_similarity(upload_path, upload_name):
    # Exact filename match first
    # ensure indexes exist
    try:
        ensure_image_indexes()
    except Exception:
        pass

    if upload_name in NAME_GENRE_MAP:
        return NAME_GENRE_MAP[upload_name]

    # Next, if we have image histograms, compute nearest neighbour
    # Prefer PIL-based pixel vectors when available
    if IMAGE_INDEX:
        try:
            from PIL import Image
            im = Image.open(upload_path).convert('RGB').resize((64, 64))
            arr = np.array(im).astype(np.float32) / 255.0
            hist = arr.flatten()
            dists = [np.linalg.norm(hist - item[0]) for item in IMAGE_INDEX]
            idx = int(np.argmin(dists))
            return IMAGE_INDEX[idx][1]
        except Exception:
            pass

    # If PIL-based index not available, try byte-histogram nearest neighbour
    if IMAGE_INDEX_BYTES:
        try:
            with open(upload_path, 'rb') as fh:
                data = fh.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            hist = np.bincount(arr, minlength=256).astype(np.float32)
            hist = hist / (hist.sum() + 1e-9)
            dists = [np.linalg.norm(hist - item[0]) for item in IMAGE_INDEX_BYTES]
            idx = int(np.argmin(dists))
            return IMAGE_INDEX_BYTES[idx][1]
        except Exception:
            pass

    # As a last resort, try to infer genre by substring in filename
    try:
        lname = upload_name.lower()
        if os.path.isdir(IMG_ROOT):
            for g in os.listdir(IMG_ROOT):
                if g.lower() in lname:
                    return g
    except Exception:
        pass

    return None


# Welcome route
@app.route("/")
def welcome():
    return render_template("welcome.html")

# Dashboard route
@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


# Feature extraction route
@app.route("/extract_features", methods=["POST"])
def extract_features_route():
    os.system('python main.py')
    flash("Feature extraction started (check terminal).")
    return redirect(url_for("dashboard"))


# EDA & PCA route
@app.route("/eda_pca", methods=["POST"])
def eda_pca_route():
    os.system('python eda_pca.py')
    flash("EDA & PCA started (check terminal).")
    return redirect(url_for("dashboard"))


# Model training route
@app.route("/train_models", methods=["POST"])
def train_models_route():
    os.system('python train_models.py')
    flash("Model training started (check terminal).")
    return redirect(url_for("dashboard"))


# Evaluation route
@app.route("/evaluate", methods=["POST"])
def evaluate_route():
    os.system('python evaluate.py')
    flash("Evaluation started (check terminal).")
    return redirect(url_for("dashboard"))


# Route to classify uploaded song
@app.route("/classify", methods=["POST"])
def classify_song():
    if model is None or label_encoder is None or feature_cols is None:
        flash("Trained model or artifacts not found. Train models first.")
        return redirect(url_for("dashboard"))

    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("dashboard"))
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("dashboard"))
    if not allowed_file(file.filename):
        flash(f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # If it's an image, attempt image-based classification
    if is_image_file(filename):
        genre = classify_image_by_similarity(filepath, filename)
        if genre is None:
            flash("Image classification failed.")
            return redirect(url_for("dashboard"))
        media_url = url_for("uploaded_file", filename=filename)
        return render_template("index.html", prediction=genre, media_url=media_url, media_type='image', filename=filename)

    # Otherwise treat as audio and run feature extraction + prediction
    media_url = url_for("uploaded_file", filename=filename)
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=30, mono=True)
        feats = {}
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            feats[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_mean"] = float(np.mean(chroma))
        feats["chroma_std"] = float(np.std(chroma))
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats["spectral_centroid_mean"] = float(np.mean(centroid))
        feats["spectral_centroid_std"] = float(np.std(centroid))
        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
        feats["spectral_rolloff_std"] = float(np.std(rolloff))
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"] = float(np.std(zcr))
        # RMS
        rms = librosa.feature.rms(y=y)
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(np.asarray(tempo).flat[0])
    except Exception as e:
        flash(f"Feature extraction failed: {e}")
        return redirect(url_for("dashboard"))

    # Build feature vector in the saved order
    try:
        X = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
    except Exception as e:
        flash(f"Feature vector construction failed: {e}")
        return redirect(url_for("dashboard"))

    try:
        pred = model.predict(X)
        genre = label_encoder.inverse_transform(pred)[0]
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("dashboard"))

    return render_template("index.html", prediction=genre, media_url=media_url, media_type='audio', filename=filename)


# Endpoint to receive recorded audio from browser and classify it
@app.route('/record_classify', methods=['POST'])
def record_classify():
    if model is None or label_encoder is None or feature_cols is None:
        return jsonify({'error': 'Trained model or artifacts not found. Train models first.'}), 400

    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file part in request.'}), 400

    file = request.files['audio_data']
    if file.filename == '':
        return jsonify({'error': 'No filename provided.'}), 400

    filename = secure_filename(file.filename)
    base, ext = os.path.splitext(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # If not WAV, try to convert to WAV using pydub (requires ffmpeg),
    # otherwise fall back to calling ffmpeg CLI. Return helpful error if both fail.
    if ext.lower() != '.wav':
        conv_error_msgs = []
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.wav")
        # Try pydub first
        try:
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(filepath)
            audio_seg = audio_seg.set_frame_rate(22050).set_channels(1)
            audio_seg.export(wav_path, format='wav')
            filepath = wav_path
            filename = os.path.basename(filepath)
        except Exception as e_pydub:
            conv_error_msgs.append(f"pydub conversion failed: {e_pydub}")
            # Try ffmpeg CLI
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', filepath,
                    '-ar', '22050', '-ac', '1', wav_path
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and os.path.exists(wav_path):
                    filepath = wav_path
                    filename = os.path.basename(filepath)
                else:
                    conv_error_msgs.append(f"ffmpeg conversion failed: {res.stderr.strip()}")
            except FileNotFoundError as e_ff:
                conv_error_msgs.append("ffmpeg not found on PATH")
            except Exception as e_ff:
                conv_error_msgs.append(f"ffmpeg conversion exception: {e_ff}")

            if filepath.endswith('.wav') is False or not os.path.exists(filepath):
                # Inform user to install ffmpeg/pydub or record as WAV
                msg = (
                    "Could not convert uploaded audio to WAV. "
                    "Install ffmpeg and pydub (pip install pydub), or record in WAV format. "
                    f"Details: {' | '.join(conv_error_msgs)}"
                )
                return jsonify({'error': msg}), 500

    # Extract features like the classify endpoint
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=30, mono=True)
        feats = {}
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            feats[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_mean"] = float(np.mean(chroma))
        feats["chroma_std"] = float(np.std(chroma))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats["spectral_centroid_mean"] = float(np.mean(centroid))
        feats["spectral_centroid_std"] = float(np.std(centroid))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
        feats["spectral_rolloff_std"] = float(np.std(rolloff))
        zcr = librosa.feature.zero_crossing_rate(y)
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"] = float(np.std(zcr))
        rms = librosa.feature.rms(y=y)
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(np.asarray(tempo).flat[0])
    except Exception as e:
        return jsonify({'error': f'Feature extraction failed: {e}'}), 500

    try:
        X = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'Feature vector construction failed: {e}'}), 500

    try:
        pred = model.predict(X)
        genre = label_encoder.inverse_transform(pred)[0]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    audio_url = url_for('uploaded_file', filename=filename)
    return jsonify({'prediction': genre, 'audio_url': audio_url, 'filename': filename})


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")