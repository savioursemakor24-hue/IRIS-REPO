import os
import io
import json
import sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_rf_model_full.joblib")
PORTABLE_PATH = os.path.join(BASE_DIR, "portable_model.json")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")
DB_PATH = os.path.join(BASE_DIR, "predictions_history.db")

FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
CLASSES = ["setosa", "versicolor", "virginica"]

app = Flask(__name__, template_folder="templates", static_folder="static")

# -------------------------------------------------------------
# Database Management (SQLite)
# -------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sepal_length REAL NOT NULL,
            sepal_width REAL NOT NULL,
            petal_length REAL NOT NULL,
            petal_width REAL NOT NULL,
            predicted_species TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_prediction(sl, sw, pl, pw, species, conf):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO predictions (timestamp, sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (now, sl, sw, pl, pw, species, conf))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Warning: Database write error: {e}")

def get_recent_history(limit=8):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        history = []
        for r in rows:
            history.append({
                "id": r[0],
                "timestamp": r[1],
                "sepal_length": r[2],
                "sepal_width": r[3],
                "petal_length": r[4],
                "petal_width": r[5],
                "species": r[6],
                "confidence": r[7]
            })
        return history
    except Exception as e:
        print(f"[!] Warning: Database read error: {e}")
        return []

# -------------------------------------------------------------
# Model Inference
# -------------------------------------------------------------
_MODEL = None
_PORTABLE = None

def get_model():
    global _MODEL, _PORTABLE
    if _MODEL is None and os.path.exists(MODEL_PATH):
        try:
            _MODEL = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[!] Warning: Could not load joblib model: {e}")

    if _PORTABLE is None and os.path.exists(PORTABLE_PATH):
        try:
            with open(PORTABLE_PATH, "r") as f:
                _PORTABLE = json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not load portable model: {e}")
    return _MODEL

PRESETS = {
    "setosa": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    "versicolor": {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5},
    "virginica": {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 1.8}
}

def predict_iris(sl, sw, pl, pw):
    model = get_model()
    features = [float(sl), float(sw), float(pl), float(pw)]

    if model is not None:
        x_arr = pd.DataFrame([features], columns=FEATURE_NAMES)
        pred_idx = int(model.predict(x_arr)[0])
        raw_probs = model.predict_proba(x_arr)[0].tolist()
        pred_class = CLASSES[pred_idx]
        conf_pct = round(raw_probs[pred_idx] * 100, 1)
        probs_dict = {
            "setosa": round(raw_probs[0] * 100, 1),
            "versicolor": round(raw_probs[1] * 100, 1),
            "virginica": round(raw_probs[2] * 100, 1)
        }
        return pred_class, conf_pct, probs_dict
    elif _PORTABLE is not None:
        mean = _PORTABLE["scaler_mean"]
        scale = _PORTABLE["scaler_scale"]
        coef = _PORTABLE["lr_coef"]
        intercept = _PORTABLE["lr_intercept"]
        x_scaled = [(features[i] - mean[i]) / scale[i] for i in range(4)]
        scores = [intercept[c] + sum(coef[c][f] * x_scaled[f] for f in range(4)) for c in range(3)]
        exp_s = [np.exp(s - max(scores)) for s in scores]
        sum_exp = sum(exp_s)
        probs = [e / sum_exp for e in exp_s]
        best_idx = int(np.argmax(probs))
        pred_class = CLASSES[best_idx]
        conf_pct = round(probs[best_idx] * 100, 1)
        probs_dict = {
            "setosa": round(probs[0] * 100, 1),
            "versicolor": round(probs[1] * 100, 1),
            "virginica": round(probs[2] * 100, 1)
        }
        return pred_class, conf_pct, probs_dict
    else:
        raise RuntimeError("No model available for inference.")

# -------------------------------------------------------------
# Web Routes
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    preset_name = request.args.get("preset", "").lower()
    if preset_name in PRESETS:
        vals = PRESETS[preset_name]
    else:
        vals = {"sepal_length": 5.8, "sepal_width": 3.0, "petal_length": 3.7, "petal_width": 1.2}

    pred_class, conf_pct, probs_dict = predict_iris(
        vals["sepal_length"], vals["sepal_width"], vals["petal_length"], vals["petal_width"]
    )
    history = get_recent_history()

    return render_template(
        "index.html",
        sepal_length=vals["sepal_length"],
        sepal_width=vals["sepal_width"],
        petal_length=vals["petal_length"],
        petal_width=vals["petal_width"],
        prediction=pred_class,
        confidence=conf_pct,
        probs=probs_dict,
        history=history,
        batch_count=request.args.get("batch_count")
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sl = float(request.form.get("sepal_length", 5.8))
        sw = float(request.form.get("sepal_width", 3.0))
        pl = float(request.form.get("petal_length", 3.7))
        pw = float(request.form.get("petal_width", 1.2))

        pred_class, conf_pct, probs_dict = predict_iris(sl, sw, pl, pw)
        log_prediction(sl, sw, pl, pw, pred_class, conf_pct)
        history = get_recent_history()

        return render_template(
            "index.html",
            sepal_length=sl,
            sepal_width=sw,
            petal_length=pl,
            petal_width=pw,
            prediction=pred_class,
            confidence=conf_pct,
            probs=probs_dict,
            history=history
        )
    except Exception as e:
        return render_template(
            "index.html",
            sepal_length=5.8, sepal_width=3.0, petal_length=3.7, petal_width=1.2,
            prediction="error",
            confidence=0.0,
            probs={"setosa": 0.0, "versicolor": 0.0, "virginica": 0.0},
            history=get_recent_history(),
            error_message=str(e)
        )

# Clear History Route
@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Error clearing database: {e}")
    return redirect(url_for("index"))

# Batch CSV Prediction Route
@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return "No file provided", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    try:
        df = pd.read_csv(file)
        # Normalize column names
        cols_map = {}
        for col in df.columns:
            clean = col.strip().lower().replace(" ", "_").replace("_(cm)", "").replace("(cm)", "")
            cols_map[col] = clean
        df_norm = df.rename(columns=cols_map)

        req_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for rc in req_cols:
            if rc not in df_norm.columns:
                return f"Missing required column: {rc}. CSV must contain sepal_length, sepal_width, petal_length, petal_width", 400

        predictions = []
        confidences = []
        for _, row in df_norm.iterrows():
            p_class, conf, _ = predict_iris(row["sepal_length"], row["sepal_width"], row["petal_length"], row["petal_width"])
            predictions.append(p_class)
            confidences.append(f"{conf}%")
            log_prediction(float(row["sepal_length"]), float(row["sepal_width"]), float(row["petal_length"]), float(row["petal_width"]), p_class, conf)

        df["predicted_species"] = predictions
        df["confidence"] = confidences

        out_io = io.StringIO()
        df.to_csv(out_io, index=False)
        out_bytes = io.BytesIO(out_io.getvalue().encode("utf-8"))
        out_bytes.seek(0)

        return send_file(
            out_bytes,
            mimetype="text/csv",
            as_attachment=True,
            download_name="iris_batch_predictions.csv"
        )
    except Exception as e:
        return f"Error processing CSV: {str(e)}", 400

# Download Sample CSV Template
@app.route("/sample-csv", methods=["GET"])
def sample_csv():
    sample_data = """sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
5.7,2.8,4.5,1.3
6.3,3.3,4.7,1.6
6.3,3.3,6.0,2.5
7.1,3.0,5.9,2.1
5.4,3.9,1.7,0.4
6.0,2.9,4.5,1.5
6.4,3.1,5.5,1.8
5.8,2.7,5.1,1.9"""
    buf = io.BytesIO(sample_data.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="iris_sample_test_batch.csv")

# JSON API
@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return jsonify({
            "service": "Iris Classification API (Python)",
            "status": "healthy",
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "classes": ["setosa", "versicolor", "virginica"],
            "database_logging": True
        })

    data = request.get_json(silent=True) or request.form.to_dict()
    sl = float(data.get("sepal_length", 5.8))
    sw = float(data.get("sepal_width", 3.0))
    pl = float(data.get("petal_length", 3.7))
    pw = float(data.get("petal_width", 1.2))

    pred_class, conf_pct, probs_dict = predict_iris(sl, sw, pl, pw)
    log_prediction(sl, sw, pl, pw, pred_class, conf_pct)

    return jsonify({
        "status": "success",
        "prediction": pred_class,
        "class_index": CLASSES.index(pred_class),
        "confidence": conf_pct / 100.0,
        "probabilities": {k: v / 100.0 for k, v in probs_dict.items()},
        "input_features": {"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[*] Starting Iris Flask Web App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
