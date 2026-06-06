import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load Model & Artifacts ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model        = pickle.load(open(os.path.join(BASE_DIR, "fraud_model.pkl"), "rb"))
    scaler       = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    feature_cols = pickle.load(open(os.path.join(BASE_DIR, "feature_cols.pkl"), "rb"))
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    data = request.json

    try:
        time_val   = float(data.get("time", 0))
        amount_val = float(data.get("amount", 0))

        amount_scaled = scaler.transform([[amount_val]])[0][0]
        time_scaled   = scaler.transform([[time_val]])[0][0]

        sample = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
        if "Amount_scaled" in feature_cols:
            sample["Amount_scaled"] = amount_scaled
        if "Time_scaled" in feature_cols:
            sample["Time_scaled"] = time_scaled

        prediction  = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        return jsonify({
            "fraud":       int(prediction),
            "probability": round(float(probability) * 100, 2),
            "label":       "Fraud" if prediction == 1 else "Genuine"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/stats", methods=["GET"])
def stats():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "model":      "Random Forest",
        "estimators": model.n_estimators,
        "features":   len(feature_cols),
        "status":     "ready"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)