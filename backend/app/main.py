"""
KissanConnect — Phase 2: Flask Prediction API

Loads the trained MobileNetV2 model from Phase 1 and exposes a /predict
endpoint that accepts a leaf image and returns the predicted disease.
"""

import io
import json
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import keras
from keras import ops

# --- Config -----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
MODEL_PATH = os.path.join(BASE_DIR, "models", "kissanconnect_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# --- App setup ----------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allows the React frontend (different port) to call this API

print("Loading model...")

@keras.saving.register_keras_serializable()
class TrueDivide(keras.Operation):
    def call(self, x, y):
        return ops.divide(x, y)


@keras.saving.register_keras_serializable()
class Subtract(keras.Operation):
    def call(self, x, y):
        return ops.subtract(x, y)
    
print("Loading model...")
model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "TrueDivide": TrueDivide,
        "Subtract": Subtract,
    },
)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)
print(f"Model loaded. {len(class_names)} classes available.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image_bytes: bytes) -> np.ndarray:
    """Resize + preprocess an uploaded image exactly like training did."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)  # same MobileNetV2 preprocessing used in training
    arr = np.expand_dims(arr, axis=0)  # model expects a batch dimension
    return arr


def format_label(raw_label: str) -> dict:
    """Turn 'Tomato___Late_blight' into a readable crop + disease pair."""
    parts = raw_label.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = condition.lower() == "healthy"
    return {"crop": crop, "condition": condition, "is_healthy": is_healthy}


# --- Routes ---------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field name 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use png, jpg, or jpeg."}), 400

    try:
        image_bytes = file.read()
        processed = prepare_image(image_bytes)

       print("Starting model prediction...", flush=True)

try:
    predictions = model.predict(processed, verbose=0)[0]
    print("Model prediction completed.", flush=True)
except Exception as e:
    print(f"MODEL PREDICTION ERROR: {repr(e)}", flush=True)
    return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])

        raw_label = class_names[top_idx]
        label_info = format_label(raw_label)

        # top 3 predictions, useful for uncertain/ambiguous cases
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = [
            {
                "label": class_names[i],
                **format_label(class_names[i]),
                "confidence": float(predictions[i]),
            }
            for i in top3_idx
        ]

        return jsonify({
            "prediction": {
                "raw_label": raw_label,
                **label_info,
                "confidence": round(confidence, 4),
            },
            "top_3": top3,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
