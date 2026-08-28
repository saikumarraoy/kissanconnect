"""KissanConnect Flask prediction API using TensorFlow Lite inference."""

import io
import json
import os
from threading import Lock

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
try:
    # The lightweight production runtime avoids loading full TensorFlow on Render.
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # local conversion environment only
    from tensorflow.lite.python.interpreter import Interpreter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "kissanconnect_model.tflite")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
frontend_url = os.environ.get("FRONTEND_URL")
CORS(app, origins=[frontend_url] if frontend_url else "*")

print("Loading TensorFlow Lite model...", flush=True)
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
interpreter_lock = Lock()

with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
    class_names = json.load(file)
print(f"TFLite model loaded. {len(class_names)} classes available.", flush=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image_bytes: bytes) -> np.ndarray:
    """Resize and apply MobileNetV2's [-1, 1] preprocessing."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array / 127.5 - 1.0, axis=0)


def predict_tflite(image: np.ndarray) -> np.ndarray:
    """Run one prediction using the shared, thread-safe interpreter."""
    with interpreter_lock:
        interpreter.set_tensor(input_details["index"], image.astype(input_details["dtype"]))
        interpreter.invoke()
        return interpreter.get_tensor(output_details["index"])[0].copy()


def format_label(raw_label: str) -> dict:
    parts = raw_label.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return {"crop": crop, "condition": condition, "is_healthy": condition.lower() == "healthy"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "runtime": "tflite"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field name 'image'."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use png, jpg, or jpeg."}), 400

    try:
        predictions = predict_tflite(prepare_image(file.read()))
        top_idx = int(np.argmax(predictions))
        raw_label = class_names[top_idx]
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = [
            {"label": class_names[index], **format_label(class_names[index]), "confidence": float(predictions[index])}
            for index in top3_idx
        ]
        return jsonify({
            "prediction": {"raw_label": raw_label, **format_label(raw_label), "confidence": round(float(predictions[top_idx]), 4)},
            "top_3": top3,
        })
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {error}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
