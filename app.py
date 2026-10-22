"""Flask web app: upload a flower photo, get the predicted species.

Run:  python app.py      then open http://127.0.0.1:5000
"""
import base64
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from features import read_image_bytes
from recognizer import LOW_CONFIDENCE, FlowerRecognizer

BASE = Path(__file__).parent
MODEL_PATH = os.environ.get("FLOWER_MODEL", str(BASE / "models" / "flower_model.joblib"))
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
_recognizer = None


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                "No trained model found. Run `python train.py` first.")
        _recognizer = FlowerRecognizer(MODEL_PATH)
    return _recognizer


def classify_upload(file_storage):
    """Validate an uploaded file and return (raw_bytes, predictions)."""
    if file_storage is None or file_storage.filename == "":
        raise ValueError("Please choose an image first.")
    if Path(file_storage.filename).suffix.lower() not in ALLOWED:
        raise ValueError("Unsupported file type. Use JPG, PNG, BMP or WEBP.")
    data = file_storage.read()
    image = read_image_bytes(data)
    return data, get_recognizer().predict(image, top_k=3)


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"threshold": LOW_CONFIDENCE}
    if request.method == "POST":
        try:
            file = request.files.get("image")
            data, preds = classify_upload(file)
            mime = file.mimetype or "image/jpeg"
            ctx.update(
                preview=f"data:{mime};base64,{base64.b64encode(data).decode()}",
                preds=preds,
                best=preds[0],
                low_confidence=preds[0][1] < LOW_CONFIDENCE,
            )
        except (ValueError, FileNotFoundError) as exc:
            ctx["error"] = str(exc)
    return render_template("index.html", **ctx)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API:  curl -F image=@rose.jpg http://127.0.0.1:5000/api/predict"""
    try:
        _, preds = classify_upload(request.files.get("image"))
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(
        prediction=preds[0][0],
        confidence=round(preds[0][1], 4),
        low_confidence=preds[0][1] < LOW_CONFIDENCE,
        top3=[{"flower": n, "confidence": round(p, 4)} for n, p in preds],
    )


@app.errorhandler(413)
def too_large(_):
    msg = "That file is too large (max 8 MB)."
    if request.path.startswith("/api/"):
        return jsonify(error=msg), 413
    return render_template("index.html", error=msg, threshold=LOW_CONFIDENCE), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
