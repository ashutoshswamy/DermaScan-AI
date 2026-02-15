"""
Skin Cancer Detection — Flask Web Application
===============================================
Serves a modern web UI for uploading skin lesion images and
receiving AI-powered classification results.

Usage:
    python app.py

Then open http://localhost:3000 in your browser.
"""

import os
import uuid
import time
import secrets
from pathlib import Path
from collections import defaultdict
from functools import wraps

from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ALLOWED_MIMETYPES = {
    "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
}
MAX_FILE_SIZE_MB = 10

# Rate limiting: max requests per IP per window
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60  # seconds

# HAM10000 class labels with clinical context
CLASS_INFO = {
    "actinic keratosis": {
        "risk": "Pre-cancerous",
        "description": "Rough, scaly patch caused by sun damage. Can progress to squamous cell carcinoma.",
        "recommendation": "Consult a dermatologist for evaluation and possible treatment.",
    },
    "basal cell carcinoma": {
        "risk": "Malignant",
        "description": "Most common skin cancer. Slow-growing, rarely metastasizes, but requires treatment.",
        "recommendation": "Seek immediate dermatological evaluation for biopsy and treatment options.",
    },
    "dermatofibroma": {
        "risk": "Benign",
        "description": "Harmless, firm nodule. No treatment necessary unless symptomatic.",
        "recommendation": "Monitor for changes. No urgent action needed.",
    },
    "melanoma": {
        "risk": "Malignant",
        "description": "Most dangerous skin cancer. Early detection is critical for survival.",
        "recommendation": "Urgent: Consult a dermatologist or oncologist immediately.",
    },
    "nevus": {
        "risk": "Benign",
        "description": "Common mole. Usually harmless but should be monitored for changes.",
        "recommendation": "Use the ABCDE rule to monitor. Consult a doctor if changes occur.",
    },
    "pigmented benign keratosis": {
        "risk": "Benign",
        "description": "Non-cancerous growth (e.g., seborrheic keratosis). Cosmetic concern only.",
        "recommendation": "No medical treatment needed. Can be removed for cosmetic reasons.",
    },
    "squamous cell carcinoma": {
        "risk": "Malignant",
        "description": "Second most common skin cancer. Can metastasize if untreated.",
        "recommendation": "Seek prompt dermatological evaluation for biopsy and treatment.",
    },
    "vascular lesion": {
        "risk": "Benign",
        "description": "Blood vessel abnormality (e.g., hemangioma). Usually harmless.",
        "recommendation": "Monitor for changes. Treatment usually not required.",
    },
}

# ── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))


# ── Security Headers ───────────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    """Add security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# ── Rate Limiting ──────────────────────────────────────────────────────────
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def rate_limit(f):
    """Simple in-memory rate limiter decorator."""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        # Prune old entries
        _rate_limit_store[ip] = [
            t for t in _rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW
        ]
        if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
            return jsonify({
                "error": "Too many requests. Please wait a minute and try again."
            }), 429
        _rate_limit_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated


# ── Error Handler ──────────────────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB."}), 413


# ── Model (loaded once at startup) ──────────────────────────────────────────
print(f"⏳ Loading model: {MODEL_ID} …")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
print("✅ Model loaded successfully.\n")


# ── Helpers ─────────────────────────────────────────────────────────────────
def _match_class_info(label: str) -> dict | None:
    """Fuzzy-match a model label to CLASS_INFO."""
    label_lower = label.lower()
    for key, info in CLASS_INFO.items():
        if key in label_lower or label_lower in key:
            return info
    return None


def _validate_image_content(filepath: Path) -> bool:
    """Verify that the file is actually a valid image (not just by extension)."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def run_prediction(image: Image.Image, top_k: int = 5) -> list[dict]:
    """Run inference and return structured results."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = model.config.id2label[idx.item()]
        confidence = round(prob.item() * 100, 2)
        info = _match_class_info(label)

        results.append({
            "label": label,
            "confidence": confidence,
            "risk": info["risk"] if info else "Unknown",
            "description": info["description"] if info else "",
            "recommendation": info["recommendation"] if info else "",
        })

    return results


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/scan")
def scan():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
@rate_limit
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{ext}'. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Validate MIME type
    if file.content_type and file.content_type not in ALLOWED_MIMETYPES:
        return jsonify({"error": "Invalid file type."}), 400

    # Save temporarily with a safe random filename
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename

    try:
        file.save(filepath)

        # Validate that the file is actually a valid image
        if not _validate_image_content(filepath):
            return jsonify({"error": "The uploaded file is not a valid image."}), 400

        image = Image.open(filepath).convert("RGB")
        results = run_prediction(image, top_k=5)
    except Exception:
        # Never expose internal error details to the client
        return jsonify({"error": "An error occurred while processing the image. Please try again."}), 500
    finally:
        # Always clean up
        if filepath.exists():
            filepath.unlink()

    return jsonify({
        "success": True,
        "predictions": results,
    })


# ── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(debug=False, host="0.0.0.0", port=port)

