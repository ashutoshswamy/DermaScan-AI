"""
Skin Cancer Detection Script
=============================
Classifies skin lesion images using a pre-trained deep learning model
fine-tuned on the HAM10000 dermatological dataset.

Usage:
    python skin_cancer_detector.py <image_path>
    python skin_cancer_detector.py image.jpg --top_k 3

Requirements:
    pip install torch torchvision transformers Pillow

Disclaimer:
    This tool is for educational/screening purposes ONLY.
    It is NOT a substitute for professional medical diagnosis.
    Always consult a dermatologist for clinical evaluation.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor


MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"

# HAM10000 class labels and their clinical descriptions
CLASS_INFO = {
    "actinic keratosis": {
        "risk": "Pre-cancerous",
        "description": "Rough, scaly patch caused by sun damage. Can progress to squamous cell carcinoma.",
    },
    "basal cell carcinoma": {
        "risk": "Malignant",
        "description": "Most common skin cancer. Slow-growing, rarely metastasizes, but requires treatment.",
    },
    "dermatofibroma": {
        "risk": "Benign",
        "description": "Harmless, firm nodule. No treatment necessary unless symptomatic.",
    },
    "melanoma": {
        "risk": "Malignant",
        "description": "Most dangerous skin cancer. Early detection is critical for survival.",
    },
    "nevus": {
        "risk": "Benign",
        "description": "Common mole. Usually harmless but should be monitored for changes.",
    },
    "pigmented benign keratosis": {
        "risk": "Benign",
        "description": "Non-cancerous growth (e.g., seborrheic keratosis). Cosmetic concern only.",
    },
    "squamous cell carcinoma": {
        "risk": "Malignant",
        "description": "Second most common skin cancer. Can metastasize if untreated.",
    },
    "vascular lesion": {
        "risk": "Benign",
        "description": "Blood vessel abnormality (e.g., hemangioma). Usually harmless.",
    },
}


def load_model():
    """Download and load the pre-trained skin cancer classification model."""
    print(f"Loading model: {MODEL_ID} ...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Make sure you have an internet connection for the first run (model download).")
        sys.exit(1)

    model.eval()
    print("Model loaded successfully.\n")
    return processor, model


def load_image(image_path: str) -> Image.Image:
    """Load and validate the input image."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if path.suffix.lower() not in valid_extensions:
        print(f"Error: Unsupported image format '{path.suffix}'. Use: {', '.join(valid_extensions)}")
        sys.exit(1)

    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    return image


def predict(processor, model, image: Image.Image, top_k: int = 3):
    """Run inference and return top-k predictions with confidence scores."""
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = model.config.id2label[idx.item()]
        results.append({"label": label, "confidence": prob.item()})

    return results


def display_results(results: list, image_path: str):
    """Pretty-print the classification results with clinical context."""
    print("=" * 60)
    print("  SKIN LESION CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Image: {image_path}")
    print("-" * 60)

    for i, result in enumerate(results):
        label = result["label"]
        confidence = result["confidence"]
        bar = "█" * int(confidence * 30) + "░" * (30 - int(confidence * 30))

        print(f"\n  #{i + 1}  {label.upper()}")
        print(f"      Confidence: [{bar}] {confidence:.1%}")

        # Look up clinical info (fuzzy match on label)
        info = None
        for key, val in CLASS_INFO.items():
            if key in label.lower() or label.lower() in key:
                info = val
                break

        if info:
            risk = info["risk"]
            risk_marker = {"Benign": "🟢", "Pre-cancerous": "🟡", "Malignant": "🔴"}.get(risk, "⚪")
            print(f"      Risk level: {risk_marker} {risk}")
            print(f"      Info: {info['description']}")

    print("\n" + "=" * 60)

    top_label = results[0]["label"].lower()
    is_concerning = any(
        keyword in top_label
        for keyword in ["melanoma", "carcinoma", "actinic"]
    )
    if is_concerning:
        print("  ⚠️  The top prediction indicates a potentially")
        print("     CANCEROUS or PRE-CANCEROUS lesion.")
        print("     Please consult a dermatologist promptly.")
    else:
        print("  ℹ️  The top prediction indicates a likely BENIGN lesion.")
        print("     Continue monitoring for changes (ABCDE rule).")

    print("\n  DISCLAIMER: This is NOT a medical diagnosis.")
    print("  Always consult a qualified healthcare professional.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Skin cancer detection from dermoscopic images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python skin_cancer_detector.py lesion.jpg --top_k 5",
    )
    parser.add_argument("image", help="Path to the skin lesion image.")
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    args = parser.parse_args()

    image = load_image(args.image)
    processor, model = load_model()
    results = predict(processor, model, image, top_k=args.top_k)
    display_results(results, args.image)


if __name__ == "__main__":
    main()
