"""
DermaScan AI — Streamlit App
============================
Streamlit UI for uploading a dermoscopic image and getting skin lesion
classification results (8 classes) with risk assessment and guidance.

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# HAM10000 class labels with clinical context
CLASS_INFO: dict[str, dict[str, str]] = {
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


def _match_class_info(label: str) -> dict[str, str] | None:
    label_lower = label.lower()
    for key, info in CLASS_INFO.items():
        if key in label_lower or label_lower in key:
            return info
    return None


def _load_image(uploaded_file: Any) -> Image.Image:
    name = (uploaded_file.name or "").lower()
    if "." not in name:
        raise ValueError("Unsupported file. Please upload an image.")
    ext = "." + name.rsplit(".", 1)[-1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Use JPG, PNG, BMP, TIFF, or WebP.")
    try:
        raw = uploaded_file.getvalue()
        image = Image.open(BytesIO(raw)).convert("RGB")
        return image
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid image file: {e}") from e


@st.cache_resource(show_spinner=False)
def _load_model() -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def _predict(image: Image.Image, top_k: int) -> list[dict[str, Any]]:
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    results: list[dict[str, Any]] = []
    for prob, idx in zip(top_probs, top_indices):
        label = model.config.id2label[idx.item()]
        confidence = round(prob.item() * 100, 2)
        info = _match_class_info(label)
        results.append(
            {
                "label": label,
                "confidence": confidence,
                "risk": info["risk"] if info else "Unknown",
                "description": info["description"] if info else "",
                "recommendation": info["recommendation"] if info else "",
            }
        )
    return results


def main() -> None:
    st.set_page_config(
        page_title="DermaScan AI — Skin Cancer Detection",
        page_icon="🔬",
        layout="wide",
    )

    st.title("DermaScan AI — Skin Cancer Detection")
    st.caption(
        "Upload a dermoscopic image and get an 8-class lesion prediction with risk assessment. "
        "For educational and screening purposes only."
    )

    with st.expander("Medical disclaimer", expanded=False):
        st.warning(
            "This tool is for educational/screening purposes only and is NOT a medical diagnosis. "
            "Always consult a qualified dermatologist."
        )

    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload an image (JPG/PNG/BMP/TIFF/WebP)",
            type=[e.lstrip(".") for e in sorted(ALLOWED_EXTENSIONS)],
        )
        top_k = st.slider("Top-K results", min_value=1, max_value=8, value=5, step=1)
        run = st.button("Analyze", type="primary", disabled=uploaded is None)

    if uploaded is None:
        with right:
            st.info("Upload an image to preview it and run analysis.")
        return

    try:
        image = _load_image(uploaded)
    except ValueError as e:
        with right:
            st.error(str(e))
        return

    with right:
        st.image(image, caption="Preview", use_container_width=True)

    if not run:
        return

    with st.spinner("Loading model and analyzing image…"):
        try:
            results = _predict(image, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
            return

    st.subheader("Results")
    if not results:
        st.info("No results returned.")
        return

    top = results[0]
    a, b, c = st.columns(3)
    a.metric("Top class", str(top["label"]).title())
    b.metric("Confidence", f"{top['confidence']}%")
    c.metric("Risk", top["risk"])
    if top.get("recommendation"):
        st.caption(f"Recommendation: {top['recommendation']}")

    for r in results:
        label = str(r["label"]).title()
        st.markdown(f"**{label}** — `{r['confidence']}%` • **{r['risk']}**")
        st.progress(min(max(r["confidence"] / 100.0, 0.0), 1.0))
        if r.get("description"):
            st.caption(r["description"])


if __name__ == "__main__":
    main()
