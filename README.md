# DermaScan AI - Skin Cancer Detection

DermaScan AI is a skin lesion classification project that analyzes dermoscopic images and returns the most likely class, confidence score, and basic clinical context for each prediction.

The project includes two ways to run the app:

- `streamlit_app.py` for a simple Streamlit interface
- `app.py` for the Flask web app with HTML templates, static assets, and an `/api/predict` endpoint

It uses a fine-tuned Vision Transformer model from Hugging Face and the HAM10000 dermatology dataset.

> Medical disclaimer: this project is for educational and screening purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist.

## Features

- 8-class skin lesion classification
- Confidence scores for the top predictions
- Risk labels with short clinical descriptions and recommendations
- Image validation and temporary upload cleanup in the Flask app
- Rate limiting and basic security headers on the prediction endpoint
- Streamlit and Flask entry points for different deployment styles

## Requirements

- Python 3.10 or newer
- pip
- Internet access on first run so the model can download from Hugging Face

## Installation

```bash
git clone https://github.com/ashutoshswamy/skin-cancer-detection.git
cd skin-cancer-detection
pip install -r requirements.txt
```

## Run the apps

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Streamlit usually opens at `http://localhost:8501`.

### Flask web app

```bash
python app.py
```

The Flask app runs on `http://localhost:3000` by default.

## API

The Flask app exposes a prediction endpoint.

`POST /api/predict`

Request format:

- `multipart/form-data`
- field name: `image`

Example response:

```json
{
  "success": true,
  "predictions": [
    {
      "label": "melanoma",
      "confidence": 87.34,
      "risk": "Malignant",
      "description": "Most dangerous skin cancer. Early detection is critical for survival.",
      "recommendation": "Urgent: Consult a dermatologist or oncologist immediately."
    }
  ]
}
```

Limits and validation:

- Maximum file size: 10 MB
- Rate limit: 20 requests per minute per IP
- Allowed image types: JPG, PNG, BMP, TIFF, WebP

## Project Structure

```text
skin cancer detection/
├── app.py                 # Flask app and API
├── streamlit_app.py       # Streamlit UI
├── skin_cancer_detector.py # Original CLI script
├── requirements.txt       # Python dependencies
├── templates/             # Flask HTML templates
├── static/                # CSS and JavaScript assets
└── uploads/               # Temporary upload directory
```

## Model Classes

- Actinic keratosis - Pre-cancerous
- Basal cell carcinoma - Malignant
- Dermatofibroma - Benign
- Melanoma - Malignant
- Nevus - Benign
- Pigmented benign keratosis - Benign
- Squamous cell carcinoma - Malignant
- Vascular lesion - Benign

## Tech Stack

- Flask
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pillow
- HTML, CSS, and JavaScript

## License

This project is intended for educational use. The model and dataset are subject to their respective licenses.
