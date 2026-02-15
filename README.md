# 🔬 DermaScan AI — Skin Cancer Detection

AI-powered skin lesion classification web application. Upload a dermoscopic image and get instant analysis across **8 lesion types** with risk assessment, confidence scores, and clinical guidance.

Built with **Flask**, **PyTorch**, and a fine-tuned **Vision Transformer** trained on the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dermatological dataset.

> ⚠️ **Medical Disclaimer:** This tool is for **educational and screening purposes only**. It is **NOT** a substitute for professional medical diagnosis. Always consult a qualified dermatologist.

---

## ✨ Features

- **8-Class Classification** — Identifies actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, and vascular lesions
- **Risk Assessment** — Each prediction includes a risk level (🟢 Benign / 🟡 Pre-cancerous / 🔴 Malignant) with clinical context
- **Drag & Drop Upload** — Intuitive image upload with preview and file validation
- **Instant Results** — Classification in under 3 seconds with animated confidence bars
- **Modern Dark UI** — Premium glassmorphism design with warm color palette
- **Security Hardened** — Rate limiting, MIME validation, security headers, safe error handling

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ashutoshswamy/skin-cancer-detection.git
cd skin-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open **http://localhost:3000** in your browser.

> 📝 The model (~85 MB) downloads automatically on first run. An internet connection is required for the initial setup.

---

## 📁 Project Structure

```
skin-cancer-detection/
├── app.py                    # Flask backend (routes, API, model inference)
├── skin_cancer_detector.py   # Original CLI script
├── requirements.txt          # Python dependencies
├── templates/
│   ├── home.html             # Landing page
│   └── index.html            # Scan / analysis page
├── static/
│   ├── css/style.css         # Design system (dark mode, warm palette)
│   └── js/app.js             # Client-side logic (upload, results)
└── uploads/                  # Temporary upload dir (auto-cleaned)
```

---

## 🔌 API

### `POST /api/predict`

Upload an image for classification.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "label": "melanoma",
      "confidence": 87.34,
      "risk": "Malignant",
      "description": "Most dangerous skin cancer...",
      "recommendation": "Urgent: Consult a dermatologist..."
    }
  ]
}
```

**Limits:** Max file size 10 MB · 20 requests/min per IP · Accepts JPG, PNG, BMP, TIFF, WebP

---

## 🔒 Security

| Feature | Description |
|---------|-------------|
| **Security Headers** | X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy |
| **Rate Limiting** | 20 requests per minute per IP on the prediction endpoint |
| **MIME Validation** | Checks both file extension and content type |
| **Image Verification** | Validates uploaded files are actual images using PIL |
| **Safe Errors** | Internal exceptions are never exposed to the client |
| **Upload Cleanup** | Temporary files are always deleted after processing |
| **Secret Key** | Auto-generated or set via `SECRET_KEY` environment variable |
| **Localhost Binding** | Server binds to `127.0.0.1` by default (not exposed to network) |

---

## 🎨 Classification Types

| Condition | Risk Level |
|-----------|-----------|
| Actinic Keratosis | 🟡 Pre-cancerous |
| Basal Cell Carcinoma | 🔴 Malignant |
| Dermatofibroma | 🟢 Benign |
| Melanoma | 🔴 Malignant |
| Nevus (Mole) | 🟢 Benign |
| Pigmented Benign Keratosis | 🟢 Benign |
| Squamous Cell Carcinoma | 🔴 Malignant |
| Vascular Lesion | 🟢 Benign |

---

## 🛠 Tech Stack

- **Backend:** Flask (Python)
- **ML Model:** HuggingFace Transformers (Vision Transformer)
- **Frontend:** Vanilla HTML, CSS, JavaScript
- **Font:** [Poppins](https://fonts.google.com/specimen/Poppins)
- **Dataset:** [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

---

## 👤 Author

**Ashutosh Swamy**

- 🌐 [ashutoshswamy.in](https://ashutoshswamy.in)
- 💼 [LinkedIn](https://linkedin.com/in/ashutoshswamy)
- 🐙 [GitHub](https://github.com/ashutoshswamy)

---

## 📄 License

This project is for educational purposes. The model and dataset are subject to their respective licenses.
