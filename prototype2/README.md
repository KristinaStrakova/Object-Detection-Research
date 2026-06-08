# Plate Scanner — Local Prototype

A local web app that runs your full licence-plate detection pipeline:
**Car detection (YOLOv11) → Plate detection (fine-tuned YOLO) → OCR (EasyOCR)**

---

## 📁 Folder structure

```
plate-detector/
├── app.py                 ← Flask backend (the pipeline)
├── requirements.txt
├── templates/
│   └── index.html         ← Frontend UI
└── models/
    ├── yolo11n.pt                       ← YOLOv11 vehicle detector
    └── license-plate-finetune-v1n.onnx  ← Fine-tuned plate detector
```

---

## ⚙️ Setup

### 1. Place your model files

Copy your two model files into the `models/` folder:
- `models/yolo11n.pt`
- `models/license-plate-finetune-v1n.onnx`

> These are the same model paths used in your notebook.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR will download its own OCR model (~100 MB) on first run.  
> This only happens once and is cached in `~/.EasyOCR/`.

---

## 🚀 Run

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## 🖥️ How to use

1. **Drop or select images** — supports JPG, PNG, WEBP, BMP. You can select 25–50 at once.
2. Click **Run Pipeline**.
3. Watch the **Detection Log** tab fill up:
   - Each unique OCR result appears as its own row.
   - Duplicates are **not** repeated — instead the count `×N` updates.
   - Confidence score shown as a colour-coded bar.
   - Country prefix (e.g. `NL`, `DE`, `BE`) extracted automatically if present.
4. The **Confirmed Plates** tab shows only plates seen **≥ 3 times** — those are your actual cars.
   - Sorted by frequency; most-seen plate shown first with 🏆.

---

## 🔧 Tweaking thresholds

Open `app.py` and adjust these constants at the top:

| Constant       | Default | What it controls                        |
|----------------|---------|-----------------------------------------|
| `CONF_CAR`     | 0.40    | Min confidence to accept a car detection |
| `CONF_PLATE`   | 0.30    | Min confidence to accept a plate crop    |
| `MIN_OCR_CONF` | 0.10    | Min EasyOCR confidence to log text       |
| (confirmed)    | ≥ 3     | Count threshold for "confirmed car"      |

The count threshold for confirmed plates is in `app.py` line:
```python
confirmed = [l for l in logs if l["count"] >= 3]
```
Change `3` to whatever suits your dataset.

---

## 🌍 Country detection

The app recognises European-style plates like `NL-AB1234`, `DE AB 1234`, `BE·123·ABC` etc.  
It tries to split a 1–3 letter prefix followed by digits as the country code.  
If the OCR reads the plate as a single string without a separator, the country field will show `—`.
