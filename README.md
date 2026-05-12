# NL / EU Licence Plate Detector — Prototype

## Stack

| Layer | Component | Notes |
|-------|-----------|-------|
| Detection | YOLOv8n — `keremberke/yolov8n-license-plate-detection` | Pretrained on EU plates, ~6 MB |
| OCR | EasyOCR (`nl` + `en`) | No cloud, runs fully local |
| Backend | Flask (Python) | REST API on port 5050 |
| Frontend | React-style HTML widget | Drag-and-drop upload + folder batch scan |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the backend (models download automatically on first run, ~300 MB)
python app.py

# 3. Open the UI
# Paste the widget HTML into any browser, or serve it with:
python -m http.server 8080
```

## API endpoints

### `POST /detect`
Detect plates in a single image.

```json
// Option A — base64 upload
{ "image_b64": "<base64 string>", "conf_threshold": 0.30 }

// Option B — local file path
{ "file_path": "/data/cases/IMG_001.jpg", "conf_threshold": 0.30 }
```

Response:
```json
{
  "count": 2,
  "detections": [
    {
      "plate_text": "AB123CD",
      "detection_conf": 0.92,
      "ocr_conf": 0.87,
      "bbox": [120, 400, 380, 460],
      "crop_b64": "<base64 cropped plate>"
    }
  ],
  "annotated_b64": "<base64 annotated full image>"
}
```

### `POST /detect_folder`
Batch-process all images in a local folder.

```json
{ "folder_path": "/data/cases/2024-001/", "conf_threshold": 0.30 }
```

## Confidence thresholds — guidelines

| Setting | When to use |
|---------|-------------|
| 20–30% | Distant / partial plates, bad lighting — catches more, more false positives |
| 30–50% | **Default** — good balance for incident photography |
| 50–70% | Close-up, well-lit images — high precision only |

## Tips for police incident photos

- **Use folder path mode** for large case sets — avoids browser memory limits
- Set confidence to **25–35%** for varying distances; filter by OCR confidence in post
- The YOLOv8 model handles Dutch yellow plates, EU white plates, motorcycle plates
- EasyOCR `nl` language profile includes Dutch character patterns
- For very blurry crops, the OCR conf will be low — treat anything below 50% as uncertain

## Extending

- **Export to CSV**: add a `/export_csv` endpoint that writes `file,plate,det_conf,ocr_conf` rows
- **Filtering**: pass `min_ocr_conf` to `/detect_folder` to drop low-confidence reads
- **GPU acceleration**: set `gpu=True` in `easyocr.Reader(...)` if CUDA is available — 5–10× faster OCR
- **Fine-tuning**: collect false-positive crops and fine-tune the YOLOv8 model on your case data
