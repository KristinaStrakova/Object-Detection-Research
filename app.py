"""
Dutch / EU Licence Plate Detector
Backend: Flask + YOLOv8 (keremberke/yolov8n-license-plate-detection) + EasyOCR
"""

import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── lazy-load heavy models once ──────────────────────────────────────────────
_yolo_model = None
_ocr_reader = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("[init] Loading YOLO11 model …", flush=True)

        # Ultralytics will automatically download the model from Hugging Face
        _yolo_model = YOLO("morsetechlab/yolov11-license-plate-detection")

        print("[init] YOLO ready.", flush=True)
    return _yolo_model

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[init] Loading EasyOCR (nl + en) …", flush=True)
        # Dutch + English covers NL plates perfectly; adds minimal overhead
        _ocr_reader = easyocr.Reader(["nl", "en"], gpu=False, verbose=False)
        print("[init] OCR ready.", flush=True)
    return _ocr_reader

# ── helpers ───────────────────────────────────────────────────────────────────

def cv2_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()

def clean_plate_text(text: str) -> str:
    """Uppercase, strip spaces/dashes, keep only alphanumerics."""
    import re
    return re.sub(r"[^A-Z0-9]", "", text.upper())

PADDING = 8  # px to expand the crop on each side (helps OCR)

def process_image(img_bgr: np.ndarray, conf_threshold: float = 0.30):
    model = get_yolo()
    reader = get_ocr()

    h, w = img_bgr.shape[:2]
    results = model(img_bgr, conf=conf_threshold, verbose=False)[0]

    detections = []
    annotated = img_bgr.copy()

    for box in results.boxes:
        det_conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Expand crop slightly
        cx1 = max(0, x1 - PADDING)
        cy1 = max(0, y1 - PADDING)
        cx2 = min(w, x2 + PADDING)
        cy2 = min(h, y2 + PADDING)

        crop = img_bgr[cy1:cy2, cx1:cx2]

        # Pre-process crop for OCR: upscale + greyscale
        scale = max(1, 200 // max(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        ocr_results = reader.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

        plate_text = ""
        ocr_conf = 0.0
        if ocr_results:
            # Pick the highest-confidence OCR result
            best = max(ocr_results, key=lambda r: r[2])
            plate_text = clean_plate_text(best[1])
            ocr_conf = float(best[2])

        # Draw on annotated image
        color = (0, 200, 60)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{plate_text}  det:{det_conf:.0%}  ocr:{ocr_conf:.0%}"
        cv2.rectangle(annotated, (x1, y1 - 22), (x2, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "plate_text":    plate_text,
            "detection_conf": round(det_conf, 4),
            "ocr_conf":       round(ocr_conf, 4),
            "bbox":           [x1, y1, x2, y2],
            "crop_b64":       cv2_to_b64(img_bgr[cy1:cy2, cx1:cx2]),
        })

    return detections, cv2_to_b64(annotated)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    conf = float(data.get("conf_threshold", 0.30))

    img_bgr = None

    # ── source: base64 upload ─────────────────────────────────────────────
    if "image_b64" in data:
        raw = base64.b64decode(data["image_b64"])
        arr = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── source: local file path ───────────────────────────────────────────
    elif "file_path" in data:
        path = Path(data["file_path"])
        if not path.exists():
            return jsonify({"error": f"File not found: {path}"}), 400
        img_bgr = cv2.imread(str(path))

    else:
        return jsonify({"error": "Provide 'image_b64' or 'file_path'"}), 400

    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    detections, annotated_b64 = process_image(img_bgr, conf)

    return jsonify({
        "detections":    detections,
        "annotated_b64": annotated_b64,
        "count":         len(detections),
    })

@app.route("/detect_folder", methods=["POST"])
def detect_folder():
    """Batch-process a local folder. Returns summary per file."""
    data = request.get_json(force=True)
    folder = Path(data.get("folder_path", ""))
    conf = float(data.get("conf_threshold", 0.30))
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not folder.is_dir():
        return jsonify({"error": f"Not a directory: {folder}"}), 400

    image_files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]
    results = []

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            results.append({"file": img_path.name, "error": "unreadable"})
            continue
        detections, annotated_b64 = process_image(img_bgr, conf)
        results.append({
            "file": img_path.name,
            "detections": detections,
            "annotated_b64": annotated_b64,
            "count": len(detections),
        })

    return jsonify({"files": results, "total_files": len(image_files)})

if __name__ == "__main__":
    print("Pre-loading models …")
    get_yolo()
    get_ocr()
    print("Starting server on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
