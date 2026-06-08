"""
Licence Plate Detection & OCR — Local Flask App
Replicates the notebook pipeline: Car detection → Plate detection → EasyOCR
"""

import os, re, time, base64
from pathlib import Path
from collections import defaultdict
from flask import Flask, request, jsonify, render_template

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# ── Config ────────────────────────────────────────────────────────────────────
CONF_CAR       = 0.40
CONF_PLATE     = 0.30
CAR_CLASS_IDS  = {2, 5, 7}          # COCO: car, bus, truck
CAR_MODEL_NAME = "models/yolo11n.pt"
PLATE_MODEL    = "models/license-plate-finetune-v1n.onnx"
MIN_OCR_CONF   = 0.10
MIN_DISPLAY_W  = 400
ALLOWLIST      = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789- "
OCR_PARAMS     = dict(
    text_threshold=0.6,
    min_size=10,
    low_text=0.5,
    mag_ratio=1.8,
    link_threshold=0.5,
    width_ths=0.7,
    decoder="beamsearch",
    beamWidth=5,
    allowlist=ALLOWLIST,
    paragraph=False,
)

# ── Country code lookup ───────────────────────────────────────────────────────
# Official vehicle-registration country codes that appear on European plates
# (EU blue-band codes + wider EEA / neighbouring-state codes).
PLATE_COUNTRY_CODES: dict[str, str] = {
    # EU member states
    "A":   "Austria",
    "B":   "Belgium",
    "BG":  "Bulgaria",
    "CY":  "Cyprus",
    "CZ":  "Czech Republic",
    "D":   "Germany",
    "DK":  "Denmark",
    "E":   "Spain",
    "EST": "Estonia",
    "F":   "France",
    "FIN": "Finland",
    "GR":  "Greece",
    "H":   "Hungary",
    "HR":  "Croatia",
    "I":   "Italy",
    "IRL": "Ireland",
    "L":   "Luxembourg",
    "LT":  "Lithuania",
    "LV":  "Latvia",
    "M":   "Malta",
    "NL":  "Netherlands",
    "P":   "Portugal",
    "PL":  "Poland",
    "RO":  "Romania",
    "S":   "Sweden",
    "SK":  "Slovakia",
    "SLO": "Slovenia",
    # EEA / candidate / neighbouring states
    "AL":  "Albania",
    "AND": "Andorra",
    "ARM": "Armenia",
    "AZ":  "Azerbaijan",
    "BA":  "Bosnia & Herzegovina",
    "BY":  "Belarus",
    "CH":  "Switzerland",
    "FL":  "Liechtenstein",
    "GB":  "United Kingdom",
    "GE":  "Georgia",
    "IS":  "Iceland",
    "KS":  "Kosovo",
    "MD":  "Moldova",
    "ME":  "Montenegro",
    "MK":  "North Macedonia",
    "N":   "Norway",
    "RS":  "Serbia",
    "RUS": "Russia",
    "TR":  "Turkey",
    "UA":  "Ukraine",
    "UK":  "United Kingdom",
}

# Pattern: optional country prefix (1-3 letters) then separator then remainder
_COUNTRY_RE = re.compile(r"^([A-Z]{1,3})[-\s](.+)$")

app = Flask(__name__)

# ── Lazy model loading ─────────────────────────────────────────────────────────
_car_model   = None
_plate_model = None
_ocr_reader  = None

def get_models():
    global _car_model, _plate_model, _ocr_reader
    if _car_model is None:
        _car_model   = YOLO(CAR_MODEL_NAME)
    if _plate_model is None:
        _plate_model = YOLO(PLATE_MODEL)
    if _ocr_reader is None:
        _ocr_reader  = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _car_model, _plate_model, _ocr_reader


# ── Helpers ────────────────────────────────────────────────────────────────────
def to_gray_bgr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def normalise_plate(text: str) -> str:
    """Upper-case, strip noise, collapse spaces."""
    t = text.upper().strip()
    t = re.sub(r"[^A-Z0-9 \-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_country(text: str):
    """
    Returns (country_code, country_name, plate_number).
    country_code and country_name are None when no country is detected.

    Three strategies in order:
      1. Explicit separator  — 'NL AB1234' or 'NL-AB1234'
      2. No separator        — 'NLAB1234' (OCR often omits the gap);
         only codes of 2+ chars to avoid false positives from A/B/D/…
      3. Exact match         — the whole OCR result is just the country code
         (EasyOCR sometimes reads the country indicator as its own region)
    """
    m = _COUNTRY_RE.match(text)
    if m:
        prefix = m.group(1)
        if prefix in PLATE_COUNTRY_CODES:
            return prefix, PLATE_COUNTRY_CODES[prefix], m.group(2).strip()

    for code in sorted((c for c in PLATE_COUNTRY_CODES if len(c) >= 2),
                       key=len, reverse=True):
        if text.startswith(code):
            remainder = text[len(code):]
            if len(remainder) >= 4 and remainder[0].isalnum():
                return code, PLATE_COUNTRY_CODES[code], remainder.strip()

    if text in PLATE_COUNTRY_CODES:
        return text, PLATE_COUNTRY_CODES[text], ""

    return None, None, text


def encode_img(img: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG data-URL."""
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def merge_crop_detections(dets: list) -> list:
    """
    Within a single plate-crop's OCR results, pair a country-indicator
    detection with a plate-number detection and merge them into one entry.

    Merging fires when:
      • one detection is country-only  (has_country=True, plate_num == "")
      • another has a valid Dutch sidecode (dutch_sidecode is not None,
        has_country=False)

    The merged entry text is formatted as  "NL | AB-12-34"  and represents
    one complete, connected pattern counted as a single detection.
    """
    if len(dets) <= 1:
        return dets

    country_det = None
    plate_det   = None
    for d in dets:
        if d["has_country"] and d["plate_num"] == "":
            if country_det is None or d["ocr_conf"] > country_det["ocr_conf"]:
                country_det = d
        elif not d["has_country"] and d["dutch_sidecode"] is not None:
            if plate_det is None or d["ocr_conf"] > plate_det["ocr_conf"]:
                plate_det = d

    if country_det is None or plate_det is None:
        return dets

    code        = country_det["country_code"]
    merged_text = f"{code} | {plate_det['plate_num']}"
    merged = {
        "text":             merged_text,
        "country_code":     code,
        "country":          country_det["country"],
        "has_country":      True,
        "plate_num":        plate_det["plate_num"],
        "ocr_conf":         round((country_det["ocr_conf"] + plate_det["ocr_conf"]) / 2, 4),
        "plate_conf":       plate_det["plate_conf"],
        "car_conf":         plate_det["car_conf"],
        "dutch_sidecode":   plate_det["dutch_sidecode"],
        "sidecode_pattern": plate_det["sidecode_pattern"],
        "image_b64":        plate_det["image_b64"],
    }
    remaining = [d for d in dets if d is not country_det and d is not plate_det]
    remaining.append(merged)
    return remaining


# ── Dutch licence plate sidecodes ─────────────────────────────────────────────
# X = letter (A-Z), 9 = digit (0-9).
# Each entry: (sidecode_number, display_pattern, regex_with_dashes, regex_no_sep)
_DUTCH_SIDECODES = [
    (1,  "XX-99-99", re.compile(r"^[A-Z]{2}-\d{2}-\d{2}$"),    re.compile(r"^[A-Z]{2}\d{4}$")),
    (2,  "99-99-XX", re.compile(r"^\d{2}-\d{2}-[A-Z]{2}$"),    re.compile(r"^\d{4}[A-Z]{2}$")),
    (3,  "99-XX-99", re.compile(r"^\d{2}-[A-Z]{2}-\d{2}$"),    re.compile(r"^\d{2}[A-Z]{2}\d{2}$")),
    (4,  "XX-99-XX", re.compile(r"^[A-Z]{2}-\d{2}-[A-Z]{2}$"), re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}$")),
    (5,  "XX-XX-99", re.compile(r"^[A-Z]{2}-[A-Z]{2}-\d{2}$"), re.compile(r"^[A-Z]{4}\d{2}$")),
    (6,  "99-XX-XX", re.compile(r"^\d{2}-[A-Z]{2}-[A-Z]{2}$"), re.compile(r"^\d{2}[A-Z]{4}$")),
    (7,  "99-XXX-9", re.compile(r"^\d{2}-[A-Z]{3}-\d$"),       re.compile(r"^\d{2}[A-Z]{3}\d$")),
    (8,  "9-XXX-99", re.compile(r"^\d-[A-Z]{3}-\d{2}$"),       re.compile(r"^\d[A-Z]{3}\d{2}$")),
    (9,  "XX-999-X", re.compile(r"^[A-Z]{2}-\d{3}-[A-Z]$"),    re.compile(r"^[A-Z]{2}\d{3}[A-Z]$")),
    (10, "X-999-XX", re.compile(r"^[A-Z]-\d{3}-[A-Z]{2}$"),    re.compile(r"^[A-Z]\d{3}[A-Z]{2}$")),
    (11, "XXX-99-X", re.compile(r"^[A-Z]{3}-\d{2}-[A-Z]$"),    re.compile(r"^[A-Z]{3}\d{2}[A-Z]$")),
]


def match_dutch_sidecode(text: str):
    """
    Check whether text matches a Dutch licence plate sidecode pattern.
    Returns (sidecode_number, pattern_string) or (None, None).
    Tries both with dashes and without separators (OCR may omit dashes).
    """
    t = text.upper().strip()
    for sc, pat, re_dash, _ in _DUTCH_SIDECODES:
        if re_dash.match(t):
            return sc, pat
    t_plain = re.sub(r"[-\s]", "", t)
    for sc, pat, _, re_plain in _DUTCH_SIDECODES:
        if re_plain.match(t_plain):
            return sc, pat
    return None, None


def process_image(img_bytes: bytes):
    """
    Run the full pipeline on one image.
    Returns list of dicts:
      { text, normalised, country, plate_conf, car_conf }
    """
    car_model, plate_model, ocr_reader = get_models()

    # Decode
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    detections = []

    # Stage 1 — Vehicle detection
    car_out = car_model(img, conf=CONF_CAR, verbose=False)[0]
    for cb in car_out.boxes:
        if int(cb.cls[0]) not in CAR_CLASS_IDS:
            continue
        cx1, cy1, cx2, cy2 = map(int, cb.xyxy[0].tolist())
        car_crop = img[cy1:cy2, cx1:cx2]
        if car_crop.size == 0:
            continue
        car_conf = float(cb.conf[0])

        # Stage 2 — Plate detection inside vehicle crop
        plate_out = plate_model(car_crop, conf=CONF_PLATE, verbose=False)[0]
        for pb in plate_out.boxes:
            px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
            plate_crop = car_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            plate_conf = float(pb.conf[0])

            # Upscale small crops
            h, w = plate_crop.shape[:2]
            if w < MIN_DISPLAY_W:
                scale = MIN_DISPLAY_W / w
                plate_crop = cv2.resize(
                    plate_crop,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            # Stage 3 — OCR
            ocr_res   = ocr_reader.readtext(to_gray_bgr(plate_crop), **OCR_PARAMS)
            plate_b64 = encode_img(plate_crop)
            crop_dets = []
            for (_, text, conf) in ocr_res:
                if conf < MIN_OCR_CONF or not text.strip():
                    continue
                norm = normalise_plate(text)
                if not norm:
                    continue
                country_code, country, plate_num = parse_country(norm)
                sc, sc_pattern = match_dutch_sidecode(plate_num)
                crop_dets.append({
                    "text":             norm,
                    "country_code":     country_code,
                    "country":          country,
                    "has_country":      country is not None,
                    "plate_num":        plate_num,
                    "ocr_conf":         round(conf, 4),
                    "plate_conf":       round(plate_conf, 4),
                    "car_conf":         round(car_conf, 4),
                    "dutch_sidecode":   sc,
                    "sidecode_pattern": sc_pattern,
                    "image_b64":        plate_b64,
                })
            detections.extend(merge_crop_detections(crop_dets))

    return detections


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    # Aggregate across all images: text → accumulated stats
    agg = defaultdict(lambda: {
        "count": 0, "best_conf": 0.0, "country_code": None, "country": None, "plate_num": "",
        "dutch_sidecode": None, "sidecode_pattern": None,
    })
    detailed = []   # per-file results with plate crop images for Detailed Log tab

    for f in files:
        img_bytes = f.read()
        filename  = f.filename or "unknown"
        try:
            dets = process_image(img_bytes)
        except Exception as e:
            detailed.append({"filename": filename, "plates": [], "error": str(e)})
            continue

        if dets:
            detailed.append({"filename": filename, "plates": dets})

        for d in dets:
            key   = d["text"]
            entry = agg[key]
            entry["count"] += 1
            if d["ocr_conf"] > entry["best_conf"]:
                entry["best_conf"] = d["ocr_conf"]
            entry["country_code"] = d["country_code"]
            entry["country"]      = d["country"]
            entry["plate_num"]    = d["plate_num"]
            if entry["dutch_sidecode"] is None:
                entry["dutch_sidecode"]   = d["dutch_sidecode"]
                entry["sidecode_pattern"] = d["sidecode_pattern"]

    logs = []
    for key, e in agg.items():
        logs.append({
            "text":             key,
            "count":            e["count"],
            "conf":             round(e["best_conf"], 4),
            "country_code":     e["country_code"],
            "country":          e["country"],
            "plate_num":        e["plate_num"],
            "has_country":      e["country"] is not None,
            "dutch_sidecode":   e["dutch_sidecode"],
            "sidecode_pattern": e["sidecode_pattern"],
        })

    logs.sort(key=lambda x: -x["count"])

    # Confirmed: seen ≥3×, no country prefix detected, matches a Dutch sidecode
    confirmed = [
        l for l in logs
        if l["count"] >= 3 and not l["has_country"] and l["dutch_sidecode"] is not None
    ]

    return jsonify({"logs": logs, "confirmed": confirmed, "detailed": detailed})


if __name__ == "__main__":
    print("=== Licence Plate Detector ===")
    print("Pre-loading models …")
    get_models()
    print("Models ready. Starting server on http://localhost:5000")
    app.run(debug=False, port=5000)
