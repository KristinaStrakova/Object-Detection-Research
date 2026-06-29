"""Licence Plate Detection & OCR - Local Flask App.
Flask entrypoint that wires routes to the detection and lookup services.
"""

from collections import defaultdict

from flask import Flask, request, jsonify, render_template

from models_loader import get_models
from pipeline import process_image
from rdw_client import fetch_rdw_vehicle_info

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    # Aggregate across all images: text -> accumulated stats
    agg = defaultdict(
        lambda: {
            "count": 0,
            "best_conf": 0.0,
            "country_code": None,
            "country": None,
            "plate_num": "",
            "dutch_sidecode": None,
            "sidecode_pattern": None,
        }
    )
    detailed = []

    for f in files:
        img_bytes = f.read()
        filename = f.filename or "unknown"
        try:
            dets = process_image(img_bytes)
        except Exception as e:
            detailed.append({"filename": filename, "plates": [], "error": str(e)})
            continue

        if dets:
            detailed.append({"filename": filename, "plates": dets})

        for d in dets:
            key = d["text"]
            entry = agg[key]
            entry["count"] += 1
            if d["ocr_conf"] > entry["best_conf"]:
                entry["best_conf"] = d["ocr_conf"]
            entry["country_code"] = d["country_code"]
            entry["country"] = d["country"]
            entry["plate_num"] = d["plate_num"]
            if entry["dutch_sidecode"] is None:
                entry["dutch_sidecode"] = d["dutch_sidecode"]
                entry["sidecode_pattern"] = d["sidecode_pattern"]

    logs = []
    for key, e in agg.items():
        logs.append(
            {
                "text": key,
                "count": e["count"],
                "conf": round(e["best_conf"], 4),
                "country_code": e["country_code"],
                "country": e["country"],
                "plate_num": e["plate_num"],
                "has_country": e["country"] is not None,
                "dutch_sidecode": e["dutch_sidecode"],
                "sidecode_pattern": e["sidecode_pattern"],
            }
        )

    logs.sort(key=lambda x: -x["count"])

    # Confirmed: seen >=3x, matches Dutch sidecode, and no country prefix or NL.
    confirmed = [
        l
        for l in logs
        if l["count"] >= 3
        and l["dutch_sidecode"] is not None
        and (not l["has_country"] or l["country_code"] == "NL")
    ]

    return jsonify({"logs": logs, "confirmed": confirmed, "detailed": detailed})


@app.route("/validate_confirmed", methods=["POST"])
def validate_confirmed():
    """Validate all confirmed plates and fetch basic vehicle information."""
    payload = request.get_json(silent=True) or {}
    confirmed = payload.get("confirmed", [])
    if not isinstance(confirmed, list) or not confirmed:
        return jsonify({"error": "No confirmed plates provided"}), 400

    results = []
    for item in confirmed:
        if not isinstance(item, dict):
            continue
        plate_text = item.get("plate_num") or item.get("text") or ""
        info = fetch_rdw_vehicle_info(plate_text)
        results.append(
            {
                "text": item.get("text"),
                "plate_num": item.get("plate_num"),
                "count": item.get("count", 0),
                "conf": item.get("conf", 0.0),
                "country_code": item.get("country_code"),
                "country": item.get("country"),
                "lookup": info,
            }
        )

    results.sort(key=lambda x: (-int(x.get("count", 0)), -(float(x.get("conf", 0.0)))))
    return jsonify({"results": results})


if __name__ == "__main__":
    print("=== Licence Plate Detector ===")
    print("Pre-loading models ...")
    get_models()
    print("Models ready. Starting server on http://localhost:5000")
    app.run(debug=False, port=5000)
