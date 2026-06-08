"""
Licence Plate Detection & OCR — Gradio UI
==========================================
Upload images → vehicle detection (YOLOv11) → plate detection (fine-tuned YOLO)
→ OCR (EasyOCR).  Results are shown one row per image with confidence badges.

Run:
    python plate_detection_ui.py
"""

import html as _html
from pathlib import Path
import base64

import cv2
import numpy as np
import easyocr
import gradio as gr
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent   # workspace root

# ── Configuration ──────────────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.30          # plate detection confidence
CAR_CONF         = 0.40          # car detection confidence
CAR_MODEL_NAME   = str(_ROOT / "models" / "yolo11n.pt")
PLATE_MODEL_NAME = str(_ROOT / "models" / "license-plate-finetune-v1n.onnx")
CAR_CLASS_IDS    = {2, 5, 7}     # COCO: car=2, bus=5, truck=7
MIN_DISPLAY_W    = 400           # minimum width for plate crop display
ALLOWLIST_CHARS  = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789- "
SUPPORTED_EXT    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OCR_PARAMS = dict(
    text_threshold=0.6,
    min_size=10,
    low_text=0.5,
    mag_ratio=1.8,
    link_threshold=0.5,
    width_ths=0.7,
    decoder="beamsearch",
    beamWidth=5,
    allowlist=ALLOWLIST_CHARS,
    paragraph=False,
)

# ── Model Loading ──────────────────────────────────────────────────────────────
print("Loading car model …")
car_model = YOLO(CAR_MODEL_NAME)

print("Loading plate model …")
plate_model = YOLO(PLATE_MODEL_NAME)

print("Loading EasyOCR (may download weights on first run) …")
ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

print("All models ready. Starting UI …")


# ── Image Helpers ──────────────────────────────────────────────────────────────

def bgr_to_b64(img: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG data-URL."""
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def to_gray_bgr(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale then back to 3-channel BGR (for OCR)."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def draw_ocr_annotations(img: np.ndarray, ocr_results) -> np.ndarray:
    """Draw OCR bounding polygons and labels onto a copy of img."""
    out = img.copy()
    for bbox, text, conf in ocr_results:
        pts = np.array([[int(x), int(y)] for x, y in bbox])
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"{text} ({conf:.2f})"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx = pts[0][0]
        ly = max(pts[0][1] - 8, th + 8)
        cv2.rectangle(out, (lx, ly - th - bl - 4), (lx + tw + 6, ly + bl),
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(out, label, (lx + 3, ly - bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def get_file_path(f) -> str:
    """Resolve a Gradio file object to an absolute path string."""
    if isinstance(f, dict):
        return f.get("name") or f.get("path") or str(f)
    if hasattr(f, "name"):
        return f.name
    return str(f)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(files, progress=gr.Progress()):
    if not files:
        return "<p style='color:red;font-family:sans-serif'>Please select a folder containing images.</p>"

    # Keep only supported image files (folder upload includes all file types)
    image_files = [
        f for f in files
        if Path(get_file_path(f)).suffix.lower() in SUPPORTED_EXT
    ]

    if not image_files:
        return "<p style='color:red;font-family:sans-serif'>No supported images found in the selected folder (JPG / PNG / BMP / WEBP).</p>"

    total = len(image_files)
    rows  = []
    files = image_files   # work only with image files from here on

    for idx, f in enumerate(image_files):
        file_path = get_file_path(f)
        filename  = Path(file_path).name

        # ── Progress: reading ─────────────────────────────────────────────────
        progress(idx / total, desc=f"[{idx + 1}/{total}] Reading {filename} …")

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            rows.append(_error_row(filename, "Could not read image"))
            continue

        # ── Progress: vehicle detection ───────────────────────────────────────
        progress((idx + 0.25) / total, desc=f"[{idx + 1}/{total}] Vehicle detection …")

        car_out = car_model(img_bgr, conf=CAR_CONF, verbose=False)[0]
        cars = []
        for cb in car_out.boxes:
            if int(cb.cls[0]) not in CAR_CLASS_IDS:
                continue
            x1, y1, x2, y2 = map(int, cb.xyxy[0].tolist())
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size:
                cars.append(dict(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf=round(float(cb.conf[0]), 4),
                    crop=crop,
                ))

        # Annotated overview image (vehicles = green box)
        vis = img_bgr.copy()
        for c in cars:
            cv2.rectangle(vis, (c["x1"], c["y1"]), (c["x2"], c["y2"]),
                          (0, 200, 60), 2)
            cv2.putText(vis, f"vehicle {c['conf']:.0%}",
                        (c["x1"] + 3, c["y1"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 60), 1, cv2.LINE_AA)

        if not cars:
            rows.append(_error_row(filename, "No vehicles detected", vis))
            continue

        # ── Progress: plate detection + OCR ───────────────────────────────────
        progress((idx + 0.55) / total, desc=f"[{idx + 1}/{total}] Plate detection + OCR …")

        plate_items = []
        for car in cars:
            plate_out = plate_model(car["crop"], conf=CONF_THRESHOLD, verbose=False)[0]
            for pb in plate_out.boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
                det_conf = round(float(pb.conf[0]), 4)
                pcrop = car["crop"][py1:py2, px1:px2]
                if not pcrop.size:
                    continue

                # Draw plate box (red) on main overview image (in original coords)
                cv2.rectangle(vis,
                              (car["x1"] + px1, car["y1"] + py1),
                              (car["x1"] + px2, car["y1"] + py2),
                              (0, 0, 220), 2)
                cv2.putText(vis, f"plate {det_conf:.0%}",
                            (car["x1"] + px1 + 2, car["y1"] + py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)

                # Upscale small crops so OCR / display is legible
                h, w = pcrop.shape[:2]
                if w < MIN_DISPLAY_W:
                    sc = MIN_DISPLAY_W / w
                    pcrop = cv2.resize(pcrop, (int(w * sc), int(h * sc)),
                                       interpolation=cv2.INTER_CUBIC)

                ocr_res = ocr_reader.readtext(to_gray_bgr(pcrop), **OCR_PARAMS)
                texts   = [(t, c) for _, t, c in ocr_res if c > 0.1]
                ocr_str = " | ".join(t for t, _ in texts) if texts else "(no text detected)"
                ocr_avg = sum(c for _, c in texts) / len(texts) if texts else 0.0

                ann = draw_ocr_annotations(pcrop, ocr_res)
                plate_items.append((ann, det_conf, ocr_str, ocr_avg))

        rows.append(_result_row(filename, vis, plate_items))

    progress(1.0, desc="Done!")
    return _wrap_table(rows)


# ── HTML Builders ──────────────────────────────────────────────────────────────

_CSS = """
<style>
.lp-wrap  { font-family: 'Segoe UI', sans-serif; }
.lp-table { border-collapse: collapse; width: 100%; }
.lp-table th {
    background: #1e3a5f; color: #fff;
    padding: 10px 14px; text-align: left;
    font-size: .9em; letter-spacing: .03em;
}
.lp-table td { border: 1px solid #dde; padding: 8px; vertical-align: top; }
.lp-table tr:nth-child(even) td { background: #f6f8fb; }
.fn   { font-weight: 700; font-size: .85em; max-width: 130px; word-break: break-all; color: #1e3a5f; }
.warn { color: #9e4600; font-style: italic; padding: 8px; }
.plates-flex { display: flex; flex-wrap: wrap; gap: 12px; }
.plate-card  {
    border: 1px solid #c8d4e8; border-radius: 8px; padding: 10px 8px;
    background: #fff; text-align: center; min-width: 220px; max-width: 300px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
.ocr-text {
    margin: 8px 0 5px; font-weight: 700;
    font-size: 1.1em; letter-spacing: .06em; color: #111;
}
.conf-row { display: flex; gap: 6px; justify-content: center; margin-top: 4px; }
.badge  { border-radius: 4px; padding: 2px 9px; font-size: .75em; font-weight: 700; color: #fff; }
.b-det  { background: #1f7a35; }
.b-ocr  { background: #1357a6; }
</style>
"""


def _img_tag(img: np.ndarray, max_h: int = 180) -> str:
    return f'<img src="{bgr_to_b64(img)}" style="max-height:{max_h}px;border-radius:4px"/>'


def _error_row(filename: str, msg: str, img: np.ndarray = None) -> str:
    fn_safe  = _html.escape(filename)
    msg_safe = _html.escape(msg)
    img_tag  = _img_tag(img) if img is not None else ""
    return (
        f'<tr>'
        f'<td class="fn">{fn_safe}</td>'
        f'<td>{img_tag}</td>'
        f'<td class="warn">{msg_safe}</td>'
        f'</tr>'
    )


def _result_row(filename: str, vis: np.ndarray, plate_items: list) -> str:
    fn_safe = _html.escape(filename)
    vis_tag = _img_tag(vis, max_h=200)

    if not plate_items:
        plates_html = '<td class="warn">No licence plates detected</td>'
    else:
        cards = []
        for ann, det_conf, ocr_str, ocr_avg in plate_items:
            ocr_safe = _html.escape(ocr_str)
            cards.append(
                f'<div class="plate-card">'
                f'  {_img_tag(ann, max_h=110)}'
                f'  <div class="ocr-text">{ocr_safe}</div>'
                f'  <div class="conf-row">'
                f'    <span class="badge b-det">det&nbsp;{det_conf:.0%}</span>'
                f'    <span class="badge b-ocr">ocr&nbsp;{ocr_avg:.0%}</span>'
                f'  </div>'
                f'</div>'
            )
        plates_html = f'<td><div class="plates-flex">{"".join(cards)}</div></td>'

    return (
        f'<tr>'
        f'<td class="fn">{fn_safe}</td>'
        f'<td>{vis_tag}</td>'
        f'{plates_html}'
        f'</tr>'
    )


def _wrap_table(rows: list) -> str:
    if not rows:
        return "<p style='font-family:sans-serif'>No results to display.</p>"
    body = "\n".join(rows)
    return (
        _CSS
        + '<div class="lp-wrap">'
        + '<table class="lp-table">'
        + '<thead><tr>'
        + '<th style="width:120px">File</th>'
        + '<th style="width:320px">Overview (vehicles&nbsp;&#x1F7E2; / plates&nbsp;&#x1F534;)</th>'
        + '<th>Detected Licence Plates &amp; OCR</th>'
        + '</tr></thead>'
        + f'<tbody>{body}</tbody>'
        + '</table>'
        + '</div>'
    )


# ── Gradio Layout ──────────────────────────────────────────────────────────────

with gr.Blocks(title="Licence Plate Detection & OCR", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# Licence Plate Detection & OCR\n"
        "**Pipeline:** vehicle detection (YOLOv11) → plate detection (fine-tuned YOLO) → EasyOCR  \n"
        "Results are shown one row per image. Green boxes = vehicles · Red boxes = plates."
    )

    with gr.Row():
        upload = gr.File(
            label="Drop a folder here or click to browse — selects all images inside (JPG / PNG / BMP / WEBP)",
            file_count="directory",
            scale=3,
        )

    run_btn = gr.Button("▶  Run Pipeline", variant="primary", size="lg")

    output = gr.HTML(label="Results")

    run_btn.click(fn=run_pipeline, inputs=upload, outputs=output)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
