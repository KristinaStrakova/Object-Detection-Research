"""Application settings and constants."""

import os

CONF_CAR = 0.40
CONF_PLATE = 0.30
CAR_CLASS_IDS = {2, 5, 7}  # COCO: car, bus, truck
CAR_MODEL_NAME = "models/yolo11n.pt"
PLATE_MODEL = "models/license-plate-finetune-v1n.onnx"
MIN_OCR_CONF = 0.10
MIN_DISPLAY_W = 400
ALLOWLIST = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789- "
OCR_PARAMS = dict(
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

RDW_API_URL = os.getenv("RDW_API_URL", "https://opendata.rdw.nl/resource/m9d7-ebf2.json")
RDW_TIMEOUT_S = float(os.getenv("RDW_TIMEOUT_S", "8"))
