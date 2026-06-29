"""Detection pipeline: car -> plate -> OCR and aggregation helpers."""

import base64

import cv2
import numpy as np

from settings import CONF_CAR, CONF_PLATE, CAR_CLASS_IDS, MIN_OCR_CONF, MIN_DISPLAY_W, OCR_PARAMS
from models_loader import get_models
from plate_rules import normalise_plate, parse_country, match_dutch_sidecode


def to_gray_bgr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def encode_img(img: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG data-URL."""
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def merge_crop_detections(dets: list) -> list:
    """
    Within a single plate-crop's OCR results, pair a country-indicator
    detection with a plate-number detection and merge them into one entry.
    """
    if len(dets) <= 1:
        return dets

    country_det = None
    plate_det = None
    for d in dets:
        if d["has_country"] and d["plate_num"] == "":
            if country_det is None or d["ocr_conf"] > country_det["ocr_conf"]:
                country_det = d
        elif not d["has_country"] and d["dutch_sidecode"] is not None:
            if plate_det is None or d["ocr_conf"] > plate_det["ocr_conf"]:
                plate_det = d

    if country_det is None or plate_det is None:
        return dets

    code = country_det["country_code"]
    merged_text = f"{code} | {plate_det['plate_num']}"
    merged = {
        "text": merged_text,
        "country_code": code,
        "country": country_det["country"],
        "has_country": True,
        "plate_num": plate_det["plate_num"],
        "ocr_conf": round((country_det["ocr_conf"] + plate_det["ocr_conf"]) / 2, 4),
        "plate_conf": plate_det["plate_conf"],
        "car_conf": plate_det["car_conf"],
        "dutch_sidecode": plate_det["dutch_sidecode"],
        "sidecode_pattern": plate_det["sidecode_pattern"],
        "image_b64": plate_det["image_b64"],
    }
    remaining = [d for d in dets if d is not country_det and d is not plate_det]
    remaining.append(merged)
    return remaining


def process_image(img_bytes: bytes):
    """
    Run the full pipeline on one image.
    Returns list of detections with OCR and sidecode metadata.
    """
    car_model, plate_model, ocr_reader = get_models()

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    detections = []

    car_out = car_model(img, conf=CONF_CAR, verbose=False)[0]
    for cb in car_out.boxes:
        if int(cb.cls[0]) not in CAR_CLASS_IDS:
            continue

        cx1, cy1, cx2, cy2 = map(int, cb.xyxy[0].tolist())
        car_crop = img[cy1:cy2, cx1:cx2]
        if car_crop.size == 0:
            continue
        car_conf = float(cb.conf[0])

        plate_out = plate_model(car_crop, conf=CONF_PLATE, verbose=False)[0]
        for pb in plate_out.boxes:
            px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
            plate_crop = car_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            plate_conf = float(pb.conf[0])

            h, w = plate_crop.shape[:2]
            if w < MIN_DISPLAY_W:
                scale = MIN_DISPLAY_W / w
                plate_crop = cv2.resize(
                    plate_crop,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            ocr_res = ocr_reader.readtext(to_gray_bgr(plate_crop), **OCR_PARAMS)
            plate_b64 = encode_img(plate_crop)
            crop_dets = []
            for _, text, conf in ocr_res:
                if conf < MIN_OCR_CONF or not text.strip():
                    continue
                norm = normalise_plate(text)
                if not norm:
                    continue
                country_code, country, plate_num = parse_country(norm)
                sc, sc_pattern = match_dutch_sidecode(plate_num)
                crop_dets.append(
                    {
                        "text": norm,
                        "country_code": country_code,
                        "country": country,
                        "has_country": country is not None,
                        "plate_num": plate_num,
                        "ocr_conf": round(conf, 4),
                        "plate_conf": round(plate_conf, 4),
                        "car_conf": round(car_conf, 4),
                        "dutch_sidecode": sc,
                        "sidecode_pattern": sc_pattern,
                        "image_b64": plate_b64,
                    }
                )
            detections.extend(merge_crop_detections(crop_dets))

    return detections
