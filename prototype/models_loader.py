"""Lazy-loaded model registry."""

from ultralytics import YOLO
import easyocr

from settings import CAR_MODEL_NAME, PLATE_MODEL

_car_model = None
_plate_model = None
_ocr_reader = None


def get_models():
    global _car_model, _plate_model, _ocr_reader
    if _car_model is None:
        _car_model = YOLO(CAR_MODEL_NAME)
    if _plate_model is None:
        _plate_model = YOLO(PLATE_MODEL)
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _car_model, _plate_model, _ocr_reader
