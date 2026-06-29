# Object Detection Research

This repository contains the full working area for researching and prototyping licence-plate detection and OCR.

It is split into two main parts:

- `jupiter_notebooks/` for experiments, comparisons, and pipeline development.
- `prototype/` for the runnable Flask application that wires the detection pipeline together.

The repo also includes model files, dataset exports, and sample imagery used during experimentation.

## Repository Layout

```text
Object-Detection-Research/
├── data-images/                 # Sample images used during development
├── images-eu/                   # Roboflow-style EU licence plate dataset export
├── jupiter_notebooks/           # Jupyter notebooks for experiments and model work
├── models/                      # Trained and downloaded model files
├── prototype/                   # Flask prototype application
├── README.md                    # This file
└── requirements.txt             # Shared Python dependencies
```

## What This Project Is For

The project explores a licence-plate workflow that typically looks like this:

1. Detect a vehicle or plate in an image.
2. Crop the plate region.
3. Run OCR on the crop.
4. Validate or post-process the result.

Different notebooks and prototype code paths focus on different parts of that flow, which is why the repository contains both research material and a small application.

## Jupyter Notebooks

The notebooks in `jupiter_notebooks/` are the research and experimentation layer of the project.

### `object_detection_training_own_model.ipynb`
Training workflow for your own object detection model. Use this notebook when you want to experiment with custom training data, labels, or hyperparameters.

### `model_comparison.ipynb`
General comparison notebook for trying different model setups and evaluating which approach works best for the dataset.

### `model_comparison_YOLO.ipynb`
Focused comparison notebook for YOLO-based experiments.

### `detect_car_then_detect_plate.ipynb`
Pipeline notebook for a two-step approach: detect the car first, then detect the licence plate inside the car crop.

### `object_detection+OCR.ipynb`
End-to-end notebook that connects object detection and OCR into one flow.

### `OBJ+OCR_prototype.ipynb`
Early prototype notebook for combining object detection with OCR.

### `OBJ+OCR_FineTuned.ipynb`
Version of the object detection + OCR pipeline that focuses on fine-tuned models and improved performance.

### How to use the notebooks

- Use them to explore model performance before changing the app.
- Use them to compare datasets, thresholds, and model versions.
- Use them as the place to prototype new logic before moving it into `prototype/`.

## Prototype Application

The `prototype/` folder contains the runnable Flask app.

### Main files

- `prototype/app.py` — Flask entrypoint and API routes.
- `prototype/pipeline.py` — image processing and inference pipeline.
- `prototype/models_loader.py` — model loading helpers.
- `prototype/rdw_client.py` — RDW vehicle lookup integration.
- `prototype/plate_rules.py` — licence plate validation and formatting helpers.
- `prototype/settings.py` — configuration values.
- `prototype/templates/index.html` — simple browser UI.

### What the prototype does

The app accepts uploaded images, runs the detection pipeline, aggregates plate reads across files, and can optionally validate confirmed plates against external vehicle information.

It is useful when you want a quick local UI for testing the pipeline without opening a notebook.

### Run the prototype

```bash
pip install -r requirements.txt
python prototype/app.py
```

Run that command from the repository root, then open the local Flask app in your browser at the port shown in the console output.

## Models

The `models/` folder stores pre-trained and exported model files used by the notebooks and the prototype application.

Example files in this repository include:

- `yolo11n.pt`
- `yolov8n.pt`
- `best.pt`
- `license-plate-finetune-v1n.pt` - fine-tuned licence plate detection model
- `license-plate-finetune-v1n.onnx` - exported licence plate detection model used by the prototype

Some notebooks and the prototype may download additional weights automatically the first time you run them.

## Datasets

The dataset folders contain exported image/label sets for training and validation.

- `images-eu/` contains a Roboflow-style export with `train/`, `valid/`, and `test/` splits.
- `data-images/` contains extra images used for testing and exploration.

## Installation

Create a virtual environment if you want to keep dependencies isolated, then install the shared requirements from the repository root.

```bash
pip install -r requirements.txt
```

The root `requirements.txt` is the canonical dependency file for this repository.

## Notes

- The repository keeps one main README at the root and one shared requirements file at the root.
- The prototype documentation was folded into this README so the project has a single entry point.
- If you add new notebooks or prototype steps, update this README so the repo stays easy to navigate.
