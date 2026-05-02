# Real-Time American Sign Language Recognition Using Hybrid CNN-SVM Architecture

A real-time American Sign Language (ASL) alphabet recognition system that uses a standard webcam as input and delivers audio output of the predicted letter. Built using a hybrid CNN-SVM pipeline combining MobileNetV2 feature extraction with a Linear SVM classifier.

---

## Demo

The system captures live hand gestures through a webcam, isolates the hand using MediaPipe landmark detection, and classifies the displayed ASL letter — announcing it through audio output.

---

---

## Dataset

This project uses the **ASL Alphabet Dataset** available on Kaggle.

Download it here: [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

After downloading, organize the data as follows:

```
ASL/
├── data/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       ├── C/
│       └── ... (all 29 classes)
├── models/
├── scripts/
└── hand_landmarker.task
```

---

## Project Structure

```
ASL/
├── data/                           # Download separately from Kaggle
├── models/
│   ├── cnn_svm_model.joblib        # Trained Linear SVM classifier
│   ├── cnn_svm_encoder.joblib      # Label encoder
│   └── model_performance_report.txt
├── scripts/
│   ├── train.py                    # Training pipeline
│   ├── asl_realtime.py             # Version 1 — baseline inference
│   ├── asl_realtime_whitebg.py     # Version 2 — MOG2 background subtraction
│   └── asl_realtime_v3.py          # Version 3 — MediaPipe segmentation
└── hand_landmarker.task            # MediaPipe hand landmark model
```

---
## Requirements

Install all dependencies:

```bash
pip install tensorflow
pip install scikit-learn
pip install opencv-python
pip install mediapipe
pip install numpy
pip install joblib
pip install tqdm
```

> **Note:** This project was developed and tested on a MacBook Air M3 (13-inch, 16GB RAM) running macOS. The text-to-speech feature uses the macOS `say` command and is macOS-specific.

---

## MediaPipe Hand Landmark Model

Download the MediaPipe hand landmark model file and place it in the root project directory:

[Download hand_landmarker.task](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

---

## Usage

### Step 1 — Train the Model

```bash
python scripts/train.py
```

This will:
- Extract MobileNetV2 features from the dataset
- Train a Linear SVM classifier
- Save the model and encoder to the `models/` directory
- Generate a performance report at `models/model_performance_report.txt`

---

### Step 2 — Run Real-Time Inference

Three versions are available, each with improved hand segmentation:

**Version 1 — Baseline (no background removal)**
```bash
python scripts/asl_realtime.py
```

**Version 2 — MOG2 Background Subtraction**
```bash
python scripts/asl_realtime_whitebg.py
```
> Press `b` to lock the background before placing your hand in the box.

**Version 3 — MediaPipe Landmark Segmentation (recommended)**
```bash
python scripts/asl_realtime_v3.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `s` | Speak the current predicted letter aloud |
| `q` | Quit the application |
| `b` | Lock background (Version 2 only) |

---

## Model Performance

| Split      | Accuracy  |
|------------|-----------|
| Training   | 100.00%   |
| Validation | 99.89%    |
| Testing    | 99.80%    |

Evaluated across 29 ASL classes with 8,700 test samples (300 per class).

---

## System Versions

| Feature | Version 1 | Version 2 | Version 3 |
|---------|-----------|-----------|-----------|
| Background Removal | None | MOG2 Subtraction | MediaPipe Hull Mask |
| Hand Detection | None | None | MediaPipe Landmarks |
| Calibration Required | No | Yes | No |
| Lighting Robustness | Low | Medium | High |
| Prediction Suppression | No | No | Yes |

---

## Author

**Shraawani Lattoo**

---

## License

This project is for academic purposes only.
