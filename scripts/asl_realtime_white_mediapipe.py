import os
import shlex
import cv2
import joblib
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter, deque
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# PATHS / SETTINGS
# -----------------------------
MODEL_PATH = "models/cnn_svm_model.joblib"
ENCODER_PATH = "models/cnn_svm_encoder.joblib"
HAND_MODEL_PATH = "hand_landmarker.task"
IMG_SIZE = (224, 224)
ROI_X1, ROI_Y1 = 50, 50
ROI_X2, ROI_Y2 = 500, 500
SMOOTHING_WINDOW = 7

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing file: {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Missing file: {ENCODER_PATH}")
if not os.path.exists(HAND_MODEL_PATH):
    raise FileNotFoundError(f"Missing file: {HAND_MODEL_PATH} — download it first!")

clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)

# -----------------------------
# MEDIAPIPE HANDS (Tasks API)
# -----------------------------
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# WHITE BACKGROUND USING HAND LANDMARKS
# -----------------------------
def apply_white_background_with_landmarks(roi_bgr):
    h, w = roi_bgr.shape[:2]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
    result = hand_landmarker.detect(mp_image)

    # No hand detected — return white frame
    if not result.hand_landmarks:
        return np.ones_like(roi_bgr, dtype=np.uint8) * 255, False

    # Get landmark points
    hand_landmarks = result.hand_landmarks[0]
    points = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append([cx, cy])

    points = np.array(points, dtype=np.int32)

    # Convex hull around hand
    hull = cv2.convexHull(points)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Dilate to cover full hand skin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    mask_float = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    white_bg = np.ones_like(roi_bgr, dtype=np.uint8) * 255
    result_roi = (roi_bgr * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)

    return result_roi, True

# -----------------------------
# TEXT TO SPEECH (macOS say)
# -----------------------------
def speak_label(label):
    if label.lower() == "space":
        text = "space"
    elif label.lower() == "del":
        text = "delete"
    elif label.lower() == "nothing":
        text = "nothing"
    else:
        text = label
    safe_text = shlex.quote(text)
    os.system(f"say {safe_text}")

# -----------------------------
# PREDICTION
# -----------------------------
def predict_roi(roi_bgr):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
    roi_array = roi_resized.astype(np.float32)
    roi_array = preprocess_input(roi_array)
    roi_batch = np.expand_dims(roi_array, axis=0)
    features = feature_model.predict(roi_batch, verbose=0)
    pred_code = clf.predict(features)[0]
    pred_label = le.inverse_transform([pred_code])[0]
    return pred_label

def majority_vote(history):
    if not history:
        return "..."
    count = Counter(history)
    return count.most_common(1)[0][0]

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    history = deque(maxlen=SMOOTHING_WINDOW)
    current_label = "..."

    print("Press 's' to speak current prediction")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        hand_detected = False

        if roi.size != 0:
            clean_roi, hand_detected = apply_white_background_with_landmarks(roi)
            frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = clean_roi

            if hand_detected:
                try:
                    pred = predict_roi(clean_roi)
                    history.append(pred)
                    current_label = majority_vote(history)
                except Exception:
                    current_label = "Error"

        # Box turns green when hand detected, yellow when not
        box_color = (0, 255, 0) if hand_detected else (0, 255, 255)
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), box_color, 2)

        hand_status = "Hand Detected!" if hand_detected else "No Hand in Box"
        status_color = (0, 255, 0) if hand_detected else (0, 165, 255)
        cv2.putText(frame, hand_status, (ROI_X2 + 10, ROI_Y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Prediction: {current_label}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Show hand in box | s=speak | q=quit", (30, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("ASL Real-Time Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print("Speaking:", current_label)
            speak_label(current_label)
        elif key == ord("q"):
            break

    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()