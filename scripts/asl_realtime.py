import os
import shlex
import cv2
import joblib
import numpy as np
import tensorflow as tf

from collections import Counter, deque
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# PATHS / SETTINGS
# -----------------------------
MODEL_PATH = "models/cnn_svm_model.joblib"
ENCODER_PATH = "models/cnn_svm_encoder.joblib"

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

clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)


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

        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        if roi.size != 0:
            try:
                pred = predict_roi(roi)
                history.append(pred)
                current_label = majority_vote(history)
            except Exception:
                current_label = "Error"

        cv2.putText(
            frame,
            f"Prediction: {current_label}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Show hand in box | Press s to speak | q to quit",
            (30, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("ASL Real-Time Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            print("Speaking:", current_label)
            speak_label(current_label)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()