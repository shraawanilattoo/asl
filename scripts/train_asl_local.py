import os
from glob import glob
from tqdm import tqdm
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# -----------------------------
# PATHS / SETTINGS
# -----------------------------
TRAIN_DIR = "data/asl_alphabet_train"
TEST_DIR = "data/asl_alphabet_test"   # kept here in case you want to use it later
IMG_SIZE = (224, 224)
BATCH_SIZE = 64


# -----------------------------
# 1. LOAD CNN FEATURE EXTRACTOR
# -----------------------------
def build_feature_extractor():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model


# -----------------------------
# 2. EXTRACT FEATURES LOOP
# -----------------------------
def extract_features_from_directory(data_dir, model):
    X = []
    y = []

    classes = sorted([
        c for c in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, c))
    ])

    print(f"Extracting features from {len(classes)} classes in {data_dir}...")

    for label in tqdm(classes):
        class_dir = os.path.join(data_dir, label)
        files = glob(os.path.join(class_dir, "*.jpg")) + glob(os.path.join(class_dir, "*.jpeg")) + glob(os.path.join(class_dir, "*.png"))

        for i in range(0, len(files), BATCH_SIZE):
            batch_files = files[i:i + BATCH_SIZE]
            batch_imgs = []

            for f in batch_files:
                try:
                    img = load_img(f, target_size=IMG_SIZE)
                    img_arr = img_to_array(img)
                    img_arr = preprocess_input(img_arr)
                    batch_imgs.append(img_arr)
                except Exception as e:
                    print(f"Skipping {f}: {e}")
                    continue

            if batch_imgs:
                preds = model.predict(np.array(batch_imgs), verbose=0)
                X.extend(preds)
                y.extend([label] * len(preds))

    return np.array(X), np.array(y)


def main():
    print("Loading MobileNetV2 feature extractor...")
    cnn_model = build_feature_extractor()
    print("MobileNetV2 Feature Extractor Loaded.")

    print("\nExtracting features from training directory...")
    X_all, y_all = extract_features_from_directory(TRAIN_DIR, cnn_model)
    print(f"Extraction Complete. Feature Matrix Shape: {X_all.shape}")

    # -----------------------------
    # 3. DATA SPLITTING (80/10/10)
    # -----------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_encoded,
        train_size=0.8,
        random_state=42,
        stratify=y_encoded
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print(f"Train Count: {X_train.shape[0]}")
    print(f"Val Count:   {X_val.shape[0]}")
    print(f"Test Count:  {X_test.shape[0]}")

    # -----------------------------
    # 4. TRAIN LINEAR SVM
    # -----------------------------
    print("\nTraining Linear SVM...")
    clf = LinearSVC(C=1.0, dual=False, max_iter=10000)
    clf.fit(X_train, y_train)
    print("Training Complete.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/cnn_svm_model.joblib")
    joblib.dump(le, "models/cnn_svm_encoder.joblib")

    # -----------------------------
    # 5. EVALUATION
    # -----------------------------
    print("\nEvaluating on Test Split...")

    y_pred = clf.predict(X_test)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Training Accuracy:   {train_acc:.2%}")
    print(f"Validation Accuracy: {val_acc:.2%}")
    print(f"Test Accuracy:       {test_acc:.2%}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    report_text = f"""
============================================================
ASL RECOGNITION MODEL PERFORMANCE REPORT (Hybrid CNN + SVM)
============================================================

1. DATASET SPLIT (80/10/10)
---------------------------
Training Samples:   {X_train.shape[0]}
Validation Samples: {X_val.shape[0]}
Testing Samples:    {X_test.shape[0]}

2. ACCURACY METRICS
-------------------
Training Accuracy:   {train_acc:.2%}
Validation Accuracy: {val_acc:.2%}
Testing Accuracy:    {test_acc:.2%}

3. CLASSIFICATION REPORT
------------------------
{classification_report(y_test, y_pred, target_names=le.classes_)}
"""

    with open("models/model_performance_report.txt", "w") as f:
        f.write(report_text)

    print("\nSaved files:")
    print("models/cnn_svm_model.joblib")
    print("models/cnn_svm_encoder.joblib")
    print("models/model_performance_report.txt")


if __name__ == "__main__":
    main()