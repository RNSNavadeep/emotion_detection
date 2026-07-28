import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import (
    DATASET_PATH, MODEL_PATH, MODEL_RF_PATH, LABEL_ENCODER_PATH,
    SCALER_PATH, METADATA_PATH, FEATURE_CSV, EMOTION_MAP, RANDOM_STATE
)
from utils import extract_features


def main():
    print("=" * 60)
    print("      SPEECH EMOTION RECOGNITION - MODEL TRAINING PIPELINE    ")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"RAVDESS dataset directory not found at '{DATASET_PATH}'")

    X = []
    y = []
    file_count = 0
    failed_count = 0

    print(f"\n[1/5] Extracting audio features from RAVDESS dataset...")
    
    actor_dirs = sorted([d for d in os.listdir(DATASET_PATH) if d.startswith("Actor_")])
    if not actor_dirs:
        raise ValueError(f"No Actor folders found in '{DATASET_PATH}'")

    for actor in actor_dirs:
        actor_path = os.path.join(DATASET_PATH, actor)
        if not os.path.isdir(actor_path):
            continue

        wav_files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
        for file in wav_files:
            file_path = os.path.join(actor_path, file)

            try:
                # RAVDESS filename standard: 03-01-01-01-01-01-01.wav
                parts = file.split("-")
                if len(parts) < 3:
                    continue

                emotion_code = int(parts[2])
                if emotion_code not in EMOTION_MAP:
                    continue

                emotion_label = EMOTION_MAP[emotion_code]
                feature_vector = extract_features(file_path)

                X.append(feature_vector)
                y.append(emotion_label)

                file_count += 1
                if file_count % 100 == 0:
                    print(f"   Processed {file_count} audio files...")

            except Exception as e:
                failed_count += 1
                print(f"   [Error] Failed to process {file}: {e}")

    print(f"   [OK] Feature extraction complete! Total valid audio files: {file_count} (Failed: {failed_count})")
    if file_count == 0:
        raise RuntimeError("No features extracted. Please check dataset directory.")

    # Save features CSV
    print(f"\n[2/5] Saving extracted feature dataset to '{FEATURE_CSV}'...")
    os.makedirs(os.path.dirname(FEATURE_CSV), exist_ok=True)
    df_features = pd.DataFrame(X)
    df_features["Emotion"] = y
    df_features.to_csv(FEATURE_CSV, index=False)
    print(f"   [OK] Feature matrix shape: {np.array(X).shape}")

    # Prepare features and labels
    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print(f"\n[3/5] Training Support Vector Machine (SVM) Classifier...")
    svm_model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE
    )
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average="weighted")

    print(f"   [OK] SVM Accuracy  : {accuracy_svm * 100:.2f}%")
    print(f"   [OK] SVM F1-Score  : {f1_svm * 100:.2f}%")

    print(f"\n[4/5] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

    print(f"   [OK] RF Accuracy   : {accuracy_rf * 100:.2f}%")
    print(f"   [OK] RF F1-Score   : {f1_rf * 100:.2f}%")

    print("\n--- SVM Classification Report ---")
    print(classification_report(y_test, y_pred_svm, target_names=encoder.classes_))

    # Save models and artifacts
    print(f"\n[5/5] Saving model artifacts to '{os.path.dirname(MODEL_PATH)}'...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(svm_model, MODEL_PATH)
    joblib.dump(rf_model, MODEL_RF_PATH)
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": file_count,
        "features_count": X.shape[1],
        "svm_accuracy": round(accuracy_svm * 100, 2),
        "svm_f1": round(f1_svm * 100, 2),
        "rf_accuracy": round(accuracy_rf * 100, 2),
        "rf_f1": round(f1_rf * 100, 2),
        "classes": list(encoder.classes_)
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=4)

    print("   [OK] Artifacts successfully saved:")
    print(f"     - SVM Model      : {MODEL_PATH}")
    print(f"     - Random Forest  : {MODEL_RF_PATH}")
    print(f"     - Label Encoder  : {LABEL_ENCODER_PATH}")
    print(f"     - Scaler         : {SCALER_PATH}")
    print(f"     - Metadata       : {METADATA_PATH}")
    print("=" * 60)
    print("                  TRAINING COMPLETED SUCCESSFULLY!            ")
    print("=" * 60)


if __name__ == "__main__":
    main()