import os
import joblib
import numpy as np
from config import MODEL_PATH, MODEL_RF_PATH, LABEL_ENCODER_PATH, SCALER_PATH
from utils import extract_features, load_audio

# Cached objects
_MODEL_SVM = None
_MODEL_RF = None
_LABEL_ENCODER = None
_SCALER = None


def _load_artifacts():
    global _MODEL_SVM, _MODEL_RF, _LABEL_ENCODER, _SCALER

    if _LABEL_ENCODER is None and os.path.exists(LABEL_ENCODER_PATH):
        _LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)

    if _SCALER is None and os.path.exists(SCALER_PATH):
        _SCALER = joblib.load(SCALER_PATH)

    if _MODEL_SVM is None and os.path.exists(MODEL_PATH):
        _MODEL_SVM = joblib.load(MODEL_PATH)

    if _MODEL_RF is None and os.path.exists(MODEL_RF_PATH):
        _MODEL_RF = joblib.load(MODEL_RF_PATH)


def predict_emotion(audio_path, model_type="svm"):
    """
    Predict emotion for a given audio file.
    
    Returns:
    - emotion (str): Top predicted emotion label.
    - confidence (float): Percentage confidence score (0-100).
    - probability_dict (dict): Map of emotion -> probability score.
    - top3 (list): List of tuples [(emotion, prob%), ...] sorted by probability.
    """
    _load_artifacts()

    if _LABEL_ENCODER is None or _SCALER is None:
        raise RuntimeError("Model artifacts missing. Please train the model first by running train_model.py")

    model = _MODEL_SVM if model_type.lower() == "svm" else _MODEL_RF
    if model is None:
        model = _MODEL_SVM if _MODEL_SVM is not None else _MODEL_RF
        if model is None:
            raise RuntimeError(f"Requested model '{model_type}' is not loaded.")

    try:
        # Check audio validity
        audio, sr = load_audio(audio_path)
        if len(audio) < 1000:
            raise ValueError("Audio recording is too short or silent.")

        feature = extract_features(audio_path)
        feature_vector = np.array(feature).reshape(1, -1)
        feature_scaled = _SCALER.transform(feature_vector)

        prediction_idx = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]

        predicted_emotion = _LABEL_ENCODER.inverse_transform([prediction_idx])[0]
        confidence = float(np.max(probabilities) * 100)

        classes = _LABEL_ENCODER.classes_
        probability_dict = {
            cls_name: float(prob) for cls_name, prob in zip(classes, probabilities)
        }

        # Sort probability dict
        sorted_probs = sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)
        top3 = [(em, round(pr * 100, 2)) for em, pr in sorted_probs[:3]]

        return predicted_emotion, confidence, probability_dict, top3

    except Exception as e:
        raise RuntimeError(f"Prediction failed for '{audio_path}': {e}")