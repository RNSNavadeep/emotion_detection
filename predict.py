import joblib
import numpy as np
from utils import extract_features

model = joblib.load("models/emotion_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict_emotion(audio_path):
    print("Predicting:", audio_path)
    try:
        feature = extract_features(audio_path)
        feature = np.array(feature).reshape(1, -1)
        feature = scaler.transform(feature)

        prediction = model.predict(feature)[0]
        probabilities = model.predict_proba(feature)[0]

        confidence = np.max(probabilities) * 100
        emotion = encoder.inverse_transform([prediction])[0]

        probability_dict = {
            emotion: float(prob)
            for emotion, prob in zip(encoder.classes_, probabilities)
        }
        print(probability_dict)

        return emotion, confidence, probability_dict

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")