import time

def predict_emotion(audio_path):

    print("Loading:", audio_path)

    start = time.time()

    feature = extract_features(audio_path)

    print("Feature extraction took:",
          time.time() - start)

    start = time.time()

    feature = np.array(feature).reshape(1, -1)

    feature = scaler.transform(feature)

    print("Scaling took:",
          time.time() - start)

    start = time.time()

    prediction = model.predict(feature)[0]

    print("Prediction took:",
          time.time() - start)

    probabilities = model.predict_proba(feature)[0]

    confidence = np.max(probabilities) * 100

    emotion = encoder.inverse_transform([prediction])[0]

    probability_dict = {
        emotion: float(prob)
        for emotion, prob in zip(encoder.classes_, probabilities)
    }

    return emotion, confidence, probability_dict