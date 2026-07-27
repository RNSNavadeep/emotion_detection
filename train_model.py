import os
import joblib
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils import extract_features
import numpy as np
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "dataset/RAVDESS"

X = []
y = []

print("=" * 50)
print("Extracting Features...")
print("=" * 50)

for actor in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        try:

            feature = extract_features(file_path)
            if len(X) == 0:
                print("Feature Shape:", feature.shape)

            emotion_code = int(file.split("-")[2])
            if emotion_code == 2:
                continue

            emotion_map = {
                1: "neutral",
                3: "happy",
                4: "sad",
                5: "angry",
                6: "fearful",
                7: "disgust",
                8: "surprised"
}

            emotion = emotion_map[emotion_code]

            X.append(feature)
            y.append(emotion)

            print(file, "✓")

        except Exception as e:

            print(file, e)

print("\nCreating DataFrame...")

df = pd.DataFrame(X)
df["Emotion"] = y

df.to_csv("features/features.csv", index=False)

print("Features saved.")

X = np.array(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Model...\n")

model = SVC(
    kernel="rbf",
    C=100,
    gamma=0.01,
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", accuracy)

print("\nClassification Report\n")

print(classification_report(y_test, prediction))

joblib.dump(model, "models/emotion_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully")