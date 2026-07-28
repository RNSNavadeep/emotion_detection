from utils import extract_features
import os

sample = "dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"

features = extract_features(sample)

print("Feature Shape :", features.shape)
print(features)


