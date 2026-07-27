import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "RAVDESS")
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
FEATURE_CSV = os.path.join(BASE_DIR, "features", "features.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "history", "session.csv")
RECORDING_PATH = os.path.join(BASE_DIR, "recordings", "recording.wav")

SAMPLE_RATE = 22050
N_MFCC = 40
RANDOM_STATE = 42