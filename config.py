import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Paths
DATASET_PATH = BASE_DIR / "dataset" / "RAVDESS"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "emotion_model.pkl"
MODEL_RF_PATH = MODELS_DIR / "emotion_model_rf.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_meta.json"

FEATURES_DIR = BASE_DIR / "features"
FEATURE_CSV = FEATURES_DIR / "features.csv"

HISTORY_DIR = BASE_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "session.csv"

RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDING_PATH = RECORDINGS_DIR / "recording.wav"

TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
for path_dir in [MODELS_DIR, FEATURES_DIR, HISTORY_DIR, RECORDINGS_DIR, TEMP_DIR]:
    path_dir.mkdir(parents=True, exist_ok=True)

# Audio Feature Constants
SAMPLE_RATE = 22050
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
RANDOM_STATE = 42

# RAVDESS Emotion Code Mapping
EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

# Emotion Emoji Mapping
EMOTION_EMOJI = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

# Color Mapping for UI & Graphs
EMOTION_COLORS = {
    "neutral": "#9E9E9E",
    "calm": "#4FC3F7",
    "happy": "#FFD54F",
    "sad": "#7986CB",
    "angry": "#E57373",
    "fearful": "#BA68C8",
    "disgust": "#81C784",
    "surprised": "#FF8A65"
}