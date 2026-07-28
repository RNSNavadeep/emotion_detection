import os
from predict import predict_emotion
from config import DATASET_PATH

def test_predictions():
    print("=" * 50)
    print("      TESTING PREDICTION ENGINE ON SAMPLE AUDIO FILES      ")
    print("=" * 50)

    if not os.path.exists(DATASET_PATH):
        print("Dataset directory not found.")
        return

    actor_dirs = sorted([d for d in os.listdir(DATASET_PATH) if d.startswith("Actor_")])
    if not actor_dirs:
        print("No actor folders found.")
        return

    count = 0
    for actor in actor_dirs[:2]:
        actor_path = os.path.join(DATASET_PATH, actor)
        wav_files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]

        for file in wav_files[:5]:
            file_path = os.path.join(actor_path, file)
            try:
                emotion, confidence, prob_dict, top3 = predict_emotion(file_path, model_type="svm")
                print(f"File: {file:<28} -> Predicted: {emotion.upper():<10} ({confidence:.1f}%) | Top-3: {top3}")
                count += 1
            except Exception as e:
                print(f"Error testing {file}: {e}")

    print("=" * 50)
    print(f"Tested {count} audio files successfully.")

if __name__ == "__main__":
    test_predictions()