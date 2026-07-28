import os

dataset_path = "dataset/RAVDESS"

actor_folders = sorted(
    [
        folder
        for folder in os.listdir(dataset_path)
        if folder.startswith("Actor_")
    ]
)

print("=" * 50)
print("RAVDESS DATASET CHECK")
print("=" * 50)

print(f"\nTotal Actor Folders : {len(actor_folders)}\n")

total_audio = 0

for actor in actor_folders:
    actor_path = os.path.join(dataset_path, actor)

    wav_files = [
        file
        for file in os.listdir(actor_path)
        if file.endswith(".wav")
    ]

    print(f"{actor:<10} : {len(wav_files)} audio files")

    total_audio += len(wav_files)

print("\n" + "=" * 50)
print(f"TOTAL AUDIO FILES : {total_audio}")
print("=" * 50)