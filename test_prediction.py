import os
from predict import predict_emotion

DATASET = "dataset/RAVDESS"

count = 0

for actor in os.listdir(DATASET):

    actor_path = os.path.join(DATASET, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            emotion, confidence, _ = predict_emotion(
                os.path.join(actor_path, file)
            )

            print(file, "->", emotion)

            count += 1

            if count == 20:
                quit()