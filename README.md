# 🎤 AI Human Emotion Detection from Voice

A Speech Emotion Recognition (SER) application built using Machine Learning and Streamlit. The system predicts human emotions from voice recordings using audio feature extraction and a Support Vector Machine (SVM) classifier.

---

## Features

- 🎤 Live Voice Recording
- 📁 Upload Audio Files
- 😊 Emotion Prediction
- 📊 Confidence Score
- 📈 Emotion Probability Chart
- 📝 Prediction History
- 🌙 Professional Dark Theme UI

---

## Tech Stack

- Python
- Streamlit
- Librosa
- Scikit-learn
- Plotly
- NumPy
- Pandas
- Joblib

---

## Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

---

## Machine Learning Pipeline

1. Audio Input
2. Feature Extraction
   - MFCC
   - Chroma
   - Spectral Contrast
   - Tonnetz
   - RMS Energy
   - Zero Crossing Rate
3. Feature Scaling
4. Support Vector Machine (SVM)
5. Emotion Prediction
6. Confidence Score
7. Visualization

---

## Supported Emotions

- Neutral 😐
- Happy 😄
- Sad 😢
- Angry 😡
- Fearful 😨
- Disgust 🤢
- Surprised 😲

---

## Model Performance

**Accuracy:** 72%

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```
Emotion_Detection/
│
├── app.py
├── predict.py
├── train_model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
├── dataset/
├── recordings/
├── history/
└── features/
```

---

## Developer

**RNS Navadeep**

B.Tech Information Technology

AI & Machine Learning Enthusiast