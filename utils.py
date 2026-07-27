import librosa
import numpy as np


def load_audio(file_path, sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr


def extract_features(file_path):

    audio, sr = load_audio(file_path)

    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # If trimming makes audio too short, reload original audio
    if len(audio) < 2048:
        audio, sr = load_audio(file_path)

    # Normalize
    audio = librosa.util.normalize(audio)

    # Choose a safe FFT size based on audio length
    n_fft = min(1024, len(audio))

    # n_fft must be even
    if n_fft % 2 != 0:
        n_fft -= 1

    stft = np.abs(librosa.stft(audio, n_fft=n_fft))
    # ---------------- MFCC ----------------
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.hstack((np.mean(mfcc, axis=1),
                      np.std(mfcc, axis=1)))

    # ---------------- Chroma ----------------
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma = np.hstack((np.mean(chroma, axis=1),
                        np.std(chroma, axis=1)))

    # ---------------- Spectral Contrast ----------------
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast = np.hstack((np.mean(contrast, axis=1),
                          np.std(contrast, axis=1)))

    # ---------------- Tonnetz ----------------
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio),
        sr=sr
    )

    tonnetz = np.hstack((np.mean(tonnetz, axis=1),
                         np.std(tonnetz, axis=1)))

    # ---------------- RMS ----------------
    rms = librosa.feature.rms(y=audio)
    rms = np.hstack((np.mean(rms),
                     np.std(rms)))

    # ---------------- ZCR ----------------
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr = np.hstack((np.mean(zcr),
                     np.std(zcr)))

    features = np.hstack((
        mfcc,
        chroma,
        contrast,
        tonnetz,
        rms,
        zcr
    ))

    return features