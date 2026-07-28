import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH


def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Load an audio file using librosa with guaranteed sample rate."""
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Unable to read audio file '{file_path}': {e}")


def extract_features(file_path):
    """
    Extract comprehensive acoustic features from an audio file (Fast & Robust):
    - MFCC (Mean & Std)
    - Chroma STFT (Mean & Std)
    - Mel Spectrogram (Mean & Std)
    - Spectral Contrast (Mean & Std)
    - RMS Energy (Mean & Std)
    - Zero Crossing Rate (Mean & Std)
    Total Feature Vector Dimensions: 378
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = load_audio(file_path)

        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio_trimmed) >= 2048:
            audio = audio_trimmed

        # Normalize amplitude
        audio = librosa.util.normalize(audio)

        # Dynamic FFT size for STFT computation
        n_fft = min(N_FFT, len(audio))
        if n_fft % 2 != 0:
            n_fft -= 1

        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=HOP_LENGTH))

        # 1. MFCCs (40 coeffs -> 80 features)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=HOP_LENGTH)
        mfcc_features = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)))

        # 2. Chroma STFT (12 bins -> 24 features)
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_features = np.hstack((np.mean(chroma, axis=1), np.std(chroma, axis=1)))

        # 3. Mel Spectrogram (128 bands -> 256 features)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_features = np.hstack((np.mean(mel_db, axis=1), np.std(mel_db, axis=1)))

        # 4. Spectral Contrast (7 bands -> 14 features)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_features = np.hstack((np.mean(contrast, axis=1), np.std(contrast, axis=1)))

        # 5. RMS Energy (1 band -> 2 features)
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
        rms_features = np.hstack((np.mean(rms), np.std(rms)))

        # 6. Zero Crossing Rate (ZCR) (1 band -> 2 features)
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)
        zcr_features = np.hstack((np.mean(zcr), np.std(zcr)))

        # Concatenate all features
        features = np.hstack((
            mfcc_features,
            chroma_features,
            mel_features,
            contrast_features,
            rms_features,
            zcr_features
        ))

        return features


def plot_waveform(audio, sr):
    """Generate a Plotly waveform figure."""
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    if len(time_axis) > 10000:
        step = len(time_axis) // 10000
        time_axis = time_axis[::step]
        audio_plot = audio[::step]
    else:
        audio_plot = audio

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=audio_plot,
            mode='lines',
            name='Amplitude',
            line=dict(color='#00E676', width=1.5)
        )
    )
    fig.update_layout(
        title="Audio Waveform (Time Domain)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_spectrogram(audio, sr):
    """Generate a Mel-Spectrogram figure using Plotly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = px.imshow(
        mel_db,
        origin='lower',
        aspect='auto',
        color_continuous_scale='Magma',
        labels=dict(x="Time Frame", y="Mel Frequency Bin", color="dB")
    )
    fig.update_layout(
        title="Mel-Spectrogram (Frequency Domain)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_mfcc_heatmap(audio, sr):
    """Generate an MFCC heatmap figure using Plotly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    fig = px.imshow(
        mfcc,
        origin='lower',
        aspect='auto',
        color_continuous_scale='Viridis',
        labels=dict(x="Time Frame", y="MFCC Coefficient", color="Value")
    )
    fig.update_layout(
        title="MFCC Feature Heatmap",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig