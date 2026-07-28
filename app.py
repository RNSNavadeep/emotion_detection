import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from config import (
    MODEL_PATH, METADATA_PATH, HISTORY_FILE, RECORDING_PATH, TEMP_DIR,
    EMOTION_EMOJI, EMOTION_COLORS
)
from utils import load_audio, plot_waveform, plot_spectrogram, plot_mfcc_heatmap
from predict import predict_emotion

# Streamlit Component Fallback for Audio Recording
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Voice Emotion AI - SER Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Ensure folders exist
os.makedirs("recordings", exist_ok=True)
os.makedirs("history", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Helper function to read history CSV safely
def load_history_safe():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 5:
        try:
            df = pd.read_csv(HISTORY_FILE, on_bad_lines='skip')
            # Ensure expected columns
            expected_cols = ["Time", "Emotion", "Confidence", "Model"]
            for col in expected_cols:
                if col not in df.columns:
                    if col == "Time" and "Timestamp" in df.columns:
                        df.rename(columns={"Timestamp": "Time"}, inplace=True)
                    else:
                        df[col] = "N/A"
            return df[expected_cols]
        except Exception:
            pass
    return pd.DataFrame(columns=["Time", "Emotion", "Confidence", "Model"])

# -------------------------------------------------------
# ULTRA-HIGH CONTRAST & CRISP UI STYLING (CSS)
# -------------------------------------------------------
st.markdown("""
<style>
/* Main High-Contrast Dark Theme */
.stApp {
    background-color: #090D16 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Hide Streamlit top margin */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Crisp Solid Header Banner */
.main-header {
    background-color: #1E293B;
    padding: 24px 30px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 24px;
    border: 1px solid #334155;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.main-header h1 {
    color: #38BDF8 !important;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.main-header p {
    font-size: 16px;
    color: #CBD5E1 !important;
    margin-bottom: 0;
}

/* Crisp Solid Cards */
.glass-card {
    background-color: #0F172A;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #334155;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}

/* Crisp Hero Card */
.hero-card {
    background-color: #1E293B;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    border: 2px solid #38BDF8;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.hero-emoji {
    font-size: 68px;
    line-height: 1;
    margin-bottom: 8px;
}

.hero-emotion {
    font-size: 38px;
    font-weight: 900;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.hero-confidence {
    font-size: 18px;
    color: #34D399 !important;
    font-weight: 700;
}

/* High Contrast Metric Cards */
.metric-card {
    background-color: #1E293B;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #475569;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #38BDF8 !important;
}

.metric-label {
    font-size: 13px;
    color: #CBD5E1 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* High Contrast Pill Tags */
.pill-tag {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin: 4px;
    background-color: #334155;
    color: #F8FAFC !important;
    border: 1px solid #475569;
}

/* Tab Overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0F172A;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #334155;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 8px;
    color: #94A3B8 !important;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
    border: 1px solid #38BDF8 !important;
}

/* Crisp Text Fix */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #F8FAFC !important;
}

div[data-testid="stMarkdownContainer"] p {
    color: #E2E8F0 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎙️ Speech Emotion AI")
    st.caption("Human Emotion Detection from Voice")
    st.divider()

    # Load Model Metadata if available
    meta_info = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta_info = json.load(f)
        except Exception:
            pass

    svm_acc = meta_info.get("svm_accuracy", 72.6)
    rf_acc = meta_info.get("rf_accuracy", 56.6)

    # Model Selector
    st.subheader("⚙️ Settings")
    model_choice = st.selectbox(
        "Classifier Algorithm",
        options=["SVM (Support Vector Machine)", "Random Forest Classifier"],
        index=0,
        help="Select the trained Machine Learning classifier to perform emotion inference."
    )
    selected_model_type = "svm" if "SVM" in model_choice else "rf"
    current_acc = svm_acc if selected_model_type == "svm" else rf_acc

    # Model Accuracy Metric
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{model_choice.split()[0]} Accuracy</div>
        <div class="metric-value">{current_acc:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Dataset & Tech Details
    st.markdown("### 📚 Project Metadata")
    st.markdown("<span class='pill-tag'>Dataset: RAVDESS</span>", unsafe_allow_html=True)
    st.markdown("<span class='pill-tag'>Emotions: 8 Classes</span>", unsafe_allow_html=True)
    st.markdown("<span class='pill-tag'>Features: 378 (MFCC+Mel+Chroma)</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 👨‍💻 Developer")
    st.success("**RNS Navadeep**  \n*B.Tech Information Technology*")

    st.divider()
    st.caption("v2.0 • Speech Emotion Recognition")

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎤 Human Emotion Detection from Voice</h1>
    <p>Detect speaker emotion in real-time from acoustic audio using Machine Learning & Signal Processing</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# MAIN DASHBOARD TABS
# -------------------------------------------------------
tab_predict, tab_trend, tab_explorer, tab_insights = st.tabs([
    "🎙️ Live Predictor",
    "📈 Emotional Trend & History",
    "🔬 Audio Signal Visualizer",
    "📊 Model & Dataset Insights"
])

# =======================================================
# TAB 1: LIVE PREDICTOR
# =======================================================
with tab_predict:
    col_input, col_display = st.columns([5, 7])

    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎵 Audio Input Source")

        input_mode = st.radio(
            "Select Audio Mode",
            ["📁 Upload Audio File", "🎙️ Record Live Voice"],
            horizontal=True
        )

        if "Upload" in input_mode:
            uploaded_file = st.file_uploader(
                "Upload a WAV, MP3, OGG, or FLAC audio sample",
                type=["wav", "mp3", "ogg", "flac", "m4a"]
            )
            if uploaded_file is not None:
                save_path = os.path.join("recordings", "upload.wav")
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state.audio_path = save_path
        else:
            if hasattr(st, "audio_input"):
                recorded_audio = st.audio_input("Record your voice")
                if recorded_audio is not None:
                    save_path = os.path.join("recordings", "live_rec.wav")
                    with open(save_path, "wb") as f:
                        f.write(recorded_audio.read())
                    st.session_state.audio_path = save_path
            elif HAS_AUDIO_RECORDER:
                st.write("Click the microphone icon below to start recording:")
                audio_bytes = audio_recorder(pause_threshold=2.5, sample_rate=22050)
                if audio_bytes:
                    save_path = os.path.join("recordings", "live_rec.wav")
                    with open(save_path, "wb") as f:
                        f.write(audio_bytes)
                    st.session_state.audio_path = save_path
            else:
                st.warning("Live microphone recording requires audio-recorder-streamlit or modern Streamlit.")

        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            st.audio(st.session_state.audio_path)
            st.success("✅ Audio ready for emotion analysis.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Predict Button
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            run_btn = st.button("🚀 Predict Speaker Emotion", use_container_width=True, type="primary")
            if run_btn:
                with st.spinner("Analyzing acoustic features & running classifier..."):
                    try:
                        emotion, confidence, prob_dict, top3 = predict_emotion(
                            st.session_state.audio_path,
                            model_type=selected_model_type
                        )
                        st.session_state.last_prediction = {
                            "emotion": emotion,
                            "confidence": confidence,
                            "prob_dict": prob_dict,
                            "top3": top3,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # Append safely to Session History CSV
                        hist_data = {
                            "Time": [st.session_state.last_prediction["time"]],
                            "Emotion": [emotion],
                            "Confidence": [round(confidence, 2)],
                            "Model": [selected_model_type.upper()]
                        }
                        df_new = pd.DataFrame(hist_data)
                        if os.path.exists(HISTORY_FILE):
                            df_existing = load_history_safe()
                            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                            df_combined.to_csv(HISTORY_FILE, index=False)
                        else:
                            df_new.to_csv(HISTORY_FILE, index=False)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

    with col_display:
        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            emotion = pred["emotion"]
            confidence = pred["confidence"]
            prob_dict = pred["prob_dict"]
            top3 = pred["top3"]
            emoji = EMOTION_EMOJI.get(emotion, "🎭")
            color = EMOTION_COLORS.get(emotion, "#38BDF8")

            # Hero Prediction Result Card
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-emoji">{emoji}</div>
                <div class="hero-emotion" style="color: {color};">{emotion}</div>
                <div class="hero-confidence">Model Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            # Top 3 Predictions Pills
            st.markdown("**Top 3 Likely Emotions:**")
            pill_cols = st.columns(3)
            for i, (em_name, pr_val) in enumerate(top3):
                em_ic = EMOTION_EMOJI.get(em_name, "🎭")
                with pill_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:20px;">{em_ic}</div>
                        <div style="font-weight:700; text-transform:capitalize; color:#F8FAFC;">{em_name}</div>
                        <div style="color:#34D399; font-size:14px; font-weight:700;">{pr_val:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.write("")

            # Probability Distribution Chart
            st.subheader("📊 Emotion Probability Distribution")
            df_prob = pd.DataFrame({
                "Emotion": [e.capitalize() for e in prob_dict.keys()],
                "Probability (%)": [p * 100 for p in prob_dict.values()],
                "RawEmotion": list(prob_dict.keys())
            }).sort_values(by="Probability (%)", ascending=True)

            fig_prob = px.bar(
                df_prob,
                x="Probability (%)",
                y="Emotion",
                orientation='h',
                text_auto='.1f',
                color="RawEmotion",
                color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()}
            )
            fig_prob.update_layout(
                template="plotly_dark",
                paper_bgcolor="#090D16",
                plot_bgcolor="#0F172A",
                font_color="#F8FAFC",
                height=320,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        else:
            st.info("👈 Upload an audio file or record your voice and click 'Predict Speaker Emotion' to see predictions!")

# =======================================================
# TAB 2: EMOTIONAL TREND & HISTORY
# =======================================================
with tab_trend:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 Session Emotional Trend & Analytics")
    st.caption("Track emotional shifts across recorded audio samples in your current session.")

    df_hist = load_history_safe()

    if not df_hist.empty:
        # Metrics Header
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Total Predictions", len(df_hist))
        with m_col2:
            mode_series = df_hist["Emotion"].mode()
            dominant_em = mode_series[0].capitalize() if not mode_series.empty else "N/A"
            st.metric("Dominant Emotion", dominant_em)
        with m_col3:
            try:
                avg_conf = pd.to_numeric(df_hist["Confidence"], errors='coerce').mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            except Exception:
                st.metric("Avg Confidence", "N/A")

        st.write("")

        # Trend Line Chart
        try:
            fig_trend = px.line(
                df_hist,
                x=df_hist.index + 1,
                y=pd.to_numeric(df_hist["Confidence"], errors='coerce'),
                color="Emotion",
                markers=True,
                title="Emotional Prediction Trend Timeline",
                labels={"x": "Sample #", "y": "Confidence Score (%)"},
                color_discrete_map=EMOTION_COLORS
            )
            fig_trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="#090D16",
                plot_bgcolor="#0F172A",
                font_color="#F8FAFC",
                height=350
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render trend chart: {e}")

        st.subheader("📜 Session Prediction Log")
        st.dataframe(df_hist.iloc[::-1], use_container_width=True)

        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn1:
            csv_bytes = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session CSV",
                data=csv_bytes,
                file_name="voice_emotion_session_history.csv",
                mime="text/csv"
            )
        with col_btn2:
            if st.button("🗑️ Clear History"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()
    else:
        st.info("No session predictions recorded yet. Run predictions in Tab 1 to build your emotional trend graph!")

    st.markdown("</div>", unsafe_allow_html=True)

# =======================================================
# TAB 3: AUDIO SIGNAL & FEATURE VISUALIZER
# =======================================================
with tab_explorer:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔬 Acoustic Signal Analysis")
    st.caption("Inspect the spectral properties and acoustic features of the active audio sample.")

    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        try:
            audio_data, sr_data = load_audio(st.session_state.audio_path)

            col_w, col_s = st.columns(2)
            with col_w:
                fig_wave = plot_waveform(audio_data, sr_data)
                st.plotly_chart(fig_wave, use_container_width=True)
            with col_s:
                fig_spec = plot_spectrogram(audio_data, sr_data)
                st.plotly_chart(fig_spec, use_container_width=True)

            fig_mfcc = plot_mfcc_heatmap(audio_data, sr_data)
            st.plotly_chart(fig_mfcc, use_container_width=True)

        except Exception as e:
            st.error(f"Unable to render audio features: {e}")
    else:
        st.info("Please upload or record an audio sample in Tab 1 to inspect its waveform, mel-spectrogram, and MFCC heatmap.")

    st.markdown("</div>", unsafe_allow_html=True)

# =======================================================
# TAB 4: MODEL & DATASET INSIGHTS
# =======================================================
with tab_insights:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset & Machine Learning Architecture")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
        ### 🎭 RAVDESS Dataset Overview
        The **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** is a validated multimodal database of emotional speech.
        
        - **Total Actors**: 24 Professional Actors (12 Female, 12 Male)
        - **Target Emotions**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
        - **Audio Format**: 16-bit, 48kHz WAV audio
        """)

    with col_info2:
        st.markdown("""
        ### 🧪 Feature Extraction Pipeline
        Each audio waveform undergoes silence removal, amplitude normalization, and multi-domain feature extraction:
        
        1. **MFCCs** (40 coefficients - Mean & Std)
        2. **Chroma STFT** (12 pitch bins - Mean & Std)
        3. **Mel-Spectrogram** (128 mel frequency bands - Mean & Std)
        4. **Spectral Contrast** (7 frequency bands)
        5. **RMS Energy & Zero Crossing Rate (ZCR)**
        """)

    st.divider()

    st.subheader("🏆 Model Performance Metrics")
    m_col_svm, m_col_rf = st.columns(2)
    with m_col_svm:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Support Vector Machine (SVM)</h3>
            <div class="metric-value">{svm_acc:.1f}%</div>
            <p style="color:#CBD5E1 !important;">Kernel: RBF | C: 10.0 | Gamma: Scale</p>
        </div>
        """, unsafe_allow_html=True)

    with m_col_rf:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Random Forest Classifier</h3>
            <div class="metric-value">{rf_acc:.1f}%</div>
            <p style="color:#CBD5E1 !important;">Trees: 200 | Max Depth: 15</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("---")
st.caption(
    "🎤 Human Emotion Detection from Voice • Python • Librosa • Scikit-Learn • Streamlit • Developed by RNS Navadeep"
)
