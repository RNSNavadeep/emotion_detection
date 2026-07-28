import os
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

from predict import predict_emotion
from audio_recorder_streamlit import audio_recorder

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Human Emotion Detection",
    page_icon="🎤",
    layout="wide"
)
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
# -------------------------------------------------------
# CREATE FOLDERS
# -------------------------------------------------------

os.makedirs("recordings", exist_ok=True)
os.makedirs("history", exist_ok=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    st.title("🤖 AI Emotion Detector")

    st.metric(
        label="Model Accuracy",
        value="72%"
    )

    st.divider()

    st.subheader("Dataset")
    st.info("RAVDESS")

    st.subheader("Algorithm")
    st.info("Support Vector Machine")

    st.subheader("Developer")
    st.success("RNS Navadeep")

    st.divider()

    st.subheader("Features")

    st.success("✔ Upload WAV Audio")

    st.success("✔ Live Voice Recording")

    st.success("✔ Emotion Prediction")

    st.success("✔ Confidence Score")

    st.success("✔ Probability Graph")

    st.success("✔ Prediction History")

    st.divider()

    st.caption("Version 1.0")

# -------------------------------------------------------
# CSS
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
background:#0E1117;
color:white;
}

.block-container{
padding-top:1rem;
padding-bottom:2rem;
}

.header{
background:#1C1F26;
padding:28px;
border-radius:20px;
text-align:center;
margin-bottom:30px;
border:1px solid #2f3542;
}

.header h1{
color:#4FC3F7;
font-size:46px;
margin-bottom:5px;
}

.header p{
font-size:18px;
color:#bdbdbd;
}

.card{
background:#1C1F26;
padding:25px;
border-radius:18px;
border:1px solid #2f3542;
margin-bottom:15px;
}

.result-card{
background:#222831;
padding:25px;
border-radius:18px;
text-align:center;
border:1px solid #3b4252;
}

.result-title{
font-size:18px;
color:#c5c5c5;
}

.result-value{
font-size:40px;
font-weight:bold;
color:#00E676;
margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""

<div class="header">

<h1>🎤 Human Emotion Detection from Voice</h1>

<p>
Speech Emotion Recognition using Machine Learning
</p>

</div>

""", unsafe_allow_html=True)

# -------------------------------------------------------
# AUDIO INPUT
# -------------------------------------------------------

tab1, tab2 = st.tabs(
[
"📁 Upload Audio",
"🎙 Record Voice"
]
)

# -------------------------------------------------------
# UPLOAD
# -------------------------------------------------------

with tab1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Upload Audio File")

    uploaded = st.file_uploader(
    "Choose Audio File",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
    )

    if uploaded:

        st.session_state.audio_path = "recordings/upload.wav"

        with open(st.session_state.audio_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(st.session_state.audio_path)

        st.success("Audio uploaded successfully.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# RECORD
# -------------------------------------------------------

with tab2:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Record Voice")

    audio_bytes = audio_recorder(
        pause_threshold=3.0,
        sample_rate=22050
    )

    if audio_bytes:

        st.session_state.audio_path = "recordings/live.wav"

        with open(st.session_state.audio_path, "wb") as f:
            f.write(audio_bytes)

        st.audio(st.session_state.audio_path)

        st.success("Recording saved successfully.")

    st.markdown("</div>", unsafe_allow_html=True)
# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------

if st.session_state.audio_path:

    st.write("")

    _, center, _ = st.columns([2, 4, 2])

    with center:

        predict = st.button(
            "🚀 Predict Emotion",
            use_container_width=True
        )

    if predict:

        try:

            emotion, confidence, prob = predict_emotion(st.session_state.audio_path)

        except Exception as e:

            st.error(f"Prediction Failed\n\n{e}")

            st.stop()

        emoji = {

            "happy": "😄",
            "sad": "😢",
            "angry": "😡",
            "fearful": "😨",
            "neutral": "😐",
            "surprised": "😲",
            "disgust": "🤢"

        }

        st.write("")

        left, right = st.columns(2)

        with left:

            st.markdown(f"""

            <div class="result-card">

            <div class="result-title">

            Predicted Emotion

            </div>

            <div class="result-value">

            {emoji.get(emotion,"🙂")} {emotion.upper()}

            </div>

            </div>

            """, unsafe_allow_html=True)

        with right:

            st.markdown(f"""

            <div class="result-card">

            <div class="result-title">

            Confidence

            </div>

            <div class="result-value">

            {confidence:.2f}%

            </div>

            </div>

            """, unsafe_allow_html=True)

        st.write("")

        st.subheader("Prediction Confidence")

        st.progress(confidence / 100)

        st.write(f"**Model Confidence:** {confidence:.2f}%")

        st.divider()
# -------------------------------------------------------
# EMOTION PROBABILITY CHART
# -------------------------------------------------------

        st.subheader("📊 Emotion Probability")

        df = pd.DataFrame({
            "Emotion": list(prob.keys()),
            "Probability": [p * 100 for p in prob.values()]
        })

        df = df.sort_values(
            by="Probability",
            ascending=False
        )

        fig = px.bar(
            df,
            x="Emotion",
            y="Probability",
            color="Probability",
            text_auto=".1f",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1C1F26",
            font_color="white",
            title="Emotion Prediction Probability",
            title_x=0.5,
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            coloraxis_showscale=False
        )

        fig.update_traces(
            marker_line_width=0
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# -------------------------------------------------------
# HISTORY
# -------------------------------------------------------

        history_file = "history/session.csv"

        if os.path.exists(history_file):

            history = pd.read_csv(history_file)

        else:

            history = pd.DataFrame(
                columns=[
                    "Time",
                    "Emotion",
                    "Confidence"
                ]
            )

        history.loc[len(history)] = [

            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            emotion,

            round(confidence, 2)

        ]

        history.to_csv(
            history_file,
            index=False
        )

        st.divider()

        with st.expander("📜 Prediction History", expanded=False):

            history = history.tail(10)

            st.dataframe(
                history,
                use_container_width=True
            )

            st.write("")

            if st.button("🗑 Clear History"):

                pd.DataFrame(
                    columns=[
                        "Time",
                        "Emotion",
                        "Confidence"
                    ]
                ).to_csv(
                    history_file,
                    index=False
                )

                st.success("History Cleared Successfully")

                st.rerun()

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.caption(
    "🎤 Human Emotion Detection from Voice | Python • Librosa • Scikit-Learn • Streamlit"
)
