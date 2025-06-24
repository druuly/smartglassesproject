import streamlit as st

import subprocess
import sys
import importlib.util

# Uninstall GUI OpenCV (if installed accidentally by a sub-dependency like DeepFace)
if importlib.util.find_spec("cv2"):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])

from deepface import DeepFace
import speech_recognition as sr
import numpy as np
import cv2
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tempfile
import os

# Download VADER lexicon if not already
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

st.title("🧠 Emotion Detection (Facial & Speech)")

# --- Facial Emotion Section ---
st.header("📸 Facial Emotion Analysis")
img_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if img_file:
    img_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image")

    try:
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        st.write("Facial Emotion:", result[0]["dominant_emotion"])
    except Exception as e:
        st.error(f"Facial emotion analysis failed: {e}")

# --- Speech Emotion Section ---
st.header("🎤 Speech Emotion Analysis")
audio_file = st.file_uploader("Upload a short audio clip (WAV only)", type=["wav"])
if audio_file:
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        audio_path = tmp_file.name

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            score = sia.polarity_scores(text)
            comp = score["compound"]

            if comp > 0.3:
                emotion = "happy"
            elif comp < -0.3:
                emotion = "angry/sad"
            else:
                emotion = "neutral"

            st.write("Transcribed Text:", f"'{text}'")
            st.write("Speech Emotion:", emotion)
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")

    os.remove(audio_path)
