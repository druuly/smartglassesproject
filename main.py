import cv2
from deepface import DeepFace
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import threading
import time
import nltk

# Download VADER lexicon once
nltk.download("vader_lexicon")

# Initialize webcam and face detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize recognizer and sentiment analyzer
recognizer = sr.Recognizer()
mic = sr.Microphone()
sia = SentimentIntensityAnalyzer()

# Globals to store current speech text and emotion
speech_text = ""
speech_emotion = ""

# Function to process speech and analyze emotion
def listen_for_speech():
    global speech_text, speech_emotion
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("🎤 Listening for speech...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                print(f"👂 Heard: {text}")
                score = sia.polarity_scores(text)
                comp = score["compound"]
                if comp > 0.3:
                    emotion = "happy"
                elif comp < -0.3:
                    emotion = "angry/sad"
                else:
                    emotion = "neutral"
                speech_text = text
                speech_emotion = emotion
        except Exception as e:
            print(f"Speech error: {e}")
            speech_text = ""
            speech_emotion = ""
        time.sleep(1)  # Slight delay between recordings

# Start the speech recognition thread
speech_thread = threading.Thread(target=listen_for_speech, daemon=True)
speech_thread.start()

print("🎥 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each face, detect and display emotion
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
            dominant_emotion = result[0]["dominant_emotion"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {dominant_emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        except Exception as e:
            print(f"Facial emotion error: {e}")

    # Display the latest transcribed speech and its emotion
    height = frame.shape[0]
    if speech_text:
        cv2.putText(frame, f"Speech: {speech_emotion}", (10, height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"'{speech_text}'", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("SmartGlasses Emotion Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
