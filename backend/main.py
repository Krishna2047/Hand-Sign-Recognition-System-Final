from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
from config.language_map import gesture_translation
from backend.database import save_conversation, get_history

from gtts import gTTS

import base64
import cv2
import numpy as np
import uuid
import os


app = FastAPI()

# =========================
# Allow React frontend
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Static folder for audio
# =========================
STATIC_DIR = "backend/static"

os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================
# Request model
# =========================
class ImageData(BaseModel):
    image: str
    user: str
    language: str


# =========================
# Generate Google TTS voice
# =========================
def generate_voice(text, lang):

    try:

        filename = f"voice_{uuid.uuid4()}.mp3"

        filepath = os.path.join(STATIC_DIR, filename)

        tts = gTTS(text=text, lang=lang)

        tts.save(filepath)

        return filename

    except Exception as e:

        print("TTS error:", e)

        return None

def cleanup_static():

    folder = "backend/static"

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print("Cleanup error:", e)

# =========================
# API Home
# =========================
@app.get("/")
def home():
    return {"message": "Hand Sign API running"}


# =========================
# Gesture prediction
# =========================
@app.post("/predict")
def predict(data: ImageData):

    try:

        # lazy import (prevents startup delay)
        from backend.recognition.recognizer import recognize

        # decode base64 image
        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "gesture": "NONE",
                "text": "",
                "confidence": 0,
                "audio": None
            }

        # run AI recognition
        gesture, confidence = recognize(frame)

        if gesture == "NONE":

            return {
                "gesture": "NONE",
                "text": "",
                "confidence": 0,
                "audio": None
            }

        # translate language
        text = gesture_translation[gesture].get(
            data.language,
            gesture_translation[gesture]["en"]
        )

        # generate voice
        audio_file = generate_voice(text, data.language)

        # save conversation
        save_conversation(data.user, gesture, text, data.language)

        return {
            "gesture": gesture,
            "text": text,
            "confidence": round(confidence * 100, 2),
            "audio": audio_file
        }

    except Exception as e:

        print("Prediction Error:", e)

        return {
            "gesture": "NONE",
            "text": "",
            "confidence": 0,
            "audio": None
        }


# =========================
# Conversation history
# =========================
@app.get("/history/{user}")
def history(user: str):
    return get_history(user)

@app.on_event("shutdown")
def shutdown_event():
    print("Cleaning audio files...")
    cleanup_static()