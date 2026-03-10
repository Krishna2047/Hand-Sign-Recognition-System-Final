from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config.language_map import gesture_translation

import base64
import cv2
import numpy as np

from backend.database import save_conversation, get_history

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageData(BaseModel):
    image: str
    user: str
    language: str


@app.get("/")
def home():
    return {"message": "Hand Sign API running"}


@app.post("/predict")
def predict(data: ImageData):

    try:

        from backend.recognition.recognizer import recognize

        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gesture, confidence = recognize(frame)

        if gesture == "NONE":
            return {
                "gesture": "NONE",
                "text": "",
                "confidence": 0
            }

        text = gesture_translation[gesture].get(
            data.language,
            gesture_translation[gesture]["en"]
        )

        save_conversation(data.user, gesture, text, data.language)

        return {
            "gesture": gesture,
            "text": text,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:

        print("Prediction Error:", e)

        return {
            "gesture": "NONE",
            "text": "",
            "confidence": 0
        }


@app.get("/history/{user}")
def history(user: str):
    return get_history(user)