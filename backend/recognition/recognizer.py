import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import threading

from config.language_map import gesture_translation


# =========================
# Model path
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sign_model.h5")

print("Loading gesture model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")


# =========================
# Classes (must match labels.txt)
# =========================
classes = [
    "HELLO",
    "YES",
    "NO",
    "HELP",
    "EMERGENCY",
    "THANK_YOU",
    "WATER",
    "FOOD"
]


# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Thread lock to prevent mediapipe crash
lock = threading.Lock()


# =========================
# Prediction smoothing
# =========================
last_prediction = "NONE"
stable_count = 0
STABLE_THRESHOLD = 3


# =========================
# Recognition
# =========================
def recognize(frame):

    global last_prediction
    global stable_count

    try:

        if frame is None:
            return "NONE", 0.0

        frame = cv2.flip(frame, 1)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with lock:
            result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            last_prediction = "NONE"
            stable_count = 0
            return "NONE", 0.0

        for handLms in result.multi_hand_landmarks:

            landmarks = []

            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            data = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(data, verbose=0)

            class_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            if class_id >= len(classes):
                return "NONE", 0.0

            gesture = classes[class_id]

            # =========================
            # Flicker smoothing
            # =========================
            if gesture == last_prediction:
                stable_count += 1
            else:
                stable_count = 0
                last_prediction = gesture

            if stable_count < STABLE_THRESHOLD:
                return "NONE", 0.0

            return gesture, confidence

        return "NONE", 0.0

    except Exception as e:

        print("Recognition error:", e)

        return "NONE", 0.0