import cv2
import numpy as np
import mediapipe as mp
import time
import os
import threading

from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from playsound import playsound

from config.language_map import gesture_translation

# =========================
# Load trained model
# =========================
model = load_model("model/sign_model.h5")
classes = list(gesture_translation.keys())

# =========================
# MediaPipe Hands (Optimized)
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # 🔥 Faster
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# =========================
# Language + timing
# =========================
language = "en"
last_spoken_text = ""
last_time = 0
delay = 2  # seconds

# =========================
# Unicode Fonts
# =========================
fonts = {
    "en": ImageFont.truetype("font/NotoSansTamil-Regular.ttf", 40),
    "ta": ImageFont.truetype("font/NotoSansTamil-Regular.ttf", 40),
    "te": ImageFont.truetype("font/NotoSansTelugu-Regular.ttf", 40),
    "ml": ImageFont.truetype("font/NotoSansMalayalam-Regular.ttf", 40),
}

# =========================
# Non-Blocking Speech
# =========================
def speak_async(text, lang):
    def run():
        filename = f"voice_{lang}.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        playsound(filename)
        os.remove(filename)

    threading.Thread(target=run, daemon=True).start()


# =========================
# Camera (Lower Resolution = Faster)
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 1-English | 2-Tamil | 3-Telugu | 4-Malayalam | Q-Quit")

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror
    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            # -------- 21 joint points only --------
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            data = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(data, verbose=0)
            class_id = np.argmax(prediction)
            gesture = classes[class_id]
            text = gesture_translation[gesture][language]

            # -------- Unicode text drawing --------
            frame_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_pil)
            draw.text((30, 30), text, font=fonts[language], fill=(0, 255, 0))
            frame = np.array(frame_pil)

            # -------- Speak only if changed --------
            current_time = time.time()
            if text != last_spoken_text and current_time - last_time > delay:
                speak_async(text, language)
                last_spoken_text = text
                last_time = current_time

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # -------- Controls --------
    cv2.putText(
        frame,
        "1-EN  2-TA  3-TE  4-ML   Q-Quit",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Hand Sign Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        language = "en"
    elif key == ord('2'):
        language = "ta"
    elif key == ord('3'):
        language = "te"
    elif key == ord('4'):
        language = "ml"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
