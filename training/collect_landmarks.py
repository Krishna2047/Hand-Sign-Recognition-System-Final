import cv2
import mediapipe as mp
import pandas as pd
import os

GESTURE = "FOOD"   # CHANGE THIS EVERY TIME
SAMPLES = 300       # 300 samples per gesture

os.makedirs("../dataset", exist_ok=True)
csv_path = f"dataset/{GESTURE}.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1),
cap = cv2.VideoCapture(0)

data = []
count = 0

print("Press 's' to start capturing")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                data.append(landmarks)
                count += 1
                print(f"Captured {count}/{SAMPLES}")

    cv2.imshow("Capture Hand Sign", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
