import cv2
import os

# CHANGE THIS TO ONE CLASS AT A TIME
GESTURE_NAME = "YES"   # <-- change later

SAVE_PATH = os.path.join("../dataset", GESTURE_NAME)
os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)
    cv2.putText(frame, f"Images: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        img = frame[100:400, 100:400]
        img_path = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(img_path, img)
        count += 1
        print(f"Saved {img_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
